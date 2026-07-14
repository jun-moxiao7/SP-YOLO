import warnings
from typing import Sequence, Union

import torch
import torch.nn as nn

try:
    # Newer timm versions
    from timm.layers import DropPath, trunc_normal_
except ImportError:
    # Compatibility with older timm versions
    from timm.models.layers import DropPath, trunc_normal_

# Use the official Ultralytics Conv module.
# Its Conv + BN layers can be recognized by BaseModel.fuse().
from ultralytics.nn.modules.block import Conv

__all__ = [
    "starnet_s050",
    "starnet_s100",
    "starnet_s150",
    "starnet_s1",
    "starnet_s2",
    "starnet_s3",
    "starnet_s4",
]

model_urls = {
    "starnet_s1": (
        "https://github.com/ma-xu/Rewrite-the-Stars/"
        "releases/download/checkpoints_v1/starnet_s1.pth.tar"
    ),
    "starnet_s2": (
        "https://github.com/ma-xu/Rewrite-the-Stars/"
        "releases/download/checkpoints_v1/starnet_s2.pth.tar"
    ),
    "starnet_s3": (
        "https://github.com/ma-xu/Rewrite-the-Stars/"
        "releases/download/checkpoints_v1/starnet_s3.pth.tar"
    ),
    "starnet_s4": (
        "https://github.com/ma-xu/Rewrite-the-Stars/"
        "releases/download/checkpoints_v1/starnet_s4.pth.tar"
    ),
}


class ConvNoBN(nn.Module):
    """
    Convolution without BatchNorm or activation.
    A wrapper is used instead of a bare nn.Conv2d so that parameter names
    remain similar to the original implementation, e.g. f1.conv.weight.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _to_stage_list(
    value: Union[int, float, Sequence],
    num_stages: int,
    name: str,
):
    if isinstance(value, (int, float)):
        return [value] * num_stages

    value = list(value)
    if len(value) != num_stages:
        raise ValueError(
            f"{name} must contain {num_stages} values, but got {len(value)}: {value}"
        )
    return value


class Block(nn.Module):
    """StarNet Block executing element-wise star product operation."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 3.0,
        kernel_size: int = 7,
        drop_path: float = 0.0,
    ):
        super().__init__()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, but got {kernel_size}."
            )

        hidden_dim = int(round(dim * mlp_ratio))
        if hidden_dim <= 0:
            raise ValueError(
                f"Invalid hidden dimension {hidden_dim} for dim={dim}, mlp_ratio={mlp_ratio}."
            )

        padding = kernel_size // 2

        # Official Ultralytics Conv: depthwise convolution + BN + Identity activation.
        self.dwconv = Conv(
            c1=dim, c2=dim, k=kernel_size, s=1, p=padding, g=dim, act=False
        )

        # These two branches intentionally contain no BatchNorm.
        self.f1 = ConvNoBN(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.f2 = ConvNoBN(
            in_channels=dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Pointwise projection with BatchNorm and no activation.
        self.g = Conv(c1=hidden_dim, c2=dim, k=1, s=1, p=0, act=False)

        # The second depthwise convolution intentionally has no BN.
        self.dwconv2 = ConvNoBN(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=dim,
            bias=True,
        )

        self.act = nn.ReLU6(inplace=True)
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.dwconv(x)
        branch_1 = self.f1(x)
        branch_2 = self.f2(x)
        x = self.act(branch_1) * branch_2
        x = self.g(x)
        x = self.dwconv2(x)

        return identity + self.drop_path(x)


class StarNet(nn.Module):
    """
    StarNet: Rewriting the Stars in Lightweight Backbones.
    Tailored for efficient deployment on resource-limited edge hardware.
    """

    def __init__(
        self,
        base_dim: int = 32,
        depths: Sequence[int] = (3, 3, 12, 5),
        mlp_ratios: Union[float, Sequence[float]] = 4.0,
        kernel_sizes: Union[int, Sequence[int]] = 7,
        drop_path_rate: float = 0.0,
        stem_channels: int = 32,
        num_classes: int = 1000,
        **kwargs,
    ):
        super().__init__()
        del kwargs

        self.num_classes = num_classes
        self.num_stages = len(depths)

        if self.num_stages != 4:
            raise ValueError(
                f"The current detection integration expects four StarNet stages, but got {self.num_stages}."
            )

        self.depths = list(depths)
        self.mlp_ratios = _to_stage_list(mlp_ratios, self.num_stages, "mlp_ratios")
        self.kernel_sizes = _to_stage_list(
            kernel_sizes, self.num_stages, "kernel_sizes"
        )

        for kernel_size in self.kernel_sizes:
            if (
                not isinstance(kernel_size, int)
                or kernel_size <= 0
                or kernel_size % 2 == 0
            ):
                raise ValueError(
                    f"Every kernel size must be a positive odd integer, but got {self.kernel_sizes}."
                )

        self.in_channel = stem_channels

        # Stem branch: 3x3 stride-2 Conv + BN + ReLU6
        self.stem = nn.Sequential(
            Conv(c1=3, c2=stem_channels, k=3, s=2, p=1, act=False),
            nn.ReLU6(inplace=True),
        )

        total_blocks = sum(self.depths)
        drop_path_rates = [
            value.item()
            for value in torch.linspace(0, drop_path_rate, total_blocks)
        ]

        self.stages = nn.ModuleList()
        current_block = 0

        for stage_index, stage_depth in enumerate(self.depths):
            embed_dim = base_dim * (2**stage_index)

            down_sampler = Conv(
                c1=self.in_channel, c2=embed_dim, k=3, s=2, p=1, act=False
            )
            self.in_channel = embed_dim

            blocks = [
                Block(
                    dim=embed_dim,
                    mlp_ratio=self.mlp_ratios[stage_index],
                    kernel_size=self.kernel_sizes[stage_index],
                    drop_path=drop_path_rates[current_block + block_index],
                )
                for block_index in range(stage_depth)
            ]
            current_block += stage_depth

            self.stages.append(nn.Sequential(down_sampler, *blocks))

        self.channel = [
            stem_channels,
            *[base_dim * (2**stage_index) for stage_index in range(self.num_stages)],
        ]

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor):
        features = []
        x = self.stem(x)
        features.append(x)

        for stage in self.stages:
            x = stage(x)
            features.append(x)

        return features


def _load_pretrained(model: nn.Module, model_name: str):
    if model_name not in model_urls:
        raise ValueError(
            f"No official pretrained checkpoint is configured for {model_name}."
        )

    checkpoint = torch.hub.load_state_dict_from_url(
        url=model_urls[model_name], map_location="cpu", check_hash=False
    )
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Remove DistributedDataParallel prefix safely
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, strict=False
    )

    if missing_keys:
        warnings.warn(
            f"{model_name}: missing pretrained keys: {missing_keys}",
            stacklevel=2,
        )
    if unexpected_keys:
        warnings.warn(
            f"{model_name}: unexpected pretrained keys: {unexpected_keys}",
            stacklevel=2,
        )
    return model


# --- Model Variant Registry ---


def starnet_s050(pretrained: bool = False, **kwargs):
    if pretrained:
        warnings.warn(
            "No official pretrained checkpoint is configured for starnet_s050. "
            "The model will be initialized from scratch.",
            stacklevel=2,
        )
    return StarNet(
        base_dim=16,
        depths=(1, 1, 3, 1),
        mlp_ratios=(3, 3, 3, 3),
        kernel_sizes=(5, 5, 7, 7),
        stem_channels=32,
        **kwargs,
    )


def starnet_s100(pretrained: bool = False, **kwargs):
    if pretrained:
        warnings.warn(
            "No official pretrained checkpoint is configured for starnet_s100. "
            "The model will be initialized from scratch.",
            stacklevel=2,
        )
    return StarNet(
        base_dim=20,
        depths=(1, 2, 4, 1),
        mlp_ratios=(4, 4, 4, 4),
        kernel_sizes=(7, 7, 7, 7),
        stem_channels=32,
        **kwargs,
    )


def starnet_s150(pretrained: bool = False, **kwargs):
    if pretrained:
        warnings.warn(
            "No official pretrained checkpoint is configured for starnet_s150. "
            "The model will be initialized from scratch.",
            stacklevel=2,
        )
    return StarNet(
        base_dim=24,
        depths=(1, 2, 4, 2),
        mlp_ratios=(3, 3, 3, 3),
        kernel_sizes=(7, 7, 7, 7),
        stem_channels=32,
        **kwargs,
    )


def starnet_s1(pretrained: bool = False, **kwargs):
    model = StarNet(
        base_dim=24,
        depths=(2, 2, 8, 3),
        mlp_ratios=(4, 4, 4, 4),
        kernel_sizes=(7, 7, 7, 7),
        stem_channels=32,
        **kwargs,
    )
    return _load_pretrained(model, "starnet_s1") if pretrained else model


def starnet_s2(pretrained: bool = False, **kwargs):
    model = StarNet(
        base_dim=32,
        depths=(1, 2, 6, 2),
        mlp_ratios=(4, 4, 4, 4),
        kernel_sizes=(7, 7, 7, 7),
        stem_channels=32,
        **kwargs,
    )
    return _load_pretrained(model, "starnet_s2") if pretrained else model


def starnet_s3(pretrained: bool = False, **kwargs):
    model = StarNet(
        base_dim=32,
        depths=(2, 2, 8, 4),
        mlp_ratios=(4, 4, 4, 4),
        kernel_sizes=(7, 7, 7, 7),
        stem_channels=32,
        **kwargs,
    )
    return _load_pretrained(model, "starnet_s3") if pretrained else model


def starnet_s4(pretrained: bool = False, **kwargs):
    model = StarNet(
        base_dim=32,
        depths=(3, 3, 12, 5),
        mlp_ratios=(4, 4, 4, 4),
        kernel_sizes=(7, 7, 7, 7),
        stem_channels=32,
        **kwargs,
    )
    return _load_pretrained(model, "starnet_s4") if pretrained else model
