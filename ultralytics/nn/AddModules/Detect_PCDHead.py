import math
import torch
import torch.nn as nn

from ultralytics.nn.modules.block import Conv, DFL
from ultralytics.nn.AddModules.C3k2_PConv import Partial_conv3
from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = ["Detect_PCDHead"]


class Detect_PCDHead(nn.Module):
    """
    Detect_PCDHead deployed for resource-constrained edge devices.
    Incorporates Partial_conv3 to reduce GFLOPs and MACs.
    """

    dynamic = False
    export = False
    format = None
    shape = None

    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc: int = 80, ch=()):
        super().__init__()

        if not ch:
            raise ValueError("Detect_PCDHead requires a non-empty input-channel list.")

        self.nc = nc  # Number of classes
        self.nl = len(ch)  # Number of detection layers
        self.reg_max = 16  # DFL resolution
        self.no = nc + self.reg_max * 4  # Number of outputs per anchor

        self.stride = torch.zeros(self.nl)
        self.shared_channels = max(64, min(int(ch[0]), 96))

        # 1. Shared input projection layer
        self.shared_stem = nn.ModuleList(
            Conv(c_in, self.shared_channels, 1, 1) for c_in in ch
        )

        # 2. Decoupled Bounding-box regression branch
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Partial_conv3(self.shared_channels, n_div=4, forward="split_cat"),
                Conv(self.shared_channels, self.shared_channels, 1, 1),
                nn.Conv2d(
                    self.shared_channels,
                    4 * self.reg_max,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
            for _ in ch
        )

        # 3. Decoupled Classification branch
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Partial_conv3(self.shared_channels, n_div=4, forward="split_cat"),
                Conv(self.shared_channels, self.shared_channels, 1, 1),
                nn.Conv2d(
                    self.shared_channels,
                    self.nc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ),
            )
            for _ in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        shape = x[0].shape
        outputs = []

        for i in range(self.nl):
            # Shared low-level channel projection
            shared_feature = self.shared_stem[i](x[i])

            # Decoupled task-specific feature extraction
            box_prediction = self.cv2[i](shared_feature)
            cls_prediction = self.cv3[i](shared_feature)

            outputs.append(torch.cat((box_prediction, cls_prediction), dim=1))

        if self.training:
            return outputs

        # Dynamic anchor generation during inference
        if self.dynamic or self.shape != shape:
            anchor_points, stride_tensor = make_anchors(outputs, self.stride, 0.5)
            self.anchors = anchor_points.transpose(0, 1)
            self.strides = stride_tensor.transpose(0, 1)
            self.shape = shape

        x_cat = torch.cat(
            [output.view(shape[0], self.no, -1) for output in outputs], dim=2
        )

        # Channel splitting logic for box and class features
        if self.export and self.format in (
            "saved_model",
            "pb",
            "tflite",
            "edgetpu",
            "tfjs",
        ):
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)

        # Bounding box coordinates regression decoding
        decoded_box = (
            dist2bbox(
                self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1
            )
            * self.strides
        )

        # Normalize bounding box for mobile deployment formats
        if self.export and self.format in ("tflite", "edgetpu"):
            image_height = shape[2] * self.stride[0]
            image_width = shape[3] * self.stride[0]
            image_size = torch.tensor(
                [image_width, image_height, image_width, image_height],
                device=decoded_box.device,
                dtype=decoded_box.dtype,
            ).reshape(1, 4, 1)
            decoded_box = decoded_box / image_size

        prediction = torch.cat((decoded_box, cls.sigmoid()), dim=1)

        return prediction if self.export else (prediction, outputs)

    def bias_init(self):
        """Initialize biases for regression and classification layers."""
        for reg_branch, cls_branch, stride in zip(
            self.cv2, self.cv3, self.stride
        ):
            # Regression bias initialization
            reg_branch[-1].bias.data.fill_(1.0)

            # Classification bias initialization (Safely handled via .fill_ to guarantee convergence)
            cls_prior = math.log(5 / self.nc / (640 / stride) ** 2)
            cls_branch[-1].bias.data.fill_(cls_prior)
