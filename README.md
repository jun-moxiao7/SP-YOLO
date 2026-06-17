# Lightweight visual computing framework for maize leaf disease targeting mobile devices

## 📖 Introduction
**SP-YOLO** is a lightweight visual computing framework derived from YOLOv11n, explicitly tailored for edge deployment with systematic architectural optimizations. It achieves a strong efficiency-accuracy balance, reducing parameters by 52.3% and FLOPs by 56.1% compared to the baseline, while improving mAP@0.5 to 98.1%.
Dataset DOI: https://doi.org/10.5281/zenodo.20129619

## ✨ Key Algorithmic Implementations
Our systematic efficiency-improving pipeline consists of three core architectural innovations:
* **StarNet-based Backbone:** Replaces the standard backbone to extract compact, disease-sensitive texture features by mapping inputs into a high-dimensional, non-linear space using "star operations" (element-wise multiplication) without widening the network.
* **C3k2_PConv Neck:** Introduces partial convolutions (PConv) into the first three fusion modules. By applying spatial convolution to only a subset of channels, it drastically reduces redundant spatial computation and memory access costs (MAC) during high-resolution multi-scale feature aggregation.
* **Detect_PCDHead:** A novel PConv-based decoupled detection head that streamlines both bounding-box regression and classification processes while retaining the high localization precision of Distribution Focal Loss (DFL).

## ⚙️ Quick Start
To allow users to effortlessly evaluate SP-YOLO, we provide a ready-to-use inference demo:

```text
SP-YOLO/
├── assets/
│   ├── img_1.jpg            # 5 sample images covering maize leaf diseases
│   ├── img_2.jpg
│   └── ...
└── demo/
    ├── best.pt              # The optimal pre-trained weights of SP-YOLO
    └── predict_demo.py      # The inference script
