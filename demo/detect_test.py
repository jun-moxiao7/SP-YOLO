import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('../ultralytics/cfg/models/11/SP-YOLO.yaml')
    model.train(
        data=r'corn_disease.yaml',
        epochs=300,
        imgsz=640,
        batch=85,
        cache=True,
        workers=9,
        lr0=0.01,
        cos_lr=True,
        mixup=0.15,
        augment=True,
        # 正则化增强
        weight_decay=0.0005,
        label_smoothing=0.1,
        close_mosaic=10,      # 最后10epoch关闭mosaic
        overlap_mask=True,
        mask_ratio=4
    )
    model.info()
