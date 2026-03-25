import sys
import argparse
import os

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg

    model = YOLO('yolov12n.pt')

    model.predict(r'ultralytics/assets/zidane.jpg', save=True, imgsz=640, conf=0.5, save_txt=True, save_conf=True)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'yolov12n.pt', help='initial weights path')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)