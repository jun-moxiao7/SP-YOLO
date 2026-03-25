import sys
import argparse
import os

# https://github.com/Dao-AILab/flash-attention 环境安装


from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    weights = opt.weights
    model = YOLO(weights)  # 直接加载权重文件进行训练
    # model = YOLO(yaml) # 加载自定义or默认的yaml配置文件
    # model = YOLO(yaml).load(weights) # 加载yaml配置文件的同时，加载权重进行训练

    model.info()

    results = model.train(data='coco128.yaml',  # 数据集yaml路径
                        epochs=300, 
                        imgsz=640, 
                        workers=8, 
                        batch=8,
                        # 所有参数均可以自行设置
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='ultralytics/cfg/models/v12/yolov12.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='yolov12n.pt', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)