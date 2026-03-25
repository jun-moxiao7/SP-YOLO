import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO
import time

model = YOLO('../ultralytics/cfg/models/11/SP-YOLO.yaml')

corn_img = model('corn.jpg')

times = []
for corn_img in range(100):
    start_time = time.time()
    corn_img = model('corn.jpg')
    end_time = time.time()
    times.append(end_time - start_time)

fps = 1 / (sum(times) / len(times))
print(f'平均FPS: {fps:.2f}')
