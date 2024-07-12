from cam_video_parse import CAM_AREAS, image_crop

from ultralytics import YOLO

import os
import cv2
import numpy as np

# Загрузка обученной модели
model = YOLO("best.pt")



# Распознавание объекта на изображении
image = image_crop('test3.jpg', '248')
predictions = model.predict(image, conf=0.55)
for predict in predictions:
    predict.show()
