
from ultralytics import YOLO

model = YOLO("yolov8n.pt")



results = model.train(
   data='data.yaml',
   epochs=10,
   batch=8,
   name='yolov8n_test'
)


