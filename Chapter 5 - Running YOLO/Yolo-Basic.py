from ultralytics import YOLO
import cv2

# model = YOLO('../Yolo-Weights/yolov8n.pt') # Nano -- Faster
model = YOLO('../Yolo-Weights/yolov8l.pt') # Larger -- Lower
results = model("Images/Yolo-2.png", show=True)
cv2.waitKey(0)