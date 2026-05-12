from ultralytics import YOLO

# Export newly trained YOLOv8s model
model = YOLO('runs/detect/sperm_detection/weights/best.pt')
model.export(format='onnx', imgsz=640)
print("Model exported as best.onnx")
