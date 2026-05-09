from ultralytics import YOLO

model = YOLO('runs/detect/sperm_detection/weights/best.pt')
model.export(format='onnx', imgsz=640)
print("模型已导出为 best.onnx")
