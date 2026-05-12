from ultralytics import YOLO
import cv2

def train():
    config_content = """
path: yolo_dataset
train: images/train
val: images/val

nc: 2
names: ['white', 'pink']
"""
    
    with open('sperm_data.yaml', 'w') as f:
        f.write(config_content)
    
    model = YOLO('yolov8s.pt')
    
    print("Starting training with optimized parameters...")
    results = model.train(
        data='sperm_data.yaml',
        epochs=100,          # Increase to 100 epochs
        batch=16,            # Increase batch size
        imgsz=800,           # Increase image size for better small object detection
        device='cpu',
        workers=2,
        name='sperm_detection_v2',
        patience=150,        # Increase patience
        lr0=0.001,           # Lower initial learning rate
        lrf=0.0001,          # Lower final learning rate
        momentum=0.937,      # Keep momentum
        weight_decay=0.0005, # Weight decay
        warmup_epochs=5,     # Increase warmup epochs
        box=7.5,             # box loss weight
        cls=0.5,             # cls loss weight
        dfl=1.5,             # dfl loss weight
        hsv_h=0.015,         # HSV-H augmentation
        hsv_s=0.7,           # HSV-S augmentation
        hsv_v=0.4,           # HSV-V augmentation
        flipud=0.1,          # Flip up-down probability
        fliplr=0.5,          # Flip left-right probability
        mosaic=1.0,          # mosaic augmentation
        mixup=0.1,           # mixup augmentation
        copy_paste=0.1,      # copy-paste augmentation
        auto_augment='randaugment',  # automatic augmentation
        erasing=0.4,         # random erasing
        close_mosaic=15,     # Close mosaic in last 15 epochs
        amp=True,            # Automatic mixed precision
        verbose=True
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    print(f"ONNX export: python export_onnx.py")
    print("="*50)

def predict(image_path):
    # Load trained model
    model = YOLO('runs/detect/sperm_detection_v2/weights/best.pt')
    
    # Predict
    results = model(image_path)
    
    # Count results
    white_count = 0
    pink_count = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 0:
                white_count += 1
            else:
                pink_count += 1
    
    print("=" * 40)
    print("Sperm Staining Status Statistics")
    print("=" * 40)
    print(f"Live sperm (white): {white_count}")
    print(f"Fragmented sperm (pink): {pink_count}")
    print(f"Total sperm count: {white_count + pink_count}")
    if white_count + pink_count > 0:
        survival_rate = (white_count / (white_count + pink_count)) * 100
        print(f"Survival rate: {survival_rate:.2f}%")
    print("=" * 40)
    
    # Save result image
    result_image = results[0].plot()
    output_path = 'prediction_result.jpg'
    cv2.imwrite(output_path, result_image)
    print(f"Result image saved to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='YOLO sperm detection training/prediction')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--predict', type=str, help='Predict on single image')
    args = parser.parse_args()
    
    if args.train:
        train()
    elif args.predict:
        predict(args.predict)
    else:
        print("Usage: python train_yolo_v2.py --train or python train_yolo_v2.py --predict <image_path>")
