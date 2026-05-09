from ultralytics import YOLO
import os

def train():
    # 创建配置文件
    config_content = """
path: yolo_dataset
train: images/train
val: images/val

nc: 2
names: ['white', 'pink']
"""
    
    with open('sperm_data.yaml', 'w') as f:
        f.write(config_content)
    
    # 加载YOLOv8n模型
    model = YOLO('yolov8n.pt')
    
    # 训练模型
    results = model.train(
        data='sperm_data.yaml',
        epochs=50,
        batch=8,
        imgsz=640,
        device='cpu',
        workers=2,
        name='sperm_detection'
    )
    
    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}")

def predict(image_path):
    # 加载训练好的模型
    model = YOLO('runs/detect/sperm_detection/weights/best.pt')
    
    # 预测
    results = model(image_path)
    
    # 统计结果
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
    print("精子染色状态统计结果")
    print("=" * 40)
    print(f"存活精子（白色）: {white_count}")
    print(f"破碎精子（粉红色）: {pink_count}")
    print(f"精子总数: {white_count + pink_count}")
    if white_count + pink_count > 0:
        survival_rate = (white_count / (white_count + pink_count)) * 100
        print(f"存活率: {survival_rate:.2f}%")
    print("=" * 40)
    
    # 保存结果图像
    result_image = results[0].plot()
    output_path = 'prediction_result.jpg'
    cv2.imwrite(output_path, result_image)
    print(f"Result image saved to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='YOLO精子检测训练/预测')
    parser.add_argument('--train', action='store_true', help='训练模型')
    parser.add_argument('--predict', type=str, help='预测单张图片')
    args = parser.parse_args()
    
    if args.train:
        train()
    elif args.predict:
        import cv2
        predict(args.predict)
    else:
        print("Usage: python train_yolo.py --train or python train_yolo.py --predict <image_path>")
