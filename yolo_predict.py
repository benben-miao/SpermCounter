from ultralytics import YOLO
import cv2
import argparse
import os

def predict(image_path, conf_threshold=0.1, iou_threshold=0.45):
    # 加载训练好的模型
    model_path = 'runs/detect/sperm_detection-2/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        print("请先运行训练脚本：python train_yolo.py --train")
        return None
    
    model = YOLO(model_path)
    
    # 预测（降低置信度阈值以检测更多物体）
    results = model(image_path, conf=conf_threshold, iou=iou_threshold)
    
    # 统计结果
    white_count = 0
    pink_count = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0:
                white_count += 1
            else:
                pink_count += 1
    
    return {
        'white_sperm': white_count,
        'pink_sperm': pink_count,
        'total': white_count + pink_count,
        'result_image': results[0].plot()
    }

def main():
    parser = argparse.ArgumentParser(description='YOLO精子检测 (最终版)')
    parser.add_argument('image_path', help='输入图片的路径')
    parser.add_argument('--conf', type=float, default=0.1, help='置信度阈值 (默认: 0.1)')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU阈值 (默认: 0.45)')
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"错误：文件不存在 {args.image_path}")
        return
    
    result = predict(args.image_path, conf_threshold=args.conf, iou_threshold=args.iou)
    
    if result is not None:
        print("=" * 40)
        print("精子染色状态统计结果 (YOLO)")
        print("=" * 40)
        print(f"置信度阈值: {args.conf}")
        print(f"存活精子（白色）: {result['white_sperm']}")
        print(f"破碎精子（粉红色）: {result['pink_sperm']}")
        print(f"精子总数: {result['total']}")
        if result['total'] > 0:
            survival_rate = (result['white_sperm'] / result['total']) * 100
            print(f"存活率: {survival_rate:.2f}%")
        print("=" * 40)
        
        # 保存结果图像
        output_path = 'yolo_result.jpg'
        cv2.imwrite(output_path, result['result_image'])
        print(f"检测结果图像已保存到 {output_path}")

if __name__ == '__main__':
    main()
