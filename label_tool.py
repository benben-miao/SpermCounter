import cv2
import numpy as np
import os
import json

# 全局变量
image = None
current_image_index = 0
images = []
labels = {}  # 存储标注数据
current_label = 0  # 0=white, 1=pink
points = []
drawing = False

def load_images(folder_path):
    global images, current_image_index, labels, image
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()
    
    # 加载已有的标注
    if os.path.exists('labels.json'):
        with open('labels.json', 'r') as f:
            labels = json.load(f)
    
    if images:
        current_image_index = 0
        load_image()

def load_image():
    global image
    if images:
        img_path = os.path.join('photos', images[current_image_index])
        image = cv2.imread(img_path)
        draw_labels()

def draw_labels():
    if image is None:
        return
    
    img_copy = image.copy()
    img_name = images[current_image_index]
    
    # 绘制已标注的精子
    if img_name in labels:
        for label in labels[img_name]:
            x1, y1, x2, y2, cls = label
            color = (0, 255, 0) if cls == 0 else (0, 0, 255)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            label_text = 'White' if cls == 0 else 'Pink'
            cv2.putText(img_copy, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 绘制当前正在标注的矩形
    if len(points) == 2:
        cv2.rectangle(img_copy, points[0], points[1], (255, 0, 0), 2)
    
    # 添加提示
    cv2.putText(img_copy, f"Image: {current_image_index+1}/{len(images)}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img_copy, f"Current Label: {'White' if current_label == 0 else 'Pink'}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if current_label == 0 else (0, 0, 255), 2)
    cv2.putText(img_copy, "Instructions:", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_copy, "  Drag to draw box", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img_copy, "  W/P: Switch label", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img_copy, "  D: Delete last label", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img_copy, "  N: Next image", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img_copy, "  P: Previous image", (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img_copy, "  S: Save labels", (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img_copy, "  Q: Quit", (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow('Annotation Tool', img_copy)

def mouse_callback(event, x, y, flags, param):
    global points, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        points = [points[0], (x, y)]
        draw_labels()
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            # 确保坐标正确
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            img_name = images[current_image_index]
            if img_name not in labels:
                labels[img_name] = []
            labels[img_name].append([x1, y1, x2, y2, current_label])
            print(f"Added label: {'White' if current_label == 0 else 'Pink'} at ({x1},{y1})-({x2},{y2})")
        points = []
        draw_labels()

def save_labels():
    with open('labels.json', 'w') as f:
        json.dump(labels, f, indent=2)
    print("Labels saved to labels.json")

def convert_to_yolo_format():
    # 创建YOLO格式的目录结构
    os.makedirs('yolo_dataset/images/train', exist_ok=True)
    os.makedirs('yolo_dataset/images/val', exist_ok=True)
    os.makedirs('yolo_dataset/labels/train', exist_ok=True)
    os.makedirs('yolo_dataset/labels/val', exist_ok=True)
    
    # 读取标注
    if not os.path.exists('labels.json'):
        print("No labels.json found!")
        return
    
    with open('labels.json', 'r') as f:
        labels_data = json.load(f)
    
    # 分割训练/验证集 (80/20)
    img_names = list(labels_data.keys())
    split_idx = int(len(img_names) * 0.8)
    train_names = img_names[:split_idx]
    val_names = img_names[split_idx:]
    
    # 类别映射
    class_map = {'white': 0, 'pink': 1}
    
    for img_name in img_names:
        # 读取图片获取尺寸
        img_path = os.path.join('photos', img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        height, width = img.shape[:2]
        img_labels = labels_data[img_name]
        
        # 生成YOLO格式的标注文件
        label_lines = []
        for label in img_labels:
            x1, y1, x2, y2, cls = label
            # 转换为YOLO格式 (中心坐标, 宽高, 归一化)
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
        # 复制图片和标签到对应目录
        if img_name in train_names:
            img_dst = os.path.join('yolo_dataset/images/train', img_name)
            label_dst = os.path.join('yolo_dataset/labels/train', os.path.splitext(img_name)[0] + '.txt')
        else:
            img_dst = os.path.join('yolo_dataset/images/val', img_name)
            label_dst = os.path.join('yolo_dataset/labels/val', os.path.splitext(img_name)[0] + '.txt')
        
        cv2.imwrite(img_dst, img)
        with open(label_dst, 'w') as f:
            f.write('\n'.join(label_lines))
    
    print(f"Converted {len(train_names)} training images and {len(val_names)} validation images")

def main():
    global current_label, current_image_index
    
    # 加载图片
    load_images('photos')
    
    if not images:
        print("No images found in photos folder!")
        return
    
    # 创建窗口和鼠标回调
    cv2.namedWindow('Annotation Tool', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Annotation Tool', mouse_callback)
    
    draw_labels()
    
    print("Annotation Tool started!")
    print("Instructions:")
    print("  Drag to draw box around sperm")
    print("  W: Label as White (存活精子)")
    print("  P: Label as Pink (破碎精子)")
    print("  D: Delete last label")
    print("  N: Next image")
    print("  P: Previous image")
    print("  S: Save labels")
    print("  Q: Quit")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('w'):
            current_label = 0
            print("Current label: White")
            draw_labels()
        elif key == ord('p'):
            current_label = 1
            print("Current label: Pink")
            draw_labels()
        elif key == ord('d'):
            img_name = images[current_image_index]
            if img_name in labels and labels[img_name]:
                labels[img_name].pop()
                print("Deleted last label")
                draw_labels()
        elif key == ord('n'):
            if current_image_index < len(images) - 1:
                current_image_index += 1
                load_image()
                print(f"Image {current_image_index+1}/{len(images)}")
        elif key == ord('b'):
            if current_image_index > 0:
                current_image_index -= 1
                load_image()
                print(f"Image {current_image_index+1}/{len(images)}")
        elif key == ord('s'):
            save_labels()
        elif key == ord('q'):
            save_labels()
            print("Exiting...")
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
