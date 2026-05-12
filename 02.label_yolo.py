import json
import os
import cv2

def convert_to_yolo_format():
    # Create YOLO format directory structure
    os.makedirs('yolo_dataset/images/train', exist_ok=True)
    os.makedirs('yolo_dataset/images/val', exist_ok=True)
    os.makedirs('yolo_dataset/labels/train', exist_ok=True)
    os.makedirs('yolo_dataset/labels/val', exist_ok=True)
    
    # Read annotations
    if not os.path.exists('labels.json'):
        print("No labels.json found!")
        return
    
    with open('labels.json', 'r') as f:
        labels_data = json.load(f)
    
    print(f"Found {len(labels_data)} annotated images")
    
    # Count annotations
    total_white = 0
    total_pink = 0
    for img_name, labels in labels_data.items():
        for label in labels:
            if label[4] == 0:
                total_white += 1
            else:
                total_pink += 1
    
    print(f"Total annotations: {total_white + total_pink}")
    print(f"  - White sperm: {total_white}")
    print(f"  - Pink sperm: {total_pink}")
    
    # Split train/val set (80/20)
    img_names = list(labels_data.keys())
    split_idx = max(1, int(len(img_names) * 0.8))
    train_names = img_names[:split_idx]
    val_names = img_names[split_idx:]
    
    print(f"\nSplitting into {len(train_names)} training and {len(val_names)} validation images")
    
    # Class mapping
    class_map = {'white': 0, 'pink': 1}
    
    for img_name in img_names:
        # Read image to get dimensions
        img_path = os.path.join('photos', img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read {img_name}")
            continue
        
        height, width = img.shape[:2]
        img_labels = labels_data[img_name]
        
        # Generate YOLO format annotation file
        label_lines = []
        for label in img_labels:
            x1, y1, x2, y2, cls = label
            # Convert to YOLO format (center coordinates, width, height, normalized)
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
        # Copy images and labels to corresponding directories
        if img_name in train_names:
            img_dst = os.path.join('yolo_dataset/images/train', img_name)
            label_dst = os.path.join('yolo_dataset/labels/train', os.path.splitext(img_name)[0] + '.txt')
        else:
            img_dst = os.path.join('yolo_dataset/images/val', img_name)
            label_dst = os.path.join('yolo_dataset/labels/val', os.path.splitext(img_name)[0] + '.txt')
        
        cv2.imwrite(img_dst, img)
        with open(label_dst, 'w') as f:
            f.write('\n'.join(label_lines))
    
    print(f"\nConverted successfully!")
    print(f"  - Training images: {len(train_names)}")
    print(f"  - Validation images: {len(val_names)}")

if __name__ == '__main__':
    convert_to_yolo_format()
