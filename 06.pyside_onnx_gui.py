import sys
import os
import glob
import csv
import json
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QProgressBar, QMessageBox, QHeaderView,
    QGroupBox, QRadioButton, QButtonGroup, QFrame, QScrollArea,
    QDoubleSpinBox, QSplitter, QComboBox, QCheckBox, QGraphicsDropShadowEffect,
    QStackedWidget, QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QToolBar
)
from PySide6.QtGui import QPixmap, QIcon, QFont, QColor, QPalette, QImage, QPainter, QAction
from PySide6.QtCore import Qt, QThread, Signal, QRectF, QPropertyAnimation, QEasingCurve, QSize


THEME = {
    "sidebar_bg": "#1e293b",
    "sidebar_text": "#e0e0e0",
    "sidebar_accent": "#5b9bd5",
    "content_bg": "#ffffff",
    "text_primary": "#1e293b",
    "text_secondary": "#64748b",
    "accent_green": "#10b981",
    "accent_blue": "#5b9bd5",
    "border_color": "#e2e8f0",
}


def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def get_config_path():
    config_dir = os.path.expanduser("~/.sperm_counter")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "config.json")


def load_config():
    config_path = get_config_path()
    default_config = {
        "output_dir": os.path.expanduser("~/Desktop/SpermResults"),
        "default_conf_thresh": 0.1,
        "default_iou_thresh": 0.45,
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                return {**default_config, **config}
        except Exception:
            pass
    return default_config


def save_config(config):
    config_path = get_config_path()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


class ONNXDetector:
    def __init__(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_size = self.session.get_inputs()[0].shape[2] if len(self.session.get_inputs()[0].shape) > 2 else 640
    
    def preprocess(self, img):
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, 0)
        return img_batch, h, w
    
    def postprocess(self, outputs, conf_thresh=0.1, iou_thresh=0.45):
        predictions = outputs[0]
        predictions = np.squeeze(predictions)
        
        if predictions.shape[0] == 6:
            predictions = predictions.T
        
        boxes = []
        scores = []
        class_ids = []
        
        for pred in predictions:
            box_coords = pred[:4]
            class_scores = pred[4:]
            score = class_scores.max()
            
            if score > conf_thresh:
                class_id = class_scores.argmax()
                boxes.append(box_coords)
                scores.append(float(score))
                class_ids.append(int(class_id))
        
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    'box': boxes[i].tolist(),
                    'score': scores[i],
                    'class_id': int(class_ids[i])
                })
        
        return results
    
    def detect(self, image_path, conf_thresh=0.1, iou_thresh=0.45):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        input_data, orig_h, orig_w = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_data})
        results = self.postprocess(outputs, conf_thresh, iou_thresh)
        
        return results
    
    def visualize(self, image_path, detections):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        h, w = img.shape[:2]
        img_copy = img.copy()
        
        for det in detections:
            box = np.array(det['box'])
            x_center, y_center, box_w, box_h = box
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)
            
            if det['class_id'] == 0:
                color = (0, 255, 0)
                label = "Alive"
            else:
                color = (0, 0, 255)
                label = "Dead"
            
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
            
            text = f"{label}: {det['score']:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            (label_w, label_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            label_y = y1 - 10 if y1 - 10 > label_h else y1 + label_h + 10
            
            cv2.rectangle(img_copy, (x1, label_y - label_h - 8), (x1 + label_w + 4, label_y + baseline - 8), color, -1)
            cv2.putText(img_copy, text, (x1 + 2, label_y - 4), font, font_scale, (255, 255, 255), thickness)
        
        return img_copy


class WorkerThread(QThread):
    progress_updated = Signal(int)
    result_ready = Signal(list)
    error_occurred = Signal(str)
    image_processed = Signal(str, object, object)
    
    def __init__(self, detector, image_paths, conf_thresh, iou_thresh):
        super().__init__()
        self.detector = detector
        self.image_paths = image_paths
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
    
    def run(self):
        results = []
        total = len(self.image_paths)
        
        for i, image_path in enumerate(self.image_paths):
            try:
                detections = self.detector.detect(image_path, self.conf_thresh, self.iou_thresh)
                
                white_count = 0
                pink_count = 0
                
                for det in detections:
                    if det['class_id'] == 0:
                        white_count += 1
                    else:
                        pink_count += 1
                
                total_count = white_count + pink_count
                survival_rate = (white_count / total_count) * 100 if total_count > 0 else 0
                
                result = {
                    'filename': os.path.basename(image_path),
                    'path': image_path,
                    'detections': detections,
                    'white': white_count,
                    'pink': pink_count,
                    'rate': survival_rate
                }
                results.append(result)
                
                if len(self.image_paths) == 1:
                    try:
                        original_img = cv2.imread(image_path)
                        annotated_img = self.detector.visualize(image_path, detections)
                        self.image_processed.emit(image_path, original_img, annotated_img)
                    except Exception:
                        pass
                
                self.progress_updated.emit(int((i + 1) / total * 100))
            except Exception as e:
                self.error_occurred.emit(f"Error processing {os.path.basename(image_path)}: {str(e)}")
                return
        
        self.result_ready.emit(results)


class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("background-color: #f8fafc; border-radius: 12px;")
        self.pixmap_item = None
        self.current_zoom = 1.0
    
    def set_image(self, image):
        self.scene().clear()
        
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            bytes_per_line = 3 * width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
        else:
            pixmap = QPixmap(image)
        
        if not pixmap.isNull():
            self.pixmap_item = QGraphicsPixmapItem(pixmap)
            self.scene().addItem(self.pixmap_item)
            self.setSceneRect(QRectF(pixmap.rect()))
            self.fit_to_view()
    
    def fit_to_view(self):
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self.current_zoom = 1.0
    
    def wheelEvent(self, event):
        if self.pixmap_item:
            zoom_in_factor = 1.1
            zoom_out_factor = 1 / zoom_in_factor
            
            if event.angleDelta().y() > 0:
                self.scale(zoom_in_factor, zoom_in_factor)
            else:
                self.scale(zoom_out_factor, zoom_out_factor)


class SidebarButton(QPushButton):
    def __init__(self, icon, text, parent=None):
        super().__init__(text, parent)
        self.setIcon(icon)
        self.setIconSize(QSize(24, 24))
        self.setCheckable(True)
        self.setMinimumHeight(50)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(f"""
            SidebarButton {{
                background-color: transparent;
                color: {THEME['sidebar_text']};
                border: none;
                border-radius: 10px;
                padding: 12px 20px;
                text-align: left;
                font-size: 15px;
                font-weight: 600;
            }}
            SidebarButton:hover {{
                background-color: rgba(255, 255, 255, 0.1);
            }}
            SidebarButton:checked {{
                background-color: {THEME['sidebar_accent']};
                color: white;
            }}
        """)


class ModernFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(f"""
            ModernFrame {{
                background-color: white;
                border-radius: 16px;
            }}
        """)
        
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(QColor(0, 0, 0, 40))
        self.setGraphicsEffect(shadow)


class SpermCounterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.setWindowTitle("Sperm Staining Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 700)
        
        icon_path = get_resource_path("assets/logos/logo.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        self.setup_palette()
        
        self.detector = None
        self.current_page = 0
        self.conf_thresh = self.config["default_conf_thresh"]
        self.iou_thresh = self.config["default_iou_thresh"]
        self.results = []
        self.current_original_img = None
        self.current_annotated_img = None
        self.sidebar_expanded = True
        
        self.init_ui()
        self.load_model()
    
    def setup_palette(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(THEME['content_bg']))
        palette.setColor(QPalette.WindowText, QColor(THEME['text_primary']))
        self.setPalette(palette)
    
    def load_model(self):
        try:
            model_path = get_resource_path('best.onnx')
            
            if not os.path.exists(model_path):
                model_path = get_resource_path('runs/detect/sperm_detection/weights/best.onnx')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.detector = ONNXDetector(model_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}\n\nPlease ensure the model file exists.")
            self.detector = None
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self.sidebar = self.create_sidebar()
        main_layout.addWidget(self.sidebar)
        
        content_widget = self.create_content_area()
        main_layout.addWidget(content_widget, 1)
    
    def create_sidebar(self):
        self.sidebar_widget = QWidget()
        self.sidebar_widget.setFixedWidth(260)
        self.sidebar_widget.setStyleSheet(f"""
            QWidget {{
                background-color: {THEME['sidebar_bg']};
            }}
        """)
        
        layout = QVBoxLayout(self.sidebar_widget)
        layout.setContentsMargins(20, 30, 20, 30)
        layout.setSpacing(15)
        
        header_layout = QHBoxLayout()
        
        self.toggle_btn = QPushButton("◀")
        self.toggle_btn.setFixedSize(32, 32)
        self.toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {THEME['sidebar_text']};
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 255, 255, 0.1);
            }}
        """)
        self.toggle_btn.clicked.connect(self.toggle_sidebar)
        header_layout.addWidget(self.toggle_btn)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        sidebar_logo_path = get_resource_path("assets/logos/logo.png")
        if os.path.exists(sidebar_logo_path):
            sidebar_logo_label = QLabel()
            sidebar_logo_pixmap = QPixmap(sidebar_logo_path)
            sidebar_logo_pixmap = sidebar_logo_pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            sidebar_logo_label.setPixmap(sidebar_logo_pixmap)
            sidebar_logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(sidebar_logo_label)
        
        title_label = QLabel("Sperm Analysis")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: white;
                font-size: 34px;
                font-weight: 800;
                padding: 10px 0;
            }}
        """)
        layout.addWidget(title_label)
        
        subtitle_label = QLabel("Staining Status Tool")
        subtitle_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_secondary']};
                font-size: 14px;
                padding-bottom: 20px;
            }}
        """)
        layout.addWidget(subtitle_label)
        
        self.nav_group = QButtonGroup(self)
        
        home_icon = QIcon.fromTheme("go-home")
        analyze_icon = QIcon.fromTheme("view-refresh")
        settings_icon = QIcon.fromTheme("preferences-system")
        
        self.home_btn = SidebarButton(home_icon, "Home")
        self.home_btn.setChecked(True)
        self.nav_group.addButton(self.home_btn, 0)
        
        self.analyze_btn = SidebarButton(analyze_icon, "Analyze")
        self.nav_group.addButton(self.analyze_btn, 1)
        
        self.settings_btn = SidebarButton(settings_icon, "Settings")
        self.nav_group.addButton(self.settings_btn, 2)
        
        layout.addWidget(self.home_btn)
        layout.addWidget(self.analyze_btn)
        layout.addWidget(self.settings_btn)
        
        layout.addStretch()
        
        version_label = QLabel("v1.0.5")
        version_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_secondary']};
                font-size: 13px;
            }}
        """)
        layout.addWidget(version_label)
        
        self.nav_group.buttonClicked.connect(self.switch_page)
        
        return self.sidebar_widget
    
    def toggle_sidebar(self):
        if self.sidebar_expanded:
            target_width = 60
            self.toggle_btn.setText("▶")
        else:
            target_width = 260
            self.toggle_btn.setText("◀")
        
        self.animation = QPropertyAnimation(self.sidebar, b"minimumWidth")
        self.animation.setDuration(300)
        self.animation.setStartValue(self.sidebar.width())
        self.animation.setEndValue(target_width)
        self.animation.setEasingCurve(QEasingCurve.InOutCubic)
        self.animation.start()
        
        self.sidebar.setFixedWidth(target_width)
        self.sidebar_expanded = not self.sidebar_expanded
    
    def create_content_area(self):
        content = QWidget()
        content.setStyleSheet(f"""
            QWidget {{
                background-color: {THEME['content_bg']};
            }}
        """)
        
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        self.pages = []
        
        home_page = self.create_home_page()
        self.pages.append(home_page)
        
        analyze_page = self.create_analyze_page()
        self.pages.append(analyze_page)
        
        settings_page = self.create_settings_page()
        self.pages.append(settings_page)
        
        layout.addWidget(home_page)
        
        return content
    
    def create_home_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(25)
        layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(25)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        welcome_frame = ModernFrame()
        welcome_layout = QVBoxLayout(welcome_frame)
        welcome_layout.setContentsMargins(45, 45, 45, 45)
        welcome_layout.setAlignment(Qt.AlignCenter)
        
        logo_path = get_resource_path("assets/logos/logo.png")
        if os.path.exists(logo_path):
            logo_label = QLabel()
            logo_pixmap = QPixmap(logo_path)
            logo_pixmap = logo_pixmap.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
            logo_label.setAlignment(Qt.AlignCenter)
            welcome_layout.addWidget(logo_label)
            welcome_layout.addSpacing(20)
        
        welcome_title = QLabel("Welcome to Sperm Analysis")
        welcome_title.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 36px;
                font-weight: 800;
                padding-bottom: 14px;
                line-height: 1.2;
            }}
        """)
        welcome_layout.addWidget(welcome_title)
        
        welcome_desc = QLabel("Professional tool for analyzing sperm staining status. Fast, accurate, and easy to use.")
        welcome_desc.setWordWrap(True)
        welcome_desc.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_secondary']};
                font-size: 18px;
                line-height: 1.6;
            }}
        """)
        welcome_layout.addWidget(welcome_desc)
        
        scroll_layout.addWidget(welcome_frame)
        
        features_frame = ModernFrame()
        features_layout = QVBoxLayout(features_frame)
        features_layout.setContentsMargins(45, 45, 45, 45)
        
        features_title = QLabel("Key Features")
        features_title.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 26px;
                font-weight: 700;
                padding-bottom: 28px;
                line-height: 1.2;
            }}
        """)
        features_layout.addWidget(features_title)
        
        features = [
            ("🔍", "YOLO-based Detection", "State-of-the-art object detection powered by YOLOv8 neural network for high accuracy"),
            ("📊", "Batch Processing", "Analyze hundreds of images simultaneously with real-time progress tracking"),
            ("🎯", "Visual Results", "View original and annotated images with detection bounding boxes"),
            ("⚙️", "Custom Parameters", "Fine-tune confidence threshold and IoU threshold to optimize detection results"),
            ("💾", "Export Results", "Save comprehensive analysis results to CSV files with clear headers and statistics"),
            ("🎨", "Modern UI", "Clean, intuitive, and fully English-localized interface with beautiful design"),
        ]
        
        for icon, title, desc in features:
            feature_layout = QHBoxLayout()
            
            icon_label = QLabel(icon)
            icon_label.setStyleSheet("""
                QLabel {
                    font-size: 38px;
                    padding-right: 24px;
                    padding-top: 2px;
                }
            """)
            feature_layout.addWidget(icon_label)
            
            text_layout = QVBoxLayout()
            
            feat_title = QLabel(title)
            feat_title.setStyleSheet(f"""
                QLabel {{
                    color: {THEME['text_primary']};
                    font-size: 18px;
                    font-weight: 700;
                    padding-bottom: 5px;
                    line-height: 1.3;
                }}
            """)
            text_layout.addWidget(feat_title)
            
            feat_desc = QLabel(desc)
            feat_desc.setWordWrap(True)
            feat_desc.setStyleSheet(f"""
                QLabel {{
                    color: {THEME['text_secondary']};
                    font-size: 16px;
                    line-height: 1.6;
                }}
            """)
            text_layout.addWidget(feat_desc)
            
            feature_layout.addLayout(text_layout, 1)
            features_layout.addLayout(feature_layout)
            features_layout.addSpacing(20)
        
        scroll_layout.addWidget(features_frame)
        
        quick_frame = ModernFrame()
        quick_layout = QVBoxLayout(quick_frame)
        quick_layout.setContentsMargins(45, 45, 45, 45)
        
        quick_title = QLabel("Quick Start Guide")
        quick_title.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 26px;
                font-weight: 700;
                padding-bottom: 28px;
                line-height: 1.2;
            }}
        """)
        quick_layout.addWidget(quick_title)
        
        steps = [
            "1. Go to the Settings page to configure default parameters (optional)",
            "2. Navigate to the Analyze page",
            "3. Select either a single image or an entire folder of images",
            "4. Adjust the Confidence and IoU thresholds as needed for your use case",
            "5. Click Start Analysis and wait for processing to complete",
            "6. View results in the table and click on rows to see image",
            "7. Export results to CSV if needed",
        ]
        
        for step in steps:
            step_label = QLabel(step)
            step_label.setWordWrap(True)
            step_label.setStyleSheet(f"""
                QLabel {{
                    color: {THEME['text_secondary']};
                    font-size: 17px;
                    padding: 10px 0;
                    line-height: 1.5;
                }}
            """)
            quick_layout.addWidget(step_label)
        
        scroll_layout.addWidget(quick_frame)
        
        scroll_layout.addStretch()
        
        return page
    
    def create_analyze_page(self):
        page = QWidget()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(20)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background: transparent;
                height: 0px;
            }
        """)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(20)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        params_frame = ModernFrame()
        params_layout = QVBoxLayout(params_frame)
        params_layout.setContentsMargins(30, 30, 30, 30)
        
        params_title = QLabel("🎯 Detection Parameters")
        params_title.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 22px;
                font-weight: 700;
                padding-bottom: 20px;
            }}
        """)
        params_layout.addWidget(params_title)
        
        params_grid = QHBoxLayout()
        
        conf_layout = QVBoxLayout()
        conf_label = QLabel("Confidence Threshold")
        conf_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_secondary']};
                font-size: 15px;
                font-weight: 600;
                padding-bottom: 10px;
            }}
        """)
        conf_layout.addWidget(conf_label)
        
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 0.99)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(self.conf_thresh)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setMinimumHeight(50)
        self.conf_spin.setStyleSheet(f"""
            QDoubleSpinBox {{
                padding: 12px 16px;
                border: 2px solid {THEME['border_color']};
                border-radius: 10px;
                background-color: white;
                font-size: 16px;
                font-weight: 500;
            }}
            QDoubleSpinBox:focus {{
                border-color: {THEME['sidebar_accent']};
            }}
        """)
        self.conf_spin.valueChanged.connect(lambda v: setattr(self, 'conf_thresh', v))
        conf_layout.addWidget(self.conf_spin)
        params_grid.addLayout(conf_layout, 1)
        
        iou_layout = QVBoxLayout()
        iou_label = QLabel("IoU Threshold")
        iou_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_secondary']};
                font-size: 15px;
                font-weight: 600;
                padding-bottom: 10px;
            }}
        """)
        iou_layout.addWidget(iou_label)
        
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 0.99)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(self.iou_thresh)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setMinimumHeight(50)
        self.iou_spin.setStyleSheet(f"""
            QDoubleSpinBox {{
                padding: 12px 16px;
                border: 2px solid {THEME['border_color']};
                border-radius: 10px;
                background-color: white;
                font-size: 16px;
                font-weight: 500;
            }}
            QDoubleSpinBox:focus {{
                border-color: {THEME['sidebar_accent']};
            }}
        """)
        self.iou_spin.valueChanged.connect(lambda v: setattr(self, 'iou_thresh', v))
        iou_layout.addWidget(self.iou_spin)
        params_grid.addLayout(iou_layout, 1)
        
        params_layout.addLayout(params_grid)
        scroll_layout.addWidget(params_frame)
        
        input_frame = ModernFrame()
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(30, 30, 30, 30)
        
        input_title = QLabel("📁 Input Selection")
        input_title.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 22px;
                font-weight: 700;
                padding-bottom: 20px;
            }}
        """)
        input_layout.addWidget(input_title)
        
        radio_layout = QHBoxLayout()
        
        self.single_radio = QRadioButton("Single Image")
        self.single_radio.setStyleSheet(f"""
            QRadioButton {{
                color: {THEME['text_secondary']};
                font-size: 16px;
                padding: 8px 0;
            }}
        """)
        
        self.folder_radio = QRadioButton("Folder")
        self.folder_radio.setChecked(True)
        self.folder_radio.setStyleSheet(f"""
            QRadioButton {{
                color: {THEME['text_secondary']};
                font-size: 16px;
                padding: 8px 0;
            }}
        """)
        
        radio_layout.addWidget(self.single_radio)
        radio_layout.addWidget(self.folder_radio)
        radio_layout.addStretch()
        input_layout.addLayout(radio_layout)
        
        path_layout = QHBoxLayout()
        
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select images or folder...")
        self.path_edit.setMinimumHeight(50)
        self.path_edit.setStyleSheet(f"""
            QLineEdit {{
                padding: 12px 16px;
                border: 2px solid {THEME['border_color']};
                border-radius: 10px;
                background-color: white;
                font-size: 16px;
                font-weight: 500;
            }}
            QLineEdit:focus {{
                border-color: {THEME['sidebar_accent']};
            }}
        """)
        path_layout.addWidget(self.path_edit, 1)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setMinimumSize(130, 50)
        self.browse_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['sidebar_accent']};
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 28px;
                font-size: 16px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #4a8cc5;
            }}
        """)
        self.browse_btn.clicked.connect(self.browse_files)
        path_layout.addWidget(self.browse_btn)
        
        input_layout.addLayout(path_layout)
        scroll_layout.addWidget(input_frame)
        
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Start Analysis")
        self.start_btn.setMinimumSize(180, 55)
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['accent_green']};
                color: white;
                border: none;
                border-radius: 14px;
                padding: 12px 25px;
                font-size: 16px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: #059669;
            }}
            QPushButton:disabled {{
                background-color: {THEME['border_color']};
            }}
        """)
        self.start_btn.clicked.connect(self.start_analysis)
        btn_layout.addWidget(self.start_btn)
        
        self.save_btn = QPushButton("💾 Save Results")
        self.save_btn.setEnabled(False)
        self.save_btn.setMinimumSize(180, 55)
        self.save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['sidebar_accent']};
                color: white;
                border: none;
                border-radius: 14px;
                padding: 12px 25px;
                font-size: 16px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: #4a8cc5;
            }}
            QPushButton:disabled {{
                background-color: {THEME['border_color']};
            }}
        """)
        self.save_btn.clicked.connect(self.save_results)
        btn_layout.addWidget(self.save_btn)
        
        btn_layout.addStretch()
        scroll_layout.addLayout(btn_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 10px;
                background-color: {THEME['border_color']};
                height: 24px;
                text-align: center;
                font-size: 14px;
                font-weight: 600;
            }}
            QProgressBar::chunk {{
                background-color: {THEME['sidebar_accent']};
                border-radius: 10px;
            }}
        """)
        scroll_layout.addWidget(self.progress_bar)
        
        self.table_frame = ModernFrame()
        table_frame_layout = QVBoxLayout(self.table_frame)
        table_frame_layout.setContentsMargins(30, 30, 30, 30)
        
        table_title = QLabel("📊 Analysis Results")
        table_title.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 20px;
                font-weight: 700;
                padding-bottom: 15px;
            }}
        """)
        table_frame_layout.addWidget(table_title)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Image Name", "Alive Count", "Dead Count", "Survival Rate"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        row_height = 40
        self.table.verticalHeader().setDefaultSectionSize(row_height)
        self.table.setMinimumHeight(row_height * 10 + 45)
        self.table.setMaximumHeight(row_height * 10 + 45)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                border: none;
                background-color: white;
                border-radius: 10px;
                gridline-color: {THEME['border_color']};
            }}
            QTableWidget::item {{
                padding: 10px;
                border: none;
            }}
            QTableWidget::item:selected {{
                background-color: {THEME['sidebar_accent']};
                color: white;
            }}
            QHeaderView::section {{
                background-color: #f8fafc;
                color: {THEME['text_secondary']};
                padding: 12px;
                border: none;
                border-bottom: 2px solid {THEME['border_color']};
                font-weight: 700;
                font-size: 14px;
            }}
        """)
        self.table.itemSelectionChanged.connect(self.on_table_selection_changed)
        table_frame_layout.addWidget(self.table)
        scroll_layout.addWidget(self.table_frame)
        
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 16px;
                font-weight: 700;
                padding: 18px;
                background-color: white;
                border-radius: 12px;
            }}
        """)
        self.stats_label.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(self.stats_label)
        
        self.viewer_frame = ModernFrame()
        viewer_frame_layout = QVBoxLayout(self.viewer_frame)
        viewer_frame_layout.setContentsMargins(30, 30, 30, 30)
        
        viewer_title = QLabel("🖼️ Image Viewer - Click on table row to view")
        viewer_title.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 20px;
                font-weight: 700;
                padding-bottom: 15px;
            }}
        """)
        viewer_frame_layout.addWidget(viewer_title)
        
        viewer_options_layout = QHBoxLayout()
        
        self.viewer_tab = QComboBox()
        self.viewer_tab.addItems(["Both Images", "Original Only", "Annotated Only"])
        self.viewer_tab.setMinimumHeight(40)
        self.viewer_tab.setStyleSheet(f"""
            QComboBox {{
                padding: 8px 16px;
                border: 2px solid {THEME['border_color']};
                border-radius: 8px;
                background-color: white;
                font-size: 14px;
                font-weight: 500;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox::down-arrow {{
                width: 20px;
                height: 20px;
            }}
        """)
        self.viewer_tab.currentIndexChanged.connect(self.switch_view)
        viewer_options_layout.addWidget(self.viewer_tab)
        
        self.reset_btn = QPushButton("Reset View")
        self.reset_btn.setMinimumSize(120, 40)
        self.reset_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #64748b;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 8px 20px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #475569;
            }}
        """)
        self.reset_btn.clicked.connect(self.reset_viewers)
        viewer_options_layout.addWidget(self.reset_btn)
        viewer_options_layout.addStretch()
        
        viewer_frame_layout.addLayout(viewer_options_layout)
        
        viewer_container = QWidget()
        viewer_container_layout = QHBoxLayout(viewer_container)
        viewer_container_layout.setContentsMargins(0, 10, 0, 0)
        viewer_container_layout.setSpacing(15)
        
        original_label = QLabel("📷 Original Image")
        original_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 16px;
                font-weight: 600;
                padding-bottom: 10px;
            }}
        """)
        
        annotated_label = QLabel("🎨 Annotated Image")
        annotated_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 16px;
                font-weight: 600;
                padding-bottom: 10px;
            }}
        """)
        
        original_viewer_widget = QWidget()
        original_viewer_layout = QVBoxLayout(original_viewer_widget)
        original_viewer_layout.setContentsMargins(0, 0, 0, 0)
        original_viewer_layout.setSpacing(5)
        original_viewer_layout.addWidget(original_label)
        
        self.original_viewer = ImageViewer()
        self.original_viewer.setMinimumHeight(300)
        original_viewer_layout.addWidget(self.original_viewer, 1)
        
        annotated_viewer_widget = QWidget()
        annotated_viewer_layout = QVBoxLayout(annotated_viewer_widget)
        annotated_viewer_layout.setContentsMargins(0, 0, 0, 0)
        annotated_viewer_layout.setSpacing(5)
        annotated_viewer_layout.addWidget(annotated_label)
        
        self.annotated_viewer = ImageViewer()
        self.annotated_viewer.setMinimumHeight(300)
        annotated_viewer_layout.addWidget(self.annotated_viewer, 1)
        
        viewer_container_layout.addWidget(original_viewer_widget, 1)
        viewer_container_layout.addWidget(annotated_viewer_widget, 1)
        
        viewer_frame_layout.addWidget(viewer_container, 1)
        
        self.viewer_frame.setVisible(False)
        scroll_layout.addWidget(self.viewer_frame)
        
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
        
        return page
    
    def create_settings_page(self):
        page = QWidget()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: transparent;
                width: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background: transparent;
                height: 0px;
            }
        """)
        
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(25)
        
        settings_frame = ModernFrame()
        settings_layout = QVBoxLayout(settings_frame)
        settings_layout.setContentsMargins(40, 40, 40, 40)
        
        settings_title = QLabel("⚙️ Application Settings")
        settings_title.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 24px;
                font-weight: 700;
                padding-bottom: 30px;
            }}
        """)
        settings_layout.addWidget(settings_title)
        
        output_layout = QVBoxLayout()
        output_label = QLabel("Default Output Directory")
        output_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_secondary']};
                font-size: 15px;
                font-weight: 600;
                padding-bottom: 12px;
            }}
        """)
        output_layout.addWidget(output_label)
        
        output_path_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit(self.config["output_dir"])
        self.output_dir_edit.setMinimumHeight(50)
        self.output_dir_edit.setStyleSheet(f"""
            QLineEdit {{
                padding: 12px 16px;
                border: 2px solid {THEME['border_color']};
                border-radius: 10px;
                background-color: white;
                font-size: 16px;
                font-weight: 500;
            }}
            QLineEdit:focus {{
                border-color: {THEME['sidebar_accent']};
            }}
        """)
        output_path_layout.addWidget(self.output_dir_edit, 1)
        
        self.output_dir_btn = QPushButton("Browse")
        self.output_dir_btn.setMinimumSize(110, 50)
        self.output_dir_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['sidebar_accent']};
                color: white;
                border: none;
                border-radius: 10px;
                padding: 12px 28px;
                font-size: 15px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #4a8cc5;
            }}
        """)
        self.output_dir_btn.clicked.connect(self.browse_output_dir)
        output_path_layout.addWidget(self.output_dir_btn)
        output_layout.addLayout(output_path_layout)
        settings_layout.addLayout(output_layout)
        
        settings_layout.addSpacing(35)
        
        thresholds_layout = QVBoxLayout()
        thresholds_title = QLabel("Default Detection Thresholds")
        thresholds_title.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_secondary']};
                font-size: 15px;
                font-weight: 600;
                padding-bottom: 18px;
            }}
        """)
        thresholds_layout.addWidget(thresholds_title)
        
        thresholds_grid = QHBoxLayout()
        
        default_conf_layout = QVBoxLayout()
        default_conf_label = QLabel("Default Confidence Threshold")
        default_conf_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_secondary']};
                font-size: 14px;
                padding-bottom: 10px;
            }}
        """)
        default_conf_layout.addWidget(default_conf_label)
        
        self.default_conf_spin = QDoubleSpinBox()
        self.default_conf_spin.setRange(0.01, 0.99)
        self.default_conf_spin.setSingleStep(0.05)
        self.default_conf_spin.setValue(self.config["default_conf_thresh"])
        self.default_conf_spin.setDecimals(2)
        self.default_conf_spin.setMinimumHeight(50)
        self.default_conf_spin.setStyleSheet(f"""
            QDoubleSpinBox {{
                padding: 12px 16px;
                border: 2px solid {THEME['border_color']};
                border-radius: 10px;
                background-color: white;
                font-size: 15px;
            }}
            QDoubleSpinBox:focus {{
                border-color: {THEME['sidebar_accent']};
            }}
        """)
        default_conf_layout.addWidget(self.default_conf_spin)
        thresholds_grid.addLayout(default_conf_layout, 1)
        
        default_iou_layout = QVBoxLayout()
        default_iou_label = QLabel("Default IoU Threshold")
        default_iou_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_secondary']};
                font-size: 14px;
                padding-bottom: 10px;
            }}
        """)
        default_iou_layout.addWidget(default_iou_label)
        
        self.default_iou_spin = QDoubleSpinBox()
        self.default_iou_spin.setRange(0.01, 0.99)
        self.default_iou_spin.setSingleStep(0.05)
        self.default_iou_spin.setValue(self.config["default_iou_thresh"])
        self.default_iou_spin.setDecimals(2)
        self.default_iou_spin.setMinimumHeight(50)
        self.default_iou_spin.setStyleSheet(f"""
            QDoubleSpinBox {{
                padding: 12px 16px;
                border: 2px solid {THEME['border_color']};
                border-radius: 10px;
                background-color: white;
                font-size: 15px;
            }}
            QDoubleSpinBox:focus {{
                border-color: {THEME['sidebar_accent']};
            }}
        """)
        default_iou_layout.addWidget(self.default_iou_spin)
        thresholds_grid.addLayout(default_iou_layout, 1)
        
        thresholds_layout.addLayout(thresholds_grid)
        settings_layout.addLayout(thresholds_layout)
        
        settings_layout.addSpacing(35)
        
        save_settings_layout = QHBoxLayout()
        self.save_settings_btn = QPushButton("Save Settings")
        self.save_settings_btn.setMinimumSize(190, 55)
        self.save_settings_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['sidebar_accent']};
                color: white;
                border: none;
                border-radius: 14px;
                padding: 12px 25px;
                font-size: 16px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: #4a8cc5;
            }}
        """)
        self.save_settings_btn.clicked.connect(self.save_settings)
        save_settings_layout.addWidget(self.save_settings_btn)
        save_settings_layout.addStretch()
        settings_layout.addLayout(save_settings_layout)
        
        layout.addWidget(settings_frame)
        layout.addStretch()
        
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
        
        return page
    
    def switch_page(self, button):
        page_index = self.nav_group.id(button)
        if page_index == self.current_page:
            return
        
        content_layout = self.centralWidget().layout().itemAt(1).widget().layout()
        old_page = content_layout.itemAt(0).widget()
        old_page.setParent(None)
        
        self.current_page = page_index
        content_layout.addWidget(self.pages[page_index])
    
    def browse_files(self):
        if self.folder_radio.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Select Folder")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        
        if path:
            self.path_edit.setText(path)
    
    def browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_edit.setText(path)
    
    def save_settings(self):
        self.config["output_dir"] = self.output_dir_edit.text()
        self.config["default_conf_thresh"] = self.default_conf_spin.value()
        self.config["default_iou_thresh"] = self.default_iou_spin.value()
        
        self.conf_thresh = self.config["default_conf_thresh"]
        self.iou_thresh = self.config["default_iou_thresh"]
        
        self.conf_spin.setValue(self.conf_thresh)
        self.iou_spin.setValue(self.iou_thresh)
        
        save_config(self.config)
        QMessageBox.information(self, "Success", "Settings saved successfully!")
    
    def start_analysis(self):
        if self.detector is None:
            QMessageBox.warning(self, "Warning", "Model not loaded, cannot start analysis")
            return
        
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Warning", "Please select images or folder first")
            return
        
        if not os.path.exists(path):
            QMessageBox.warning(self, "Warning", "Path does not exist")
            return
        
        if os.path.isfile(path):
            image_paths = [path]
        else:
            image_paths = []
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for ext in extensions:
                image_paths.extend(glob.glob(os.path.join(path, ext)))
                image_paths.extend(glob.glob(os.path.join(path, ext.upper())))
            
            if not image_paths:
                QMessageBox.warning(self, "Warning", "No images found in folder")
                return
        
        self.start_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.table.setRowCount(0)
        self.stats_label.setText("")
        self.viewer_frame.setVisible(False)
        
        self.worker = WorkerThread(self.detector, image_paths, self.conf_thresh, self.iou_thresh)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.result_ready.connect(self.show_results)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.image_processed.connect(self.on_image_processed)
        self.worker.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def show_results(self, results):
        self.results = results
        
        self.table.setRowCount(len(results))
        for i, result in enumerate(results):
            self.table.setItem(i, 0, QTableWidgetItem(result['filename']))
            self.table.setItem(i, 1, QTableWidgetItem(str(result['white'])))
            self.table.setItem(i, 2, QTableWidgetItem(str(result['pink'])))
            self.table.setItem(i, 3, QTableWidgetItem(f"{result['rate']:.2f}%"))
        
        total_white = sum(r['white'] for r in results)
        total_pink = sum(r['pink'] for r in results)
        total_count = total_white + total_pink
        overall_rate = (total_white / total_count) * 100 if total_count > 0 else 0
        
        self.stats_label.setText(
            f"Total Images: {len(results)} | Total Alive: {total_white} | Total Dead: {total_pink} | Overall Survival Rate: {overall_rate:.2f}%"
        )
        
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        if len(results) > 0:
            if len(results) == 1:
                self.viewer_frame.setVisible(True)
            else:
                self.table.selectRow(0)
    
    def on_image_processed(self, image_path, original_img, annotated_img):
        self.current_original_img = original_img
        self.current_annotated_img = annotated_img
        
        if original_img is not None:
            self.original_viewer.set_image(original_img)
        if annotated_img is not None:
            self.annotated_viewer.set_image(annotated_img)
        
        self.viewer_frame.setVisible(True)
    
    def on_table_selection_changed(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if selected_rows and self.detector:
            row = selected_rows[0].row()
            if row < len(self.results):
                result = self.results[row]
                image_path = result.get('path')
                if image_path and os.path.exists(image_path):
                    try:
                        original_img = cv2.imread(image_path)
                        annotated_img = self.detector.visualize(image_path, result.get('detections', []))
                        
                        self.current_original_img = original_img
                        self.current_annotated_img = annotated_img
                        
                        self.original_viewer.set_image(original_img)
                        self.annotated_viewer.set_image(annotated_img)
                        self.viewer_frame.setVisible(True)
                    except Exception as e:
                        print(f"Error loading image: {e}")
    
    def switch_view(self, index):
        if index == 0:
            self.original_viewer.setVisible(True)
            self.annotated_viewer.setVisible(True)
        elif index == 1:
            self.original_viewer.setVisible(True)
            self.annotated_viewer.setVisible(False)
        else:
            self.original_viewer.setVisible(False)
            self.annotated_viewer.setVisible(True)
    
    def reset_viewers(self):
        self.original_viewer.fit_to_view()
        self.annotated_viewer.fit_to_view()
    
    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
    
    def save_results(self):
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to save")
            return
        
        default_dir = self.config["output_dir"]
        os.makedirs(default_dir, exist_ok=True)
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", 
            os.path.join(default_dir, "sperm_results.csv"), 
            "CSV Files (*.csv)"
        )
        if not path:
            return
        
        if not path.endswith('.csv'):
            path += '.csv'
        
        try:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Image Name", "Alive Count", "Dead Count", "Survival Rate"])
                
                for result in self.results:
                    writer.writerow([
                        result['filename'],
                        result['white'],
                        result['pink'],
                        f"{result['rate']:.2f}%"
                    ])
                
                total_white = sum(r['white'] for r in self.results)
                total_pink = sum(r['pink'] for r in self.results)
                total_count = total_white + total_pink
                overall_rate = (total_white / total_count) * 100 if total_count > 0 else 0
                
                writer.writerow([])
                writer.writerow(["Overall Stats", total_white, total_pink, f"{overall_rate:.2f}%"])
            
            QMessageBox.information(self, "Success", f"Results saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpermCounterApp()
    window.show()
    sys.exit(app.exec())
