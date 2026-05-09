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
    QDoubleSpinBox, QSplitter, QComboBox, QCheckBox, QGraphicsDropShadowEffect
)
from PySide6.QtGui import QPixmap, QIcon, QFont, QColor, QPalette
from PySide6.QtCore import Qt, QThread, Signal, QSize, QPropertyAnimation, QEasingCurve


# ==============================================
# Configuration & Styling
# ==============================================
# Modify these values to customize the appearance
THEME = {
    "sidebar_bg": "#1e293b",
    "sidebar_text": "#e0e0e0",
    "sidebar_accent": "#5b9bd5",
    "content_bg": "#ffffff",  # Pure white
    "text_primary": "#1e293b",
    "text_secondary": "#64748b",
    "accent_green": "#10b981",
    "accent_blue": "#5b9bd5",
    "border_color": "#e2e8f0",
}


def get_resource_path(relative_path):
    """Get resource file path (compatible with packaged apps)"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


def get_config_path():
    """Get the path to the configuration file"""
    config_dir = os.path.expanduser("~/.sperm_counter")
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "config.json")


def load_config():
    """Load configuration from file"""
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
        except:
            pass
    return default_config


def save_config(config):
    """Save configuration to file"""
    config_path = get_config_path()
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


# ==============================================
# ONNX Detector
# ==============================================
class ONNXDetector:
    """ONNX Runtime-based object detector"""
    
    def __init__(self, model_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_size = self.input_shape[2] if len(self.input_shape) > 2 else 640
        
    def preprocess(self, img):
        """Preprocess image"""
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)
        return img_batch, h, w
    
    def postprocess(self, outputs, conf_thresh=0.1, iou_thresh=0.45):
        """Postprocess detection results"""
        predictions = outputs[0]
        predictions = np.squeeze(predictions)
        
        # YOLOv8 ONNX output format: (6, 8400) -> transpose to (8400, 6)
        if predictions.shape[0] == 6:
            predictions = predictions.T
        
        boxes = []
        scores = []
        class_ids = []
        
        for pred in predictions:
            # pred format: [x, y, w, h, class0_score, class1_score]
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
        
        # NMSBoxes expects boxes in [x, y, w, h] format
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    'box': boxes[i],
                    'score': scores[i],
                    'class_id': int(class_ids[i])
                })
        
        return results
    
    def detect(self, image_path, conf_thresh=0.1, iou_thresh=0.45):
        """Execute detection"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        input_data, orig_h, orig_w = self.preprocess(img)
        outputs = self.session.run(None, {self.input_name: input_data})
        results = self.postprocess(outputs, conf_thresh, iou_thresh)
        
        return results


# ==============================================
# Worker Thread
# ==============================================
class WorkerThread(QThread):
    progress_updated = Signal(int)
    result_ready = Signal(list)
    error_occurred = Signal(str)

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
                
                results.append({
                    'filename': os.path.basename(image_path),
                    'white': white_count,
                    'pink': pink_count,
                    'rate': survival_rate
                })
                
                self.progress_updated.emit(int((i + 1) / total * 100))
            except Exception as e:
                self.error_occurred.emit(f"Error processing {os.path.basename(image_path)}: {str(e)}")
                return
        
        self.result_ready.emit(results)


# ==============================================
# Sidebar Button
# ==============================================
class SidebarButton(QPushButton):
    def __init__(self, text, icon_path=None, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setMinimumHeight(50)
        self.setCursor(Qt.PointingHandCursor)
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


# ==============================================
# Modern Frame
# ==============================================
class ModernFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            ModernFrame {{
                background-color: white;
                border-radius: 16px;
            }}
        """)
        
        # Add premium drop shadow effect for cards
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(QColor(0, 0, 0, 40))  # Soft, subtle shadow
        self.setGraphicsEffect(shadow)


# ==============================================
# Main Application
# ==============================================
class SpermCounterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.setWindowTitle("Sperm Staining Analysis Tool")
        self.setGeometry(100, 100, 1200, 850)
        
        # Set minimum size to ensure it can be adjusted smaller
        self.setMinimumSize(850, 600)
        
        # Set window icon
        icon_path = get_resource_path("assets/logos/logo.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Set modern palette
        self.setup_palette()
        
        self.detector = None
        self.current_page = 0
        self.conf_thresh = self.config["default_conf_thresh"]
        self.iou_thresh = self.config["default_iou_thresh"]
        
        self.init_ui()
        self.load_model()

    def setup_palette(self):
        """Setup modern color palette"""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(THEME['content_bg']))
        palette.setColor(QPalette.WindowText, QColor(THEME['text_primary']))
        self.setPalette(palette)

    def load_model(self):
        """Load ONNX model"""
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
        """Initialize modern UI"""
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        sidebar = self.create_sidebar()
        main_layout.addWidget(sidebar)
        
        # Content area
        content_widget = self.create_content_area()
        main_layout.addWidget(content_widget, 1)

    def create_sidebar(self):
        """Create modern sidebar navigation"""
        sidebar = QWidget()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet(f"""
            QWidget {{
                background-color: {THEME['sidebar_bg']};
            }}
        """)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 30, 20, 30)
        layout.setSpacing(15)
        
        # Small logo in sidebar
        sidebar_logo_path = get_resource_path("assets/logos/logo.png")
        if os.path.exists(sidebar_logo_path):
            sidebar_logo_label = QLabel()
            sidebar_logo_pixmap = QPixmap(sidebar_logo_path)
            sidebar_logo_pixmap = sidebar_logo_pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            sidebar_logo_label.setPixmap(sidebar_logo_pixmap)
            sidebar_logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(sidebar_logo_label)
        
        # Logo/Title
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
        
        # Navigation buttons
        self.nav_group = QButtonGroup(self)
        
        self.home_btn = SidebarButton("Home")
        self.home_btn.setChecked(True)
        self.nav_group.addButton(self.home_btn, 0)
        
        self.analyze_btn = SidebarButton("Analyze")
        self.nav_group.addButton(self.analyze_btn, 1)
        
        self.settings_btn = SidebarButton("Settings")
        self.nav_group.addButton(self.settings_btn, 2)
        
        layout.addWidget(self.home_btn)
        layout.addWidget(self.analyze_btn)
        layout.addWidget(self.settings_btn)
        
        layout.addStretch()
        
        # Version label
        version_label = QLabel("v1.2.0")
        version_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_secondary']};
                font-size: 13px;
            }}
        """)
        layout.addWidget(version_label)
        
        # Connect navigation
        self.nav_group.buttonClicked.connect(self.switch_page)
        
        return sidebar

    def create_content_area(self):
        """Create main content area with stacked pages"""
        content = QWidget()
        content.setStyleSheet(f"""
            QWidget {{
                background-color: {THEME['content_bg']};
            }}
        """)
        
        layout = QVBoxLayout(content)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(25)
        
        # Stacked pages container
        self.pages = []
        
        # Home page
        home_page = self.create_home_page()
        self.pages.append(home_page)
        
        # Analyze page
        analyze_page = self.create_analyze_page()
        self.pages.append(analyze_page)
        
        # Settings page
        settings_page = self.create_settings_page()
        self.pages.append(settings_page)
        
        # Add initial page
        layout.addWidget(home_page)
        
        return content

    def create_home_page(self):
        """Create modern, premium home page with introduction"""
        # ==============================================
        # Modify this section to update home page content
        # ==============================================
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(25)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area to ensure all content is visible
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
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
        
        # Welcome section
        welcome_frame = ModernFrame()
        welcome_layout = QVBoxLayout(welcome_frame)
        welcome_layout.setContentsMargins(45, 45, 45, 45)
        welcome_layout.setAlignment(Qt.AlignCenter)
        
        # Logo
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
        
        # Features section
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
        
        # Quick start
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
            "6. Review the detailed results in the table and export to CSV if needed",
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
        """Create analyze page with parameter controls"""
        page = QWidget()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area to prevent component squeezing
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(25)
        
        # Parameters section
        params_frame = ModernFrame()
        params_layout = QVBoxLayout(params_frame)
        params_layout.setContentsMargins(40, 40, 40, 40)
        
        params_title = QLabel("Detection Parameters")
        params_title.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 24px;
                font-weight: 700;
                padding-bottom: 24px;
            }}
        """)
        params_layout.addWidget(params_title)
        
        params_grid = QHBoxLayout()
        
        # Confidence threshold
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
        self.conf_spin.setMinimumSize(0, 55)
        self.conf_spin.setMaximumSize(16777215, 55)
        self.conf_spin.setStyleSheet(f"""
            QDoubleSpinBox {{
                padding: 14px 18px;
                border: 2px solid {THEME['border_color']};
                border-radius: 12px;
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
        
        # IoU threshold
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
        self.iou_spin.setMinimumSize(0, 55)
        self.iou_spin.setMaximumSize(16777215, 55)
        self.iou_spin.setStyleSheet(f"""
            QDoubleSpinBox {{
                padding: 14px 18px;
                border: 2px solid {THEME['border_color']};
                border-radius: 12px;
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
        layout.addWidget(params_frame)
        
        # Input section
        input_frame = ModernFrame()
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(40, 40, 40, 40)
        
        input_title = QLabel("Input Selection")
        input_title.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 24px;
                font-weight: 700;
                padding-bottom: 24px;
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
        self.path_edit.setMinimumSize(0, 55)
        self.path_edit.setMaximumSize(16777215, 55)
        self.path_edit.setStyleSheet(f"""
            QLineEdit {{
                padding: 14px 18px;
                border: 2px solid {THEME['border_color']};
                border-radius: 12px;
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
        self.browse_btn.setMinimumSize(140, 55)
        self.browse_btn.setMaximumSize(140, 55)
        self.browse_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['sidebar_accent']};
                color: white;
                border: none;
                border-radius: 12px;
                padding: 14px 32px;
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
        layout.addWidget(input_frame)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.setMinimumSize(200, 58)
        self.start_btn.setMaximumSize(200, 58)
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
        
        self.save_btn = QPushButton("Save Results")
        self.save_btn.setEnabled(False)
        self.save_btn.setMinimumSize(200, 58)
        self.save_btn.setMaximumSize(200, 58)
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
        layout.addLayout(btn_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 12px;
                background-color: {THEME['border_color']};
                height: 28px;
                text-align: center;
                font-size: 14px;
                font-weight: 600;
            }}
            QProgressBar::chunk {{
                background-color: {THEME['sidebar_accent']};
                border-radius: 12px;
            }}
        """)
        layout.addWidget(self.progress_bar)
        
        # Results table
        table_frame = ModernFrame()
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(40, 40, 40, 40)
        
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Image Name", "Alive Count", "Dead Count", "Survival Rate"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                border: none;
                background-color: white;
                border-radius: 12px;
                gridline-color: {THEME['border_color']};
            }}
            QTableWidget::item {{
                padding: 14px;
                border: none;
            }}
            QHeaderView::section {{
                background-color: #f8fafc;
                color: {THEME['text_secondary']};
                padding: 18px;
                border: none;
                border-bottom: 2px solid {THEME['border_color']};
                font-weight: 700;
                font-size: 15px;
            }}
        """)
        table_layout.addWidget(self.table)
        layout.addWidget(table_frame, 1)
        
        # Stats label
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 17px;
                font-weight: 700;
                padding: 22px;
                background-color: white;
                border-radius: 14px;
            }}
        """)
        self.stats_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.stats_label)
        
        self.results = []
        
        layout.addStretch()
        
        # Set up scroll area
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
        
        return page

    def create_settings_page(self):
        """Create settings page"""
        page = QWidget()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Scroll area to prevent component squeezing
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(25)
        
        settings_frame = ModernFrame()
        settings_layout = QVBoxLayout(settings_frame)
        settings_layout.setContentsMargins(45, 45, 45, 45)
        
        settings_title = QLabel("Application Settings")
        settings_title.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 26px;
                font-weight: 700;
                padding-bottom: 32px;
            }}
        """)
        settings_layout.addWidget(settings_title)
        
        # Output directory setting
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
        self.output_dir_edit.setMinimumSize(0, 55)
        self.output_dir_edit.setMaximumSize(16777215, 55)
        self.output_dir_edit.setStyleSheet(f"""
            QLineEdit {{
                padding: 14px 18px;
                border: 2px solid {THEME['border_color']};
                border-radius: 12px;
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
        self.output_dir_btn.setMinimumSize(120, 55)
        self.output_dir_btn.setMaximumSize(120, 55)
        self.output_dir_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['sidebar_accent']};
                color: white;
                border: none;
                border-radius: 12px;
                padding: 14px 30px;
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
        
        settings_layout.addSpacing(28)
        
        # Default thresholds
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
        self.default_conf_spin.setMinimumSize(0, 50)
        self.default_conf_spin.setMaximumSize(16777215, 50)
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
        self.default_iou_spin.setMinimumSize(0, 50)
        self.default_iou_spin.setMaximumSize(16777215, 50)
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
        
        # Save button
        save_settings_layout = QHBoxLayout()
        self.save_settings_btn = QPushButton("Save Settings")
        self.save_settings_btn.setMinimumSize(190, 55)
        self.save_settings_btn.setMaximumSize(190, 55)
        self.save_settings_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['sidebar_accent']};
                color: white;
                border: none;
                border-radius: 14px;
                padding: 12px 20px;
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
        
        # Set up scroll area
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)
        
        return page

    def switch_page(self, button):
        """Switch between pages"""
        page_index = self.nav_group.id(button)
        if page_index == self.current_page:
            return
        
        # Remove old page
        content_layout = self.centralWidget().layout().itemAt(1).widget().layout()
        old_page = content_layout.itemAt(0).widget()
        old_page.setParent(None)
        
        # Add new page
        self.current_page = page_index
        content_layout.addWidget(self.pages[page_index])

    def browse_files(self):
        """Browse files or folder"""
        if self.folder_radio.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Select Folder")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        
        if path:
            self.path_edit.setText(path)

    def browse_output_dir(self):
        """Browse output directory"""
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_edit.setText(path)

    def save_settings(self):
        """Save application settings"""
        self.config["output_dir"] = self.output_dir_edit.text()
        self.config["default_conf_thresh"] = self.default_conf_spin.value()
        self.config["default_iou_thresh"] = self.default_iou_spin.value()
        
        # Also update current thresholds
        self.conf_thresh = self.config["default_conf_thresh"]
        self.iou_thresh = self.config["default_iou_thresh"]
        
        save_config(self.config)
        QMessageBox.information(self, "Success", "Settings saved successfully!")

    def start_analysis(self):
        """Start analysis process"""
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
        
        self.worker = WorkerThread(self.detector, image_paths, self.conf_thresh, self.iou_thresh)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.result_ready.connect(self.show_results)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.start()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def show_results(self, results):
        """Display analysis results"""
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

    def show_error(self, message):
        """Display error message"""
        QMessageBox.critical(self, "Error", message)
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)

    def save_results(self):
        """Save analysis results to file"""
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


# ==============================================
# Main Execution
# ==============================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpermCounterApp()
    window.show()
    sys.exit(app.exec())
