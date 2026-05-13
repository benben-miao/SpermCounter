import sys
import os
import glob
import csv
import json
import re
import html
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QProgressBar, QMessageBox, QHeaderView,
    QGroupBox, QRadioButton, QButtonGroup, QFrame, QScrollArea,
    QDoubleSpinBox, QSplitter, QComboBox, QCheckBox,
    QStackedWidget, QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QToolBar, QTextBrowser
)
from PySide6.QtGui import QPixmap, QIcon, QFont, QColor, QPalette, QImage, QPainter, QAction, QTextOption
from PySide6.QtCore import Qt, QThread, Signal, QRectF, QPropertyAnimation, QEasingCurve, QUrl, QSize

try:
    import qtawesome as qta
    HAS_QTAWESOME = True
except ImportError:
    HAS_QTAWESOME = False


def create_icon_label(icon_name, text, color=None, font_size=22):
    container = QWidget()
    container.setAttribute(Qt.WA_StyledBackground, True)
    container.setStyleSheet("background-color: transparent; border: none;")
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 12)
    layout.setSpacing(10)

    if HAS_QTAWESOME and color:
        icon_label = QLabel()
        icon_label.setPixmap(qta.icon(icon_name, color=color, scale=1.1).pixmap(font_size + 2, font_size + 2))
        icon_label.setFixedSize(font_size + 6, font_size + 6)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("background-color: transparent; border: none;")
        layout.addWidget(icon_label)

    text_label = QLabel(text)
    text_label.setStyleSheet(f"""
        QLabel {{
            color: {THEME['text_primary']};
            font-size: {font_size}px;
            font-weight: 700;
            background-color: transparent;
        }}
    """)
    layout.addWidget(text_label)
    layout.addStretch()
    return container


def create_field_label(icon_name, text, color=None, font_size=14, weight=600):
    return create_icon_label(icon_name, text, color or THEME['text_secondary'], font_size)


def apply_fontawesome_icon(button, icon_name, color="white", size=18):
    if HAS_QTAWESOME:
        button.setIcon(qta.icon(icon_name, color=color, scale=1.1))
        button.setIconSize(QSize(size, size))


THEME_PRESETS = {
    "Clinical Light": {
        "sidebar_bg": "#172033",
        "sidebar_text": "#dbeafe",
        "sidebar_accent": "#2563eb",
        "content_bg": "#f6f8fb",
        "surface": "#ffffff",
        "muted_surface": "#eef3f8",
        "text_primary": "#142033",
        "text_secondary": "#64748b",
        "accent_green": "#059669",
        "accent_blue": "#2563eb",
        "border_color": "#d8e0ea",
        "shadow": (15, 23, 42, 32),
    },
    "Graphite": {
        "sidebar_bg": "#101418",
        "sidebar_text": "#d5dde7",
        "sidebar_accent": "#38bdf8",
        "content_bg": "#171b21",
        "surface": "#20262e",
        "muted_surface": "#14191f",
        "text_primary": "#f2f6fb",
        "text_secondary": "#a9b4c2",
        "accent_green": "#34d399",
        "accent_blue": "#38bdf8",
        "border_color": "#303844",
        "shadow": (0, 0, 0, 70),
    },
    "Laboratory": {
        "sidebar_bg": "#12312d",
        "sidebar_text": "#d9fff6",
        "sidebar_accent": "#0f766e",
        "content_bg": "#f4faf8",
        "surface": "#ffffff",
        "muted_surface": "#e8f3ef",
        "text_primary": "#173330",
        "text_secondary": "#5f746f",
        "accent_green": "#16a34a",
        "accent_blue": "#0f766e",
        "border_color": "#cfddd8",
        "shadow": (15, 42, 36, 30),
    },
}
THEME = THEME_PRESETS["Laboratory"].copy()


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
        "theme": "Laboratory",
        "show_confidence": True,
        "auto_open_viewer": True,
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
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        
        nms_boxes = []
        for x_center, y_center, box_w, box_h in boxes:
            nms_boxes.append([
                float(x_center - box_w / 2),
                float(y_center - box_h / 2),
                float(box_w),
                float(box_h),
            ])
        
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores.tolist(), conf_thresh, iou_thresh)
        
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
    
    def visualize(self, image_path, detections, show_confidence=True):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        h, w = img.shape[:2]
        img_copy = img.copy()
        
        for det in detections:
            box = np.array(det['box'])
            x_center, y_center, box_w, box_h = box
            scale_x = w / float(self.img_size)
            scale_y = h / float(self.img_size)
            x1 = int((x_center - box_w / 2) * scale_x)
            y1 = int((y_center - box_h / 2) * scale_y)
            x2 = int((x_center + box_w / 2) * scale_x)
            y2 = int((y_center + box_h / 2) * scale_y)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            
            if det['class_id'] == 0:
                color = (0, 255, 0)
                label = "Alive"
            else:
                color = (0, 0, 255)
                label = "Dead"
            
            line_width = max(2, int(min(w, h) / 300))
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, line_width)
            
            text = f"{label}: {det['score']:.2f}" if show_confidence else label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.5, min(w, h) / 1100)
            thickness = max(1, line_width - 1)
            
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
    
    def __init__(self, detector, image_paths, conf_thresh, iou_thresh, show_confidence=True):
        super().__init__()
        self.detector = detector
        self.image_paths = image_paths
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.show_confidence = show_confidence
    
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
                        annotated_img = self.detector.visualize(image_path, detections, self.show_confidence)
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
        self.setStyleSheet(f"background-color: {THEME['muted_surface']}; border-radius: 12px;")
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
    def __init__(self, icon_name, text, parent=None):
        super().__init__(text, parent)
        self.icon_name = icon_name
        self.label = text
        self.setIconSize(QSize(20, 20))
        self.setToolTip(text)
        self.setCheckable(True)
        self.setMinimumHeight(50)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.update_icon(False)
    
    def update_icon(self, collapsed):
        if HAS_QTAWESOME:
            icon_color = THEME['sidebar_text']
            if self.isChecked():
                icon_color = 'white'
            
            icon = qta.icon(
                self.icon_name,
                color=icon_color,
                scale=1.3,
            )
            self.setIcon(icon)
            self.setIconSize(QSize(20, 20))
        
        if collapsed:
            self.setText("")
            self.setMinimumWidth(76)
            self.setMaximumWidth(76)
        else:
            self.setText(f"  {self.label}")
            self.setMaximumWidth(16777215)
        
        self.setStyleSheet(f"""
            SidebarButton {{
                background-color: transparent;
                color: {THEME['sidebar_text']};
                border: none;
                border-radius: 12px;
                padding: {8 if collapsed else 12}px {16 if collapsed else 20}px;
                text-align: { 'center' if collapsed else 'left' };
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
    
    def set_collapsed(self, collapsed):
        self.update_icon(collapsed)


class ModernFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(f"""
            ModernFrame {{
                background-color: {THEME['surface']};
                border: 1px solid {THEME['border_color']};
                border-radius: 12px;
            }}
        """)


class SpermCounterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.apply_theme_name(self.config.get("theme", "Laboratory"))
        self.setWindowTitle("Sperm Staining Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 700)
        
        icon_path = get_resource_path("assets/logos/logo.png")
        if os.path.exists(icon_path):
            window_icon = QIcon(icon_path)
            self.setWindowIcon(window_icon)
        elif HAS_QTAWESOME:
            microscope_icon = qta.icon('fa5s.microscope', color='#0f766e', scale=3)
            self.setWindowIcon(microscope_icon)
        
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
    
    def apply_theme_name(self, theme_name):
        if theme_name not in THEME_PRESETS:
            theme_name = "Laboratory"
        THEME.clear()
        THEME.update(THEME_PRESETS[theme_name])
        self.config["theme"] = theme_name
    
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
        self.sidebar_widget.setObjectName("sidebarRoot")
        self.sidebar_widget.setFixedWidth(260)
        self.sidebar_widget.setStyleSheet(f"""
            #sidebarRoot {{
                background-color: {THEME['sidebar_bg']};
            }}
        """)
        
        layout = QVBoxLayout(self.sidebar_widget)
        layout.setContentsMargins(16, 24, 16, 24)
        layout.setSpacing(15)
        
        header_layout = QHBoxLayout()
        
        self.toggle_btn = QPushButton()
        self.toggle_btn.setFixedSize(44, 34)
        if HAS_QTAWESOME:
            toggle_icon = qta.icon("fa5s.bars", color=THEME['sidebar_text'], scale=1.2)
            self.toggle_btn.setIcon(toggle_icon)
            self.toggle_btn.setIconSize(QSize(20, 20))
        else:
            self.toggle_btn.setText("≡")
        self.toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {THEME['sidebar_text']};
                border: none;
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 10px;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: rgba(255, 255, 255, 0.1);
            }}
        """)
        self.toggle_btn.clicked.connect(self.toggle_sidebar)
        header_layout.addWidget(self.toggle_btn, 0, Qt.AlignLeft)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        sidebar_logo_path = get_resource_path("assets/logos/logo.png")
        if os.path.exists(sidebar_logo_path):
            self.sidebar_logo_label = QLabel()
            sidebar_logo_pixmap = QPixmap(sidebar_logo_path)
            sidebar_logo_pixmap = sidebar_logo_pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.sidebar_logo_label.setPixmap(sidebar_logo_pixmap)
            self.sidebar_logo_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.sidebar_logo_label)
        
        self.sidebar_title_label = QLabel("Sperm Analysis")
        self.sidebar_title_label.setStyleSheet(f"""
            QLabel {{
                color: white;
                font-size: 34px;
                font-weight: 800;
                padding: 10px 0;
            }}
        """)
        layout.addWidget(self.sidebar_title_label)
        
        self.sidebar_subtitle_label = QLabel("Staining Status Tool")
        self.sidebar_subtitle_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_secondary']};
                font-size: 14px;
                padding-bottom: 20px;
            }}
        """)
        layout.addWidget(self.sidebar_subtitle_label)
        
        self.nav_group = QButtonGroup(self)
        
        self.home_btn = SidebarButton("fa5s.home", "Home")
        self.home_btn.setChecked(True)
        self.nav_group.addButton(self.home_btn, 0)
        
        self.analyze_btn = SidebarButton("fa5s.microscope", "Analyze")
        self.nav_group.addButton(self.analyze_btn, 1)
        
        self.settings_btn = SidebarButton("fa5s.cog", "Settings")
        self.nav_group.addButton(self.settings_btn, 2)
        
        layout.addWidget(self.home_btn)
        layout.addWidget(self.analyze_btn)
        layout.addWidget(self.settings_btn)
        
        layout.addStretch()
        
        self.sidebar_version_label = QLabel("v1.0.5")
        self.sidebar_version_label.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_secondary']};
                font-size: 13px;
            }}
        """)
        layout.addWidget(self.sidebar_version_label)
        
        self.nav_group.buttonClicked.connect(self.switch_page)
        
        return self.sidebar_widget
    
    def toggle_sidebar(self):
        if self.sidebar_expanded:
            target_width = 76
            if HAS_QTAWESOME:
                toggle_icon = qta.icon("fa5s.bars", color=THEME['sidebar_text'], scale=1.2)
                self.toggle_btn.setIcon(toggle_icon)
                self.toggle_btn.setIconSize(QSize(20, 20))
                self.toggle_btn.setText("")
            else:
                self.toggle_btn.setText("≡")
            self.toggle_btn.setFixedSize(44, 34)
            for widget in (self.home_btn, self.analyze_btn, self.settings_btn):
                widget.set_collapsed(True)
            for widget in (self.sidebar_title_label, self.sidebar_subtitle_label, self.sidebar_version_label):
                widget.setVisible(False)
            if hasattr(self, "sidebar_logo_label"):
                self.sidebar_logo_label.setVisible(False)
        else:
            target_width = 260
            if HAS_QTAWESOME:
                toggle_icon = qta.icon("fa5s.bars", color=THEME['sidebar_text'], scale=1.2)
                self.toggle_btn.setIcon(toggle_icon)
                self.toggle_btn.setIconSize(QSize(20, 20))
                self.toggle_btn.setText("")
            else:
                self.toggle_btn.setText("≡")
            self.toggle_btn.setFixedSize(44, 34)
            for widget in (self.home_btn, self.analyze_btn, self.settings_btn):
                widget.set_collapsed(False)
            for widget in (self.sidebar_title_label, self.sidebar_subtitle_label, self.sidebar_version_label):
                widget.setVisible(True)
            if hasattr(self, "sidebar_logo_label"):
                self.sidebar_logo_label.setVisible(True)
        
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
        content.setObjectName("contentRoot")
        content.setStyleSheet(f"""
            #contentRoot {{
                background-color: {THEME['content_bg']};
                color: {THEME['text_primary']};
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
        layout.setSpacing(16)
        layout.setContentsMargins(0, 0, 0, 0)
        
        header = QHBoxLayout()
        title = QLabel("Sperm Staining Analysis")
        title.setStyleSheet(f"""
            QLabel {{
                color: {THEME['text_primary']};
                font-size: 28px;
                font-weight: 800;
            }}
        """)
        header.addWidget(title)
        header.addStretch()
        
        badge = QLabel("README")
        badge.setStyleSheet(f"""
            QLabel {{
                color: white;
                background-color: {THEME['sidebar_accent']};
                border-radius: 9px;
                padding: 6px 13px;
                font-size: 13px;
                font-weight: 700;
            }}
        """)
        header.addWidget(badge)
        
        refresh_btn = QPushButton("Refresh")
        apply_fontawesome_icon(refresh_btn, "fa5s.sync-alt", "white", 14)
        refresh_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {THEME['sidebar_accent']};
                color: white;
                border: none;
                border-radius: 9px;
                padding: 6px 13px;
                font-size: 13px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: {THEME['accent_green']};
            }}
        """)
        refresh_btn.clicked.connect(self.refresh_readme)
        header.addWidget(refresh_btn)
        
        layout.addLayout(header)
        
        self.readme_browser = QTextBrowser()
        self.readme_browser.setOpenExternalLinks(True)
        self.readme_browser.setLineWrapMode(QTextBrowser.WidgetWidth)
        self.readme_browser.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.readme_browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.readme_browser.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.readme_browser.setAcceptRichText(True)
        self.readme_browser.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {THEME['surface']};
                color: {THEME['text_primary']};
                border: 1px solid {THEME['border_color']};
                border-radius: 12px;
                padding: 28px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
                font-size: 16px;
                line-height: 1.7;
            }}
            QTextBrowser h1 {{
                color: {THEME['text_primary']};
                font-size: 32px;
                font-weight: 700;
                margin-top: 0px;
                margin-bottom: 16px;
                padding-bottom: 8px;
                border-bottom: 2px solid {THEME['border_color']};
            }}
            QTextBrowser h2 {{
                color: {THEME['text_primary']};
                font-size: 24px;
                font-weight: 650;
                margin-top: 32px;
                margin-bottom: 16px;
                padding-bottom: 6px;
                border-bottom: 1px solid {THEME['border_color']};
            }}
            QTextBrowser h3 {{
                color: {THEME['text_primary']};
                font-size: 20px;
                font-weight: 600;
                margin-top: 24px;
                margin-bottom: 12px;
            }}
            QTextBrowser h4 {{
                color: {THEME['text_primary']};
                font-size: 16px;
                font-weight: 600;
                margin-top: 20px;
                margin-bottom: 10px;
            }}
            QTextBrowser p {{
                color: {THEME['text_primary']};
                margin-top: 12px;
                margin-bottom: 12px;
                text-align: justify;
            }}
            QTextBrowser pre {{
                background-color: {THEME['muted_surface']};
                color: {THEME['text_primary']};
                border: 1px solid {THEME['border_color']};
                border-radius: 8px;
                padding: 18px;
                white-space: pre-wrap;
                font-family: 'Menlo', 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 14px;
                margin: 16px 0;
                line-height: 1.5;
            }}
            QTextBrowser code {{
                background-color: {THEME['muted_surface']};
                color: {THEME['text_primary']};
                font-family: 'Menlo', 'Consolas', 'Monaco', 'Courier New', monospace;
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 14px;
            }}
            QTextBrowser pre code {{
                background-color: transparent;
                padding: 0;
            }}
            QTextBrowser table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                border-radius: 8px;
                overflow: hidden;
            }}
            QTextBrowser tr {{
                border-bottom: 1px solid {THEME['border_color']};
            }}
            QTextBrowser tr:last-child {{
                border-bottom: none;
            }}
            QTextBrowser th {{
                padding: 12px 16px;
                border: 1px solid {THEME['border_color']};
                background-color: {THEME['muted_surface']};
                font-weight: 650;
                color: {THEME['text_primary']};
                text-align: left;
                font-size: 15px;
            }}
            QTextBrowser td {{
                padding: 12px 16px;
                border: 1px solid {THEME['border_color']};
                color: {THEME['text_primary']};
                font-size: 15px;
            }}
            QTextBrowser ul {{
                margin: 12px 0;
                padding-left: 32px;
            }}
            QTextBrowser ol {{
                margin: 12px 0;
                padding-left: 32px;
            }}
            QTextBrowser li {{
                margin: 6px 0;
                line-height: 1.6;
            }}
            QTextBrowser li::marker {{
                color: {THEME['sidebar_accent']};
            }}
            QTextBrowser img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 6px 0 10px 0;
                border-radius: 10px;
            }}
            QTextBrowser a {{
                color: {THEME['sidebar_accent']};
                text-decoration: none;
                font-weight: 500;
            }}
            QTextBrowser a:hover {{
                text-decoration: underline;
            }}
            QTextBrowser blockquote {{
                border-left: 4px solid {THEME['sidebar_accent']};
                margin: 16px 0;
                padding-left: 20px;
                padding-right: 20px;
                color: {THEME['text_secondary']};
                background-color: {THEME['muted_surface']};
                border-radius: 0 8px 8px 0;
            }}
            QTextBrowser hr {{
                border: none;
                border-top: 2px solid {THEME['border_color']};
                margin: 24px 0;
            }}
        """)
        self.readme_browser.setHtml(self.load_readme_html())
        layout.addWidget(self.readme_browser, 1)
        
        return page
    
    def load_readme_html(self):
        readme_path = get_resource_path("README.md")
        if not os.path.exists(readme_path):
            readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                markdown = f.read()
                return self.build_readme_html(markdown)
        except Exception as e:
            message = html.escape(f"README.md could not be loaded: {str(e)}")
            return f"<h1>Sperm Staining Analysis</h1><p>{message}</p>"
    
    def build_readme_html(self, markdown):
        body = self.markdown_to_html(markdown)
        return f"""
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    margin: 0;
                    color: {THEME['text_primary']};
                    background: {THEME['surface']};
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
                    font-size: 16px;
                    line-height: 1.65;
                }}
                h1 {{
                    font-size: 32px;
                    margin: 0 0 18px 0;
                    padding-bottom: 10px;
                    border-bottom: 2px solid {THEME['border_color']};
                }}
                h2 {{
                    font-size: 24px;
                    margin: 30px 0 14px 0;
                    padding-bottom: 6px;
                    border-bottom: 1px solid {THEME['border_color']};
                }}
                h3 {{ font-size: 20px; margin: 18px 0 6px 0; }}
                h4 {{ font-size: 17px; margin: 20px 0 10px 0; }}
                p {{ margin: 12px 0; }}
                .readme-image {{
                    margin: 8px 0 16px 0;
                    padding: 0;
                    line-height: 1;
                }}
                a {{ color: {THEME['sidebar_accent']}; text-decoration: none; }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0;
                    border-radius: 10px;
                }}
                pre {{
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    overflow-wrap: anywhere;
                    background: {THEME['muted_surface']};
                    border: 1px solid {THEME['border_color']};
                    border-radius: 8px;
                    padding: 14px;
                    margin: 14px 0;
                    font-family: Menlo, Consolas, Monaco, monospace;
                    font-size: 13px;
                    line-height: 1.5;
                }}
                code {{
                    background: {THEME['muted_surface']};
                    border-radius: 4px;
                    padding: 2px 5px;
                    font-family: Menlo, Consolas, Monaco, monospace;
                    font-size: 13px;
                }}
                pre code {{ background: transparent; padding: 0; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 16px 0;
                }}
                th, td {{
                    border: 1px solid {THEME['border_color']};
                    padding: 8px 10px;
                    vertical-align: top;
                }}
                th {{ background: {THEME['muted_surface']}; }}
                blockquote {{
                    border-left: 4px solid {THEME['sidebar_accent']};
                    background: {THEME['muted_surface']};
                    margin: 14px 0;
                    padding: 10px 14px;
                }}
            </style>
        </head>
        <body>{body}</body>
        </html>
        """
    
    def markdown_to_html(self, markdown):
        lines = markdown.splitlines()
        out = []
        in_code_block = False
        code_block = []
        in_list = False
        list_type = "ul"
        table_rows = []
        
        def close_list():
            nonlocal in_list
            if in_list:
                out.append(f"</{list_type}>")
                in_list = False
        
        def flush_table():
            nonlocal table_rows
            if table_rows:
                out.extend(self.process_table(table_rows))
                table_rows = []
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith("```"):
                if in_code_block:
                    out.append("<pre><code>" + html.escape("\n".join(code_block)) + "</code></pre>")
                    code_block = []
                    in_code_block = False
                else:
                    close_list()
                    flush_table()
                    in_code_block = True
                continue
            
            if in_code_block:
                code_block.append(line)
                continue
            
            if stripped.startswith("|") and stripped.endswith("|"):
                close_list()
                if not re.match(r"^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", stripped):
                    table_rows.append([cell.strip() for cell in stripped.split("|")[1:-1]])
                continue
            flush_table()
            
            image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
            if image_match:
                close_list()
                out.append(self.render_image(image_match.group(1), image_match.group(2)))
            elif stripped.startswith("# "):
                close_list()
                out.append(f"<h1>{html.escape(stripped[2:])}</h1>")
            elif stripped.startswith("## "):
                close_list()
                out.append(f"<h2>{html.escape(stripped[3:])}</h2>")
            elif stripped.startswith("### "):
                close_list()
                out.append(f"<h3>{html.escape(stripped[4:])}</h3>")
            elif stripped.startswith("#### "):
                close_list()
                out.append(f"<h4>{html.escape(stripped[5:])}</h4>")
            elif stripped.startswith("- ") or stripped.startswith("* "):
                if not in_list or list_type != "ul":
                    close_list()
                    out.append("<ul>")
                    in_list = True
                    list_type = "ul"
                out.append(f"<li>{self.process_inline_formatting(html.escape(stripped[2:]))}</li>")
            elif re.match(r"^\d+\. ", stripped):
                if not in_list or list_type != "ol":
                    close_list()
                    out.append("<ol>")
                    in_list = True
                    list_type = "ol"
                item = re.sub(r"^\d+\. ", "", stripped)
                out.append(f"<li>{self.process_inline_formatting(html.escape(item))}</li>")
            elif stripped.startswith("> "):
                close_list()
                out.append(f"<blockquote>{self.process_inline_formatting(html.escape(stripped[2:]))}</blockquote>")
            elif stripped in ("---", "***", "___"):
                close_list()
                out.append("<hr>")
            elif stripped:
                close_list()
                out.append(f"<p>{self.process_inline_formatting(html.escape(stripped))}</p>")
            else:
                close_list()
        
        if in_code_block:
            out.append("<pre><code>" + html.escape("\n".join(code_block)) + "</code></pre>")
        close_list()
        flush_table()
        return "\n".join(out)
    
    def render_image(self, alt_text, image_path):
        image_path = image_path.strip()
        if image_path.startswith(("http://", "https://", "file://")):
            src = image_path
        else:
            readme_base = sys._MEIPASS if hasattr(sys, "_MEIPASS") else os.path.dirname(os.path.abspath(__file__))
            src = QUrl.fromLocalFile(os.path.abspath(os.path.join(readme_base, image_path))).toString()
        return f'<p class="readme-image"><img src="{src}" alt="{html.escape(alt_text)}"></p>'
    
    def escape_html(self, text):
        """Escape HTML special characters"""
        return html.escape(text)
    
    def process_inline_formatting(self, text):
        """Process inline formatting like bold, italic, code"""
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
        text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
        return text
    
    def process_table(self, rows):
        """Process table rows into HTML"""
        if not rows:
            return []
        
        result = []
        result.append('<table>')
        
        if rows:
            result.append('<thead><tr>')
            for cell in rows[0]:
                result.append(f'<th>{self.process_inline_formatting(html.escape(cell))}</th>')
            result.append('</tr></thead>')
        
        if len(rows) > 1:
            result.append('<tbody>')
            for row in rows[1:]:
                result.append('<tr>')
                for cell in row:
                    result.append(f'<td>{self.process_inline_formatting(html.escape(cell))}</td>')
                result.append('</tr>')
            result.append('</tbody>')
        
        result.append('</table>')
        return result
    
    def refresh_readme(self):
        """Refresh the README display"""
        try:
            self.readme_browser.setHtml(self.load_readme_html())
            QMessageBox.information(self, "Success", "README refreshed successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to refresh README: {str(e)}")
    
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
        
        params_title = create_icon_label("fa5s.bullseye", "Detection Parameters", THEME['sidebar_accent'], 22)
        params_layout.addWidget(params_title)
        
        params_grid = QHBoxLayout()
        
        conf_layout = QVBoxLayout()
        conf_label = create_field_label("fa5s.sliders-h", "Confidence Threshold", THEME['text_secondary'], 15)
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
                background-color: {THEME['surface']};
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
        iou_label = create_field_label("fa5s.crosshairs", "IoU Threshold", THEME['text_secondary'], 15)
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
                background-color: {THEME['surface']};
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
        
        input_title = create_icon_label("fa5s.folder-open", "Input Selection", THEME['sidebar_accent'], 22)
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
                background-color: {THEME['surface']};
                font-size: 16px;
                font-weight: 500;
            }}
            QLineEdit:focus {{
                border-color: {THEME['sidebar_accent']};
            }}
        """)
        path_layout.addWidget(self.path_edit, 1)
        
        self.browse_btn = QPushButton("Browse...")
        apply_fontawesome_icon(self.browse_btn, "fa5s.folder-open", "white", 16)
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
        
        self.start_btn = QPushButton("Start Analysis")
        apply_fontawesome_icon(self.start_btn, "fa5s.play", "white", 16)
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
        
        self.save_btn = QPushButton("Save Results")
        apply_fontawesome_icon(self.save_btn, "fa5s.save", "white", 16)
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
        
        table_title = create_icon_label("fa5s.chart-bar", "Analysis Results", THEME['accent_blue'], 20)
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
                background-color: {THEME['surface']};
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
                background-color: {THEME['muted_surface']};
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
                background-color: {THEME['surface']};
                border-radius: 12px;
            }}
        """)
        self.stats_label.setAlignment(Qt.AlignCenter)
        scroll_layout.addWidget(self.stats_label)
        
        self.viewer_frame = ModernFrame()
        viewer_frame_layout = QVBoxLayout(self.viewer_frame)
        viewer_frame_layout.setContentsMargins(30, 30, 30, 30)
        
        viewer_title = create_icon_label("fa5s.images", "Image Viewer - Click on table row to view", THEME['sidebar_accent'], 20)
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
                background-color: {THEME['surface']};
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
        apply_fontawesome_icon(self.reset_btn, "fa5s.expand-arrows-alt", "white", 14)
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
        
        original_label = create_icon_label("fa5s.camera", "Original Image", THEME['sidebar_accent'], 16)
        
        annotated_label = create_icon_label("fa5s.image", "Annotated Image", THEME['accent_green'], 16)
        
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
        
        settings_title = create_icon_label("fa5s.cogs", "Application Settings", THEME['sidebar_accent'], 24)
        settings_layout.addWidget(settings_title)
        
        output_layout = QVBoxLayout()
        output_label = create_field_label("fa5s.folder", "Default Output Directory", THEME['text_secondary'], 15)
        output_layout.addWidget(output_label)
        
        output_path_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit(self.config["output_dir"])
        self.output_dir_edit.setMinimumHeight(50)
        self.output_dir_edit.setStyleSheet(f"""
            QLineEdit {{
                padding: 12px 16px;
                border: 2px solid {THEME['border_color']};
                border-radius: 10px;
                background-color: {THEME['surface']};
                font-size: 16px;
                font-weight: 500;
            }}
            QLineEdit:focus {{
                border-color: {THEME['sidebar_accent']};
            }}
        """)
        output_path_layout.addWidget(self.output_dir_edit, 1)
        
        self.output_dir_btn = QPushButton("Browse")
        apply_fontawesome_icon(self.output_dir_btn, "fa5s.folder-open", "white", 15)
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
        thresholds_title = create_field_label("fa5s.tachometer-alt", "Default Detection Thresholds", THEME['text_secondary'], 15)
        thresholds_layout.addWidget(thresholds_title)
        
        thresholds_grid = QHBoxLayout()
        
        default_conf_layout = QVBoxLayout()
        default_conf_label = create_field_label("fa5s.sliders-h", "Default Confidence Threshold", THEME['text_secondary'], 14)
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
                background-color: {THEME['surface']};
                font-size: 15px;
            }}
            QDoubleSpinBox:focus {{
                border-color: {THEME['sidebar_accent']};
            }}
        """)
        default_conf_layout.addWidget(self.default_conf_spin)
        thresholds_grid.addLayout(default_conf_layout, 1)
        
        default_iou_layout = QVBoxLayout()
        default_iou_label = create_field_label("fa5s.crosshairs", "Default IoU Threshold", THEME['text_secondary'], 14)
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
                background-color: {THEME['surface']};
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

        appearance_layout = QVBoxLayout()
        appearance_title = create_icon_label("fa5s.paint-brush", "Appearance and Workflow", THEME['text_secondary'], 15)
        appearance_layout.addWidget(appearance_title)

        appearance_grid = QHBoxLayout()
        theme_layout = QVBoxLayout()
        theme_label = create_field_label("fa5s.palette", "Theme", THEME['text_secondary'], 14)
        theme_layout.addWidget(theme_label)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(THEME_PRESETS.keys()))
        self.theme_combo.setCurrentText(self.config.get("theme", "Laboratory"))
        self.theme_combo.setMinimumHeight(50)
        self.theme_combo.setStyleSheet(f"""
            QComboBox {{
                padding: 10px 16px;
                border: 2px solid {THEME['border_color']};
                border-radius: 10px;
                background-color: {THEME['surface']};
                color: {THEME['text_primary']};
                font-size: 15px;
                font-weight: 500;
            }}
            QComboBox:focus {{
                border-color: {THEME['sidebar_accent']};
            }}
        """)
        theme_layout.addWidget(self.theme_combo)
        appearance_grid.addLayout(theme_layout, 1)

        options_layout = QVBoxLayout()
        options_label = create_field_label("fa5s.eye", "Display Options", THEME['text_secondary'], 14)
        options_layout.addWidget(options_label)

        self.show_conf_checkbox = QCheckBox("Show confidence labels on annotated images")
        self.show_conf_checkbox.setChecked(self.config.get("show_confidence", True))
        self.auto_open_viewer_checkbox = QCheckBox("Open image viewer automatically after analysis")
        self.auto_open_viewer_checkbox.setChecked(self.config.get("auto_open_viewer", True))
        for checkbox in (self.show_conf_checkbox, self.auto_open_viewer_checkbox):
            checkbox.setStyleSheet(f"""
                QCheckBox {{
                    color: {THEME['text_primary']};
                    background-color: transparent;
                    border: none;
                    font-size: 14px;
                    padding: 5px 0;
                }}
                QCheckBox::indicator {{
                    width: 18px;
                    height: 18px;
                }}
            """)
            options_layout.addWidget(checkbox)
        appearance_grid.addLayout(options_layout, 1)
        appearance_layout.addLayout(appearance_grid)
        settings_layout.addLayout(appearance_layout)
        
        settings_layout.addSpacing(35)
        
        save_settings_layout = QHBoxLayout()
        self.save_settings_btn = QPushButton("Save Settings")
        apply_fontawesome_icon(self.save_settings_btn, "fa5s.save", "white", 16)
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
        previous_theme = self.config.get("theme", "Laboratory")
        self.config["output_dir"] = self.output_dir_edit.text()
        self.config["default_conf_thresh"] = self.default_conf_spin.value()
        self.config["default_iou_thresh"] = self.default_iou_spin.value()
        self.config["theme"] = self.theme_combo.currentText()
        self.config["show_confidence"] = self.show_conf_checkbox.isChecked()
        self.config["auto_open_viewer"] = self.auto_open_viewer_checkbox.isChecked()
        
        self.conf_thresh = self.config["default_conf_thresh"]
        self.iou_thresh = self.config["default_iou_thresh"]
        
        self.conf_spin.setValue(self.conf_thresh)
        self.iou_spin.setValue(self.iou_thresh)
        
        save_config(self.config)
        if self.config["theme"] != previous_theme:
            page_index = self.current_page
            self.apply_theme_name(self.config["theme"])
            self.setup_palette()
            self.init_ui()
            self.current_page = page_index
            if page_index != 0:
                self.nav_group.button(page_index).setChecked(True)
                content_layout = self.centralWidget().layout().itemAt(1).widget().layout()
                old_page = content_layout.itemAt(0).widget()
                old_page.setParent(None)
                content_layout.addWidget(self.pages[page_index])
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
        
        self.worker = WorkerThread(
            self.detector,
            image_paths,
            self.conf_thresh,
            self.iou_thresh,
            self.config.get("show_confidence", True),
        )
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
        
        if len(results) > 0 and self.config.get("auto_open_viewer", True):
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
                        annotated_img = self.detector.visualize(
                            image_path,
                            result.get('detections', []),
                            self.config.get("show_confidence", True),
                        )
                        
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
