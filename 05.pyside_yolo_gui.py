import sys
import os
import glob
import csv
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QProgressBar, QMessageBox, QHeaderView,
    QGroupBox, QRadioButton, QButtonGroup
)
from PySide6.QtGui import QPixmap, QIcon, QFont
from PySide6.QtCore import Qt, QThread, Signal
from ultralytics import YOLO
import cv2

def get_resource_path(relative_path):
    """Get resource file path (compatible with packaged version)"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

class WorkerThread(QThread):
    progress_updated = Signal(int)
    result_ready = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, model, image_paths):
        super().__init__()
        self.model = model
        self.image_paths = image_paths

    def run(self):
        results = []
        total = len(self.image_paths)
        
        for i, image_path in enumerate(self.image_paths):
            try:
                result = self.model(image_path, conf=0.1, iou=0.45)
                
                white_count = 0
                pink_count = 0
                
                for r in result:
                    boxes = r.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls == 0:
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

class SpermCounterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sperm Staining Status Counter")
        self.setGeometry(100, 100, 900, 600)
        
        self.model = None
        self.init_ui()
        self.load_model()

    def load_model(self):
        try:
            model_path = get_resource_path('runs/detect/sperm_detection/weights/best.pt')
            
            if not os.path.exists(model_path):
                model_path = get_resource_path('best.pt')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = YOLO(model_path)
            self.status_bar.showMessage("Model loaded successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Model loading failed: {str(e)}\n\nPlease ensure the model file exists.")
            self.status_bar.showMessage(f"Model loading failed: {str(e)}")
            self.model = None

    def init_ui(self):
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        input_group = QGroupBox("Input Selection")
        input_layout = QHBoxLayout(input_group)
        
        self.single_radio = QRadioButton("Single Image")
        self.folder_radio = QRadioButton("Folder")
        self.folder_radio.setChecked(True)
        
        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.single_radio)
        self.radio_group.addButton(self.folder_radio)
        
        input_layout.addWidget(self.single_radio)
        input_layout.addWidget(self.folder_radio)
        
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Please select image or folder")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_files)
        
        input_layout.addWidget(self.path_edit)
        input_layout.addWidget(self.browse_btn)
        
        main_layout.addWidget(input_group)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.clicked.connect(self.start_analysis)
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.save_btn)
        main_layout.addLayout(btn_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Image Name", "Live Count", "Dead Count", "Survival Rate"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        main_layout.addWidget(self.table)

        self.stats_label = QLabel()
        self.stats_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.stats_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.stats_label)

        self.results = []

    def browse_files(self):
        if self.folder_radio.isChecked():
            path = QFileDialog.getExistingDirectory(self, "Select Folder")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
        
        if path:
            self.path_edit.setText(path)

    def start_analysis(self):
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Model not loaded, cannot start analysis")
            return
        
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Warning", "Please select image or folder first")
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
        
        self.worker = WorkerThread(self.model, image_paths)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.result_ready.connect(self.show_results)
        self.worker.error_occurred.connect(self.show_error)
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
            f"Processed {len(results)} images | Total Alive: {total_white} | Total Dead: {total_pink} | Overall Survival Rate: {overall_rate:.2f}%"
        )
        
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.status_bar.showMessage("Analysis complete")

    def show_error(self, message):
        QMessageBox.critical(self, "Error", message)
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.status_bar.showMessage("Analysis failed")

    def save_results(self):
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to save")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "Text Files (*.txt)")
        if not path:
            return
        
        if not path.endswith('.txt'):
            path += '.txt'
        
        try:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
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
            
            QMessageBox.information(self, "Success", f"Results saved to\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpermCounterApp()
    window.show()
    sys.exit(app.exec())
