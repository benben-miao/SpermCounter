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
    """获取资源文件路径（兼容打包后）"""
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
                self.error_occurred.emit(f"处理 {os.path.basename(image_path)} 时出错: {str(e)}")
                return
        
        self.result_ready.emit(results)

class SpermCounterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("精子染色状态统计工具")
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
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            self.model = YOLO(model_path)
            self.status_bar.showMessage("模型加载成功")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}\n\n请确保模型文件存在。")
            self.status_bar.showMessage(f"模型加载失败: {str(e)}")
            self.model = None

    def init_ui(self):
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        input_group = QGroupBox("输入选择")
        input_layout = QHBoxLayout(input_group)
        
        self.single_radio = QRadioButton("单张图片")
        self.folder_radio = QRadioButton("文件夹")
        self.folder_radio.setChecked(True)
        
        self.radio_group = QButtonGroup()
        self.radio_group.addButton(self.single_radio)
        self.radio_group.addButton(self.folder_radio)
        
        input_layout.addWidget(self.single_radio)
        input_layout.addWidget(self.folder_radio)
        
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("请选择图片或文件夹")
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_files)
        
        input_layout.addWidget(self.path_edit)
        input_layout.addWidget(self.browse_btn)
        
        main_layout.addWidget(input_group)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始统计")
        self.start_btn.clicked.connect(self.start_analysis)
        self.save_btn = QPushButton("保存结果")
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
        self.table.setHorizontalHeaderLabels(["图片名称", "存活数目", "死亡数目", "存活率"])
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
            path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        
        if path:
            self.path_edit.setText(path)

    def start_analysis(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "模型未加载，无法开始分析")
            return
        
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "警告", "请先选择图片或文件夹")
            return
        
        if not os.path.exists(path):
            QMessageBox.warning(self, "警告", "路径不存在")
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
                QMessageBox.warning(self, "警告", "文件夹中没有找到图片")
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
            f"共处理 {len(results)} 张图片 | 总计存活: {total_white} | 总计死亡: {total_pink} | 总体存活率: {overall_rate:.2f}%"
        )
        
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.status_bar.showMessage("分析完成")

    def show_error(self, message):
        QMessageBox.critical(self, "错误", message)
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.status_bar.showMessage("分析失败")

    def save_results(self):
        if not self.results:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "保存结果", "", "文本文件 (*.txt)")
        if not path:
            return
        
        if not path.endswith('.txt'):
            path += '.txt'
        
        try:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["图片名称", "存活数目", "死亡数目", "存活率"])
                
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
                writer.writerow(["总体统计", total_white, total_pink, f"{overall_rate:.2f}%"])
            
            QMessageBox.information(self, "成功", f"结果已保存到\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpermCounterApp()
    window.show()
    sys.exit(app.exec())
