# SpermCounter - 精子染色状态统计工具

基于 YOLOv8 的精子检测与计数工具，支持检测白色存活精子和粉红色破碎精子。

## 功能特点

- 🖼️ 支持单张图片或批量文件夹处理
- 📊 输出统计表格（存活数、死亡数、存活率）
- 💾 结果可导出为 TXT 文件
- 🎨 可视化检测结果
- 🔄 跨平台支持（macOS / Windows / Linux）

## 下载

最新版本可从 [GitHub Actions Artifacts](https://github.com/benben-miao/SpermCounter/actions) 下载。

## 使用方法

### 运行应用
- **macOS**: 双击 `SpermCounter.app`
- **Windows**: 双击 `SpermCounter.exe`
- **Linux**: 终端运行 `./SpermCounter`

### 操作步骤
1. 点击「选择图片」或「选择文件夹」
2. 等待检测完成
3. 查看统计结果
4. 点击「导出结果」保存为 TXT 文件

## 输出格式

```
图片名称	存活数目	死亡数目	存活率
11-1.jpg	44	47	48.35%
12-4.jpg	12	56	17.65%
```

## 开发

### 安装依赖
```bash
pip install pyinstaller pyside6 ultralytics opencv-python numpy
```

### 运行源代码
```bash
python sperm_counter_gui.py
```

### 打包
```bash
pyinstaller sperm_counter.spec --clean --noconfirm
```

## 技术栈

- Python 3.11
- PySide6 (Qt GUI)
- YOLOv8 (目标检测)
- PyInstaller (打包)

## 项目结构

```
SpermCounter/
├── sperm_counter_gui.py    # 主应用脚本
├── sperm_counter.spec      # PyInstaller 配置
├── label_tool.py           # 交互式标注工具
├── convert_to_yolo.py      # 标注格式转换
├── train_yolo.py           # YOLO 训练脚本
├── runs/                   # YOLO 模型权重
├── photos/                 # 测试图片
└── labels.json             # 标注数据
```
