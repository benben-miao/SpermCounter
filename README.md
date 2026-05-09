# Sperm Staining Analysis Tool

A professional desktop application for sperm staining status analysis, built with Python and PySide6, using YOLOv8-based object detection to count white (alive) and pink (dead) sperm.

## Features

- 🎨 **Modern, Clean UI**: Beautiful interface with sidebar navigation and card-based design
- 🖼️ **Flexible Input**: Support for both single image and batch folder processing
- ⚙️ **Customizable Parameters**: Adjustable confidence threshold and IoU threshold
- 📊 **Comprehensive Results**: Detailed statistics including total count and survival rate
- 💾 **Export Support**: Results can be saved as CSV files with headers
- 🔧 **Persistent Settings**: Configure default output directory and thresholds that save automatically
- 🌐 **Cross-Platform**: Compatible with macOS, Windows, and Linux
- 📱 **Resizable Window**: Flexible window size with minimum dimensions for ease of use

## Download

The latest version can be downloaded from the [GitHub Actions Artifacts](https://github.com/benben-miao/SpermCounter/actions).

## Getting Started

### Running the Application

- **macOS**: Double-click `SpermCounter.app`
- **Windows**: Double-click `SpermCounter.exe`
- **Linux**: Run in terminal: `./SpermCounter`

### Using the Tool

1. **Go to Settings Page** (optional): Configure your default output directory and detection thresholds
2. **Go to Analyze Page**: 
   - Select either "Single Image" or "Folder" mode
   - Click "Browse" to choose your images
   - Adjust the Confidence Threshold and IoU Threshold as needed
3. **Start Analysis**: Click "Start Analysis" and wait for processing to complete
4. **View Results**: Check the detailed results table
5. **Save Results**: Click "Save Results" to export to CSV

## Output Format

The CSV export includes the following structure:

```csv
Image Name,Alive Count,Dead Count,Survival Rate
11-1.jpg,44,47,48.35%
12-4.jpg,12,56,17.65%

Overall Stats,56,103,35.22%
```

## Configuration

### Settings File Location

- **macOS / Linux**: `~/.sperm_counter/config.json`
- **Windows**: `C:\Users\<YourUsername>\.sperm_counter\config.json`

### Configuration Options

- **Output Directory**: Default location to save your results
- **Default Confidence Threshold**: Sets the default confidence level for detection
- **Default IoU Threshold**: Sets the default IoU level for non-maximum suppression

## Development

### Requirements

- Python 3.11+
- PySide6
- Ultralytics (YOLOv8)
- OpenCV
- NumPy
- PyInstaller (for building)
- ONNX Runtime (for inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/benben-miao/SpermCounter.git
cd SpermCounter

# Install dependencies
pip install -r requirements.txt
# Or install manually
pip install pyinstaller pyside6 ultralytics opencv-python numpy onnxruntime
```

### Running from Source

```bash
python sperm_counter_gui_onnx.py
```

### Building the Application

#### macOS Build

```bash
pyinstaller sperm_counter_onnx.spec --clean --noconfirm
```

#### Windows Build

```bash
pyinstaller sperm_counter_windows.spec --clean --noconfirm
```

## Customization

See [CUSTOMIZATION_GUIDE.md](CUSTOMIZATION_GUIDE.md) for detailed instructions on how to customize:
- Theme colors and styles
- Home page content
- UI elements
- Adding new features

## Assets

For logo and icon placement, refer to [ASSETS_GUIDE.md](ASSETS_GUIDE.md).

## Technology Stack

- **GUI**: PySide6 (Qt6)
- **Detection**: YOLOv8 (with ONNX Runtime)
- **Computer Vision**: OpenCV
- **Numerics**: NumPy
- **Packaging**: PyInstaller

## Project Structure

```
SpermCounter/
├── sperm_counter_gui_onnx.py    # Main application script (ONNX version)
├── sperm_counter_gui.py         # Original GUI script (ultralytics)
├── sperm_counter_onnx.spec      # macOS build configuration
├── sperm_counter_windows.spec   # Windows build configuration
├── sperm_counter.spec           # Legacy spec file
├── label_tool.py                # Interactive labeling tool
├── convert_to_yolo.py           # Label format conversion script
├── train_yolo.py                # YOLO training script
├── runs/                        # YOLO model weights (runs/detect/sperm_detection/weights)
├── photos/                      # Test images
├── assets/                      # Assets directory (logos, images)
│   └── logos/                   # Application logos (logo.icns, logo.ico)
├── labels.json                  # Labeling data
├── CUSTOMIZATION_GUIDE.md       # Customization guide
└── ASSETS_GUIDE.md              # Assets setup guide
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source.

## Acknowledgments

- YOLOv8 by Ultralytics for excellent object detection capabilities
- PySide6 for modern cross-platform GUI
- All contributors who helped improve this tool

## Support

If you encounter any issues or have questions, please open an issue on GitHub.
