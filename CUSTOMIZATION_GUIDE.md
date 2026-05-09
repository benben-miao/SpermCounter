# Sperm Analysis Tool - Customization Guide

## Overview
This guide explains how to customize the Sperm Analysis Tool to fit your specific needs.

## Theme Customization

### Modifying Colors
The theme colors are defined in the `THEME` dictionary at the top of `sperm_counter_gui_onnx.py`.

```python
THEME = {
    "sidebar_bg": "#1e293b",          # Sidebar background
    "sidebar_text": "#e0e0e0",         # Sidebar text
    "sidebar_accent": "#5b9bd5",       # Sidebar accent color
    "content_bg": "#ffffff",           # Content background (white)
    "text_primary": "#1e293b",         # Primary text color
    "text_secondary": "#64748b",       # Secondary text color
    "accent_green": "#10b981",         # Green accent (for buttons)
    "accent_blue": "#5b9bd5",          # Blue accent
    "border_color": "#e2e8f0",         # Border color
}
```

Simply change the hex values to your preferred colors.

### Modifying Styles
All widgets use Qt Style Sheets (QSS) for styling. You can find the style definitions in:
- `SidebarButton.__init__()` - Sidebar button styling
- `ModernFrame.__init__()` - Frame styling
- Widget-specific styles in each `create_*_page()` method

## Home Page Content

### Modifying the Home Page
The home page content is created in `create_home_page()` method. Look for this section:

```python
# ==============================================
# Modify this section to update home page content
# ==============================================
```

### Changing Welcome Text
Modify the `welcome_title` and `welcome_desc` QLabels:
```python
welcome_title = QLabel("Your Custom Title")
welcome_desc = QLabel("Your custom description")
```

### Changing Features
Update the `features` list:
```python
features = [
    ("🖼️", "Custom Feature 1", "Custom feature description"),
    ("⚡", "Custom Feature 2", "Another custom description"),
    # Add more features as needed
]
```

### Changing Quick Start Steps
Modify the `steps` list:
```python
steps = [
    "1. Your custom step 1",
    "2. Your custom step 2",
    # Add more steps as needed
]
```

## Settings Configuration

### Adding New Settings
To add new settings:
1. Add a default value in `load_config()`
2. Add UI elements in `create_settings_page()`
3. Save the value in `save_settings()`

### Configuration File Location
- **macOS/Linux**: `~/.sperm_counter/config.json`
- **Windows**: `C:\Users\<User>\.sperm_counter\config.json`

## App Metadata

### Changing App Name
Modify the `setWindowTitle()` call in `SpermCounterApp.__init__()`:
```python
self.setWindowTitle("Your App Name")
```

### Changing Version
Update the version label in `create_sidebar()`:
```python
version_label = QLabel("v2.0.0")
```

## Logo/Icon

### Adding Icons
1. **macOS**: Place `logo.icns` in `assets/logos/`
2. **Windows**: Place `logo.ico` in `assets/logos/`

For detailed instructions, see `ASSETS_GUIDE.md`.

## Building a Custom Version

### macOS Build
```bash
# Clean previous build
rm -rf build dist

# Build new version
pyinstaller sperm_counter_onnx.spec --clean --noconfirm

# Open the app
open dist/SpermCounter.app
```

### Windows Build
```bash
# Clean previous build
rmdir /s /q build dist

# Build new version
pyinstaller sperm_counter_windows.spec --clean --noconfirm
```

## Rebuilding After Changes
After making any customizations:
1. Save the file
2. Rebuild the app using the commands above
3. Test the changes

## Troubleshooting

### Styles Not Applying
- Ensure you're modifying the correct style definition
- Check that the widget uses the style via `setStyleSheet()`
- Rebuild the app after changes

### Configuration Not Saving
- Verify permissions on `~/.sperm_counter/`
- Check that `save_config()` is called after modifications

### App Won't Build
- Ensure all required packages are installed
- Check that PyInstaller is available
- Review the build output for errors
