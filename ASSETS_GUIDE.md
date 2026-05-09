# Assets Guide

## Logo Placement

Place your logo files in the `assets/logos/` directory:

### macOS
- **File**: `assets/logos/logo.icns`
- **Format**: ICNS (Apple icon format)
- **Sizes**: Include 16x16, 32x32, 64x64, 128x128, 256x256, 512x512, and 1024x1024 (Retina)

### Windows
- **File**: `assets/logos/logo.ico`
- **Format**: ICO (Windows icon format)
- **Sizes**: Include 16x16, 32x32, 48x48, 64x64, 128x128, and 256x256

## How to Create Logos

### Using Online Tools
1. **macOS**: Use [CloudConvert](https://cloudconvert.com/png-to-icns) or [ICNS Maker](https://icnsmaker.com/)
2. **Windows**: Use [Convertio](https://convertio.co/png-to-ico/) or [ICO Convert](https://www.icoconvert.com/)

### Using Command Line (macOS)
```bash
# Create ICNS from PNG
iconutil -c icns your_logo.iconset
```

### Using ImageMagick
```bash
# Create ICO with multiple sizes
convert your_logo.png -define icon:auto-resize=256,128,64,48,32,16 logo.ico
```

## Other Images

Place additional images in `assets/images/` directory. These can be used in the home page or other parts of the application.

## Note

If you don't have a logo yet, the application will still work fine - it just won't have a custom icon.
