# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['sperm_counter_gui.py'],
    pathex=['/Users/benbenmiao/Research/04.Code/RanSe'],
    binaries=[],
    datas=[
        ('runs/detect/sperm_detection/weights/best.pt', 'runs/detect/sperm_detection/weights'),
    ],
    hiddenimports=[
        'ultralytics',
        'ultralytics.nn.modules',
        'ultralytics.nn.modules.head',
        'ultralytics.nn.modules.block',
        'ultralytics.engine',
        'ultralytics.data',
        'ultralytics.utils',
        'torch',
        'torchvision',
        'cv2',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SpermCounter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SpermCounter',
)

app = BUNDLE(
    coll,
    name='SpermCounter.app',
    icon=None,
    bundle_identifier=None,
    info_plist={
        'NSHighResolutionCapable': 'True',
        'CFBundleDisplayName': '精子染色状态统计工具',
        'CFBundleName': 'SpermCounter',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'CFBundleIdentifier': 'com.example.SpermCounter',
        'LSMinimumSystemVersion': '10.13',
    },
)
