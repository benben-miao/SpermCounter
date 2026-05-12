# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

icon_path = 'assets/logos/logo.ico'
if not os.path.exists(icon_path):
    icon_path = None

a = Analysis(
    ['06.pyside_onnx_gui.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('runs/detect/sperm_detection/weights/best.onnx', '.'),
        ('assets', 'assets'),
    ],
    hiddenimports=[
        'onnxruntime',
        'onnxruntime.capi',
        'onnxruntime.capi._pybind_state',
        'cv2',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'scipy', 'pandas', 'tkinter', 'PIL', 'IPython',
        'torch', 'torchvision', 'ultralytics',
        'sklearn', 'seaborn', 'plotly', 'networkx', 'sympy',
        'pygame', 'wx', 'xmlrpc', 'smtplib', 'email', 'urllib3',
        'requests', 'beautifulsoup4', 'sqlalchemy', 'psycopg2',
        'mysql', 'boto3', 'google', 'aws', 'azure', 'tensorflow',
        'keras', 'jax', 'mxnet', 'openvino', 'paddlepaddle',
        'setuptools', 'pip', 'wheel', 'pip._vendor', 'pkg_resources',
        'unittest', 'doctest', 'pdb', 'curses', 'readline',
        'rlcompleter', 'charset_normalizer.md__mypyc',
    ],
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
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SpermCounter',
)
