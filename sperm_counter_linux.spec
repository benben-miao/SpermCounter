# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None
model_path = 'runs/detect/sperm_detection/weights/best.onnx'
logo_path = 'assets/logos/logo.png'

datas = [(model_path, '.')]
if os.path.exists('README.md'):
    datas.append(('README.md', '.'))
if os.path.exists('assets/images'):
    datas.append(('assets/images', 'assets/images'))
for chart_name in [
    'BoxPR_curve.png',
    'BoxP_curve.png',
    'BoxR_curve.png',
    'BoxF1_curve.png',
    'confusion_matrix.png',
    'results.png',
]:
    chart_path = os.path.join('runs/detect/sperm_detection', chart_name)
    if os.path.exists(chart_path):
        datas.append((chart_path, 'runs/detect/sperm_detection'))
if os.path.exists(logo_path):
    datas.append((logo_path, 'assets/logos'))

a = Analysis(
    ['06.pyside_onnx_gui.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
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
        'PySide6.Qt3DAnimation', 'PySide6.Qt3DCore', 'PySide6.Qt3DExtras',
        'PySide6.Qt3DInput', 'PySide6.Qt3DLogic', 'PySide6.Qt3DRender',
        'PySide6.QtBluetooth', 'PySide6.QtCharts', 'PySide6.QtConcurrent',
        'PySide6.QtDataVisualization', 'PySide6.QtDesigner', 'PySide6.QtHelp',
        'PySide6.QtMultimedia', 'PySide6.QtMultimediaWidgets',
        'PySide6.QtNetwork', 'PySide6.QtNfc', 'PySide6.QtOpenGL',
        'PySide6.QtOpenGLWidgets', 'PySide6.QtPdf', 'PySide6.QtPdfWidgets',
        'PySide6.QtPositioning', 'PySide6.QtPrintSupport', 'PySide6.QtQml',
        'PySide6.QtQuick', 'PySide6.QtQuick3D', 'PySide6.QtQuickControls2',
        'PySide6.QtQuickWidgets', 'PySide6.QtRemoteObjects',
        'PySide6.QtScxml', 'PySide6.QtSensors', 'PySide6.QtSerialPort',
        'PySide6.QtSpatialAudio', 'PySide6.QtSql', 'PySide6.QtStateMachine',
        'PySide6.QtSvg', 'PySide6.QtSvgWidgets', 'PySide6.QtTest',
        'PySide6.QtTextToSpeech', 'PySide6.QtUiTools', 'PySide6.QtWebChannel',
        'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineQuick',
        'PySide6.QtWebEngineWidgets', 'PySide6.QtWebSockets',
        'PySide6.QtXml',
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
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
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
    strip=True,
    upx=True,
    upx_exclude=[],
    name='SpermCounter',
)
