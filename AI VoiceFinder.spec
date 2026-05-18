# -*- mode: python ; coding: utf-8 -*-

import sys
import os

block_cipher = None

# Находим путь к tcl/tk
python_dir = os.path.dirname(sys.executable)
tcl_dir = os.path.join(python_dir, 'tcl')

# Подготавливаем список данных
datas_list = [('Sources', 'Sources')]

# Добавляем tcl/tk если существует
if os.path.exists(tcl_dir):
    datas_list.append((tcl_dir, 'tcl'))
    print(f"✓ Включен tcl/tk из: {tcl_dir}")
else:
    print(f"⚠ Предупреждение: tcl/tk не найден в {tcl_dir}")
    print("  tkinter может не работать в EXE!")

a = Analysis(
    ['AI VoiceFinder.py'],
    pathex=[],
    binaries=[],
    datas=datas_list,
    hiddenimports=[
        # tkinter (критично!)
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        'tkinter.constants',
        '_tkinter',
        # PIL для изображений
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'PIL._tkinter_finder',
        # WhisperX
        'whisperx',
        'torch',
        'torchaudio',
        'soundfile',
        # NeMo
        'nemo',
        'nemo.collections.asr',
        'omegaconf',
        # Pyannote
        'pyannote.core',
        'pyannote.audio',
        # Системные
        'threading',
        'queue',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'numpy.testing',
        'pytest',
        'IPython',
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AI VoiceFinder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Сжатие UPX
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Показываем консоль для отладки
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Можно добавить иконку
)
