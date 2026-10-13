# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the RLBot v5 bot pack (built by bob). PyTorch is not bundled: the pack ships one shared copy
# (torch-archive) that src/bot.py falls back to. The hidden imports are standard-library modules PyTorch loads.


a = Analysis(
    ['src/bot.py'],
    pathex=['src'],
    binaries=[],
    datas=[('src/model.pt', '.'), ('src/whiskers/kickoffs.json', 'whiskers')],
    hiddenimports=['timeit', 'pickletools', 'uuid', 'unittest.mock', 'ctypes.wintypes'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='bot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
