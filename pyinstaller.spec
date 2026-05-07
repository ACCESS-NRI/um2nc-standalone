# -*- mode: python ; coding: utf-8 -*-

from shutil import which
from importlib.metadata import distribution
from importlib.util import find_spec

# Add dist-info for common packages that need it
packages=(
    'um2nc',
    'numpy',
    'scipy',
    'dask'
)
datas = [
    (str(distribution(package)._path), str(distribution(package)._path.name)) 
    for package in packages
    if find_spec(package) is not None
]

a = Analysis(
    [which('um2nc')],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='um2nc',
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