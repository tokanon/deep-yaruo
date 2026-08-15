from pathlib import Path


root = Path(SPECPATH)
datas = [
    (str(root / "frontend" / "dist"), "frontend/dist"),
    (str(root / "assets" / "fonts" / "Saitamaar.ttf"), "assets/fonts"),
    (str(root / "models" / "deepaa-light.onnx"), "models"),
    (str(root / "models" / "deepaa-start-locator.onnx"), "models"),
    (str(root / "models" / "deepaa-charset.csv"), "models"),
    (str(root / "models" / "deepaa-surface-v0-ls.onnx"), "models"),
    (str(root / "models" / "deepaa-surface-v0-vocabulary.json"), "models"),
    (str(root / "backend" / "surface_texture_catalog.json"), "backend"),
    (str(root / "datasets" / "bootstrap" / "LICENSE"), "licenses/DeepAA"),
    (str(root / "LICENSE"), "."),
    (str(root / "README.md"), "."),
    (str(root / "THIRD_PARTY.md"), "."),
]

a = Analysis(
    [str(root / "backend" / "desktop.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "backend.app",
        "backend.contracts",
        "backend.deepaa",
        "backend.image_io",
        "backend.rendering",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["onnx", "pandas", "tensorflow", "torch"],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="YaruoAAStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="YaruoAAStudio",
)
