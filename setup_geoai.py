#!/usr/bin/env python3
"""
GeoAI Setup Script
==================
Run this script to install the required packages and verify the installation.

Usage:
    python setup_geoai.py
"""

import sys
import subprocess
import importlib
import os

# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED PACKAGES
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED = [
    # Import name        pip name              Required?
    ("numpy",            "numpy",               True),
    ("pandas",           "pandas",              True),
    ("scipy",            "scipy",               True),
    ("sklearn",          "scikit-learn",        True),
    ("matplotlib",       "matplotlib",          True),
    ("joblib",           "joblib",              True),
    ("rasterio",         "rasterio",            False),   # for GeoTIFF
    ("PIL",              "Pillow",              False),   # fallback when rasterio is missing
]

# ─────────────────────────────────────────────────────────────────────────────

def check_python_version():
    major, minor = sys.version_info[:2]
    print(f"Python version: {major}.{minor}")
    if major < 3 or (major == 3 and minor < 9):
        print("ERROR: Python 3.9 or newer is required.")
        print(f"Current: Python {major}.{minor}")
        print("Please upgrade from https://www.python.org/downloads/")
        sys.exit(1)
    print("  ✓ Python version is compatible")


def install_package(pip_name: str) -> bool:
    """Install a package with pip. Returns whether the installation succeeded."""
    print(f"  Installing: {pip_name} ...", end=" ", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_name, "--quiet"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("✓")
        return True
    else:
        print("✗ FAILED")
        if result.stderr:
            # Show only the first line
            first_err = result.stderr.strip().split('\n')[0]
            print(f"    Error: {first_err}")
        return False


def check_and_install():
    print("\n" + "="*55)
    print("  GeoAI Package Installation")
    print("="*55)

    missing_required   = []
    missing_optional   = []

    for import_name, pip_name, required in REQUIRED:
        try:
            importlib.import_module(import_name)
            print(f"  ✓ {import_name} already installed")
        except ImportError:
            if required:
                missing_required.append((import_name, pip_name))
            else:
                missing_optional.append((import_name, pip_name))

    if missing_required:
        print(f"\n  Installing required packages ({len(missing_required)}):")
        for _, pip_name in missing_required:
            ok = install_package(pip_name)
            if not ok:
                print(f"\n  CRITICAL ERROR: {pip_name} could not be installed.")
                print("  Try installing it manually:")
                print(f"    pip install {pip_name}")
                print("  or, if you use conda:")
                print(f"    conda install {pip_name}")
                sys.exit(1)

    if missing_optional:
        print(f"\n  Installing optional packages ({len(missing_optional)}):")
        for import_name, pip_name in missing_optional:
            ok = install_package(pip_name)
            if not ok:
                if import_name == "rasterio":
                    print("    → No rasterio GeoTIFF support (the Pillow fallback will be used)")
                    print("    → If you need full GeoTIFF support: conda install rasterio")


def verify_installation():
    print("\n" + "="*55)
    print("  Installation Verification")
    print("="*55)

    all_ok = True
    for import_name, pip_name, required in REQUIRED:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            status = "✓" if required else "✓ (opt)"
            print(f"  {status} {import_name:<15} {version}")
        except ImportError:
            if required:
                print(f"  ✗ {import_name:<15} MISSING (required)")
                all_ok = False
            else:
                print(f"  - {import_name:<15} not installed (optional)")

    return all_ok


def verify_geoai():
    """Check that the geoai package is importable."""
    print("\n" + "="*55)
    print("  GeoAI Package Check")
    print("="*55)

    # Add the script's directory to the Python path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        from geoai.pipeline import GeoAIPipeline  # noqa: F401
        from geoai.io.loaders import load_grid  # noqa: F401
        from geoai.models.prospectivity import GeoProspectivityModel, ImbalanceAnalyzer  # noqa: F401
        print("  ✓ geoai.pipeline")
        print("  ✓ geoai.io.loaders")
        print("  ✓ geoai.models.prospectivity")
        return True
    except ImportError as e:
        print(f"  ✗ GeoAI import error: {e}")
        print("  → Make sure this script is in the same directory as the geoai/ folder.")
        print("  → Expected layout:")
        print("      project/")
        print("      ├── geoai/          ← package folder")
        print("      │   ├── pipeline.py")
        print("      │   └── ...")
        print("      └── setup_geoai.py  ← this script")
        return False


def run_quick_test():
    """Simple functional smoke test."""
    print("\n" + "="*55)
    print("  Quick Functional Test")
    print("="*55)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        import numpy as np
        from geoai.models.prospectivity import ImbalanceAnalyzer, geo_smote

        # Class-imbalance analysis test
        y = np.array([1]*5 + [0]*95)
        imb = ImbalanceAnalyzer(y)
        assert imb.severity == 'severe', "Incorrect severity detection"
        assert abs(imb.ratio - 19.0) < 0.01, "Incorrect ratio"
        print("  ✓ ImbalanceAnalyzer works (19:1 → SEVERE)")

        # SMOTE test
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        X[:5] += 3.0
        X_aug, y_aug = geo_smote(X, y, target_ratio=1/3, k_neighbors=4)
        assert y_aug.sum() > 5, "SMOTE is not working"
        print(f"  ✓ SMOTE works ({y_aug.sum()} positives → {len(y_aug)} total)")

        # Co-registration test
        from geoai.core.preprocessor import CoRegistrar
        import numpy as np
        layers = {
            'mag': {'grid': np.random.randn(50, 50), 'x': np.linspace(0,1000,50), 'y': np.linspace(0,1000,50)},
            'grav':{'grid': np.random.randn(80, 80), 'x': np.linspace(100,900,80),'y': np.linspace(100,900,80)},
        }
        cr = CoRegistrar()
        reg = cr.fit_transform(layers, target_nx=40, target_ny=40)
        assert len(reg) == 2
        print(f"  ✓ Co-registration works ({len(reg)} layers aligned)")

        print("\n  All tests passed ✓")
        return True

    except Exception as e:
        print(f"  ✗ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_usage():
    print("\n" + "="*55)
    print("  USAGE GUIDE")
    print("="*55)
    print("""
  1. DEMO (with synthetic data):
  ─────────────────────────────
  python demo.py

  2. With REAL DATA:
  ─────────────────────────────
  python run_project.py

  3. With a JUPYTER NOTEBOOK:
  ─────────────────────────────
  jupyter notebook geoai_notebook.ipynb

  4. COMMAND LINE (JSON configuration):
  ─────────────────────────────
  python -m geoai template project.json   # example configuration
  python -m geoai run project.json --save-model model.joblib
  python -m geoai apply model.joblib block_b.json   # trained model → new area
  python -m geoai info data/magnetic.grd

  5. REALISTIC EXAMPLES (synthetic survey deliverables):
  ─────────────────────────────
  python examples/01_porphyry_survey_from_files.py
  python generate_test_data.py --scenario groundwater --survey --output-dir survey_gw

  Supported file formats:
    .grd   → Surfer ASCII / Binary 6 / Binary 7, Geosoft (Oasis Montaj),
             GMT netCDF (native netCDF-3 reader)
    .nc    → netCDF-3 (COARDS / CF / legacy GMT)
    .tif   → GeoTIFF (rasterio; otherwise Pillow + GeoTIFF tags)
    .ers   → ER Mapper
    .asc   → ESRI ASCII Grid
    .xyz / .csv / .dat / .txt → regular lattice, scattered points or
             Geosoft XYZ flight lines (gridded by interpolation)

  Example well CSV:
    X,Y,LABEL,TARGET_TYPE
    312000,4012000,1,mineral    ← positive (mineralization/water found)
    320000,4018000,0,mineral    ← negative (dry/barren well)
  LABEL may also be yes/no or true/false.
""")


if __name__ == "__main__":
    print("\n  ╔══════════════════════════════════╗")
    print("  ║  GeoAI v4.0 — Setup Script       ║")
    print("  ╚══════════════════════════════════╝\n")

    check_python_version()
    check_and_install()
    deps_ok  = verify_installation()
    geoai_ok = verify_geoai()

    if deps_ok and geoai_ok:
        test_ok = run_quick_test()
    else:
        test_ok = False

    print_usage()

    print("\n" + "="*55)
    if deps_ok and geoai_ok and test_ok:
        print("  ✓ Installation SUCCESSFUL — GeoAI is ready to use!")
    else:
        print("  ✗ Installation has problems — review the errors above.")
    print("="*55 + "\n")
