"""
run_project.py — Running GeoAI on Real Data
===========================================
Edit this file to run the system on your own geophysical data.

Usage:
    python run_project.py

Alternative: keep the same settings in a JSON file and run from the command line:
    python -m geoai template project.json   →   python -m geoai run project.json
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geoai.cli import print_targets  # noqa: E402
from geoai.pipeline import GeoAIPipeline  # noqa: E402

# ══════════════════════════════════════════════════════════════
#  SETTINGS — Edit this section
# ══════════════════════════════════════════════════════════════

PROJECT_NAME = "My_Project"
OUTPUT_DIR = "output/"

# If no real layers are defined, fall back to the synthetic demo
DEMO_IF_NO_LAYERS = True
DEMO_TARGET_TYPES = ["mineral", "groundwater"]
DEMO_NX = 120
DEMO_NY = 120
DEMO_SHOW_PLOTS = False
DEMO_SAVE_PLOTS = True

# Choose the target type(s) (one or more):
#   'mineral'     → Mineral/sulphide/gold
#   'groundwater' → Groundwater
#   'geothermal'  → Geothermal
#   'generic'     → Generic (all layers weighted equally)
TARGET_TYPES = ["mineral", "groundwater"]  # ← edit

# Paths to your geophysical files
# Supported formats: .grd (Surfer ASCII/6/7, Geosoft), .tif (GeoTIFF),
#                    .ers (ER Mapper), .asc (ESRI ASCII), .xyz/.csv/.dat
# The layer name determines the layer type (magnetic, gravity, resistivity, IP,
# geochemistry...): e.g. 'magnetic_tmi', 'mag_rtp', 'gravity_bouguer', 'resistivity',
# 'ip_chargeability', 'geochemistry_cu' or 'Cu_ppm'.
LAYERS = {
    # 'layer_name': 'path/to/file.grd',
    # Example:
    # 'magnetic':    'data/magnetic_tmi.grd',
    # 'gravity':     'data/bouguer.grd',
    # 'resistivity': 'data/resistivity.tif',
    # 'ip':          'data/ip_chargeability.xyz',
    # 'geochemistry_cu': 'data/Cu_ppm.csv',
}

# Column names for XYZ/CSV files (if needed)
# If the columns are named X,Y,Z you do not need to set this.
XYZ_COLUMNS = {
    # 'ip': {'x_col': 'Easting', 'y_col': 'Northing', 'z_col': 'IP_ms'},
    # 'geochemistry_cu': {'x_col': 'X', 'y_col': 'Y', 'z_col': 'Cu_ppm'},
}

# Well / drillhole data (CSV)
# Required columns: X (or Easting), Y (or Northing), LABEL (1=positive, 0=negative)
# If a single file has a TARGET_TYPE column, only the matching rows are used for
# each target type (you can pass the same file for several types).
WELL_FILES = {
    # 'mineral': 'data/wells_mineral.csv',
    # 'groundwater': 'data/wells_water.csv',
}

# Grid resolution — large area + high resolution = slow
# Typical values: 200x200 (fast), 500x500 (detailed), 1000x1000 (high res)
# None → the finest cell size among the layers is used
GRID_NX = 300
GRID_NY = 300

# Target extraction and output settings
MIN_PROBABILITY = None       # None = optimal threshold found by cross-validation
UNCERTAINTY_PENALTY = 0.3    # λ in the confidence score S = P − λ·U
EXPORT_GRIDS = True          # write the P/U/S maps as Surfer GRD + GeoTIFF
SAVE_MODELS = False          # save trained models as output/models/*.joblib
SHOW_PLOTS = True

# ══════════════════════════════════════════════════════════════
#  EXECUTION — Usually no changes are needed here
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not LAYERS:
        if not DEMO_IF_NO_LAYERS:
            print("ERROR: the LAYERS dictionary is empty!")
            print("Open run_project.py and enter the paths to your geophysical files.")
            print(
                "Alternatively, set DEMO_IF_NO_LAYERS = True to run the synthetic demo."
            )
            sys.exit(1)

        print("WARNING: the LAYERS dictionary is empty. Running the synthetic DEMO...")
        results = GeoAIPipeline.run_demo(
            nx=DEMO_NX,
            ny=DEMO_NY,
            target_types=DEMO_TARGET_TYPES,
            output_dir=OUTPUT_DIR,
            show_plots=DEMO_SHOW_PLOTS,
            save_plots=DEMO_SAVE_PLOTS,
        )
        print_targets(results)
        sys.exit(0)

    if not WELL_FILES:
        print("WARNING: no well data defined.")
        print("The system runs in unsupervised (knowledge-driven) mode; targets will be less reliable.")

    # Create the pipeline
    pipe = GeoAIPipeline(project_name=PROJECT_NAME)

    # Load the layers
    print("\nLoading geophysical layers...")
    for name, path in LAYERS.items():
        kwargs = XYZ_COLUMNS.get(name, {})
        pipe.add_layer(path, name=name, **kwargs)

    # Load the well data
    if WELL_FILES:
        print("\nLoading well data...")
        for target_type, path in WELL_FILES.items():
            pipe.add_wells(path, target_type=target_type)

    if pipe.load_errors:
        print("\nWARNING: some files could not be loaded:")
        for _, error in pipe.load_errors:
            print(f"  - {error}")
    if not pipe.layers:
        print("ERROR: no layers could be loaded. Check the file paths.")
        sys.exit(1)

    # Run
    print("\nStarting the pipeline...\n")
    results = pipe.run(
        target_types=TARGET_TYPES,
        target_nx=GRID_NX,
        target_ny=GRID_NY,
        common_extent="intersection",  # Intersection of all layer extents
        n_targets=15,
        min_prob=MIN_PROBABILITY,
        cv_folds=5,
        output_dir=OUTPUT_DIR,
        show_plots=SHOW_PLOTS,
        save_plots=True,
        uncertainty_penalty=UNCERTAINTY_PENALTY,
        export_grids=EXPORT_GRIDS,
        save_models=SAVE_MODELS,
    )

    print_targets(results)
