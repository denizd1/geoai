"""
demo.py — GeoAI Synthetic Data Demo
===================================
Run this to test the system without real geophysical data.

Usage:
    python demo.py
"""

import sys
import os

# Locate the geoai package (this script must sit in the same directory as the geoai/ folder)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geoai.pipeline import GeoAIPipeline

if __name__ == "__main__":

    print("\n  GeoAI Demo — running on synthetic geophysical data")
    print("  Outputs will be saved to the 'output_demo/' directory\n")

    results = GeoAIPipeline.run_demo(
        nx=120,           # Grid size (you can increase it: 200, 300...)
        ny=120,
        target_types=[
            'mineral',        # Mineral/sulphide exploration
            'groundwater',    # Groundwater
            # 'geothermal',   # Geothermal (remove the # to enable)
        ],
        output_dir='output_demo',
        show_plots=True,   # Set to False if you do not want plot windows to open
        save_plots=True,   # Save as PNG
    )

    # Summary
    print("\nTOP TARGETS:")
    for target_type, res in results.items():
        targets = res['targets']
        if targets:
            t = targets[0]
            pri = t['priority'] if t.get('priority') is not None else '-'
            print(
                f"  {target_type:12} [{res['mode']}]: "
                f"X={t['x']:.0f}  Y={t['y']:.0f}  "
                f"P={t['max_probability']:.3f}  "
                f"Area={t['area_km2']:.2f} km²  "
                f"Priority={pri}"
            )
    print("\nOpen the HTML report in output_demo/ for the full summary "
          "(targets, CV metrics and figures).")
