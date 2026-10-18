"""
Example 01 — Porphyry copper district from raw survey deliverables
===================================================================

A realistic exploration workflow, end to end:

1. A synthetic porphyry Cu-Au district (one main system and two satellite
   intrusive centres, ~15 × 15 km) is turned into the files a project
   actually receives from contractors and the lab:

     mag_survey.xyz         airborne magnetics, Geosoft XYZ flight lines
                            (E-W lines every 300 m, N-S tie lines)
     gravity_stations.csv   irregular ground gravity stations (Bouguer, mGal)
     ert_profiles.csv       ERT apparent-resistivity profiles (Ω·m)
     ip_chargeability.grd   contractor IP chargeability grid (Surfer 7 binary)
     soil_geochem.csv       soil assays, Cu and Au in ppm, with detection-limit
                            flags (values below DL reported as DL/2)
     drillholes.csv         drill collars, RESULT = yes / no
     truth_zones.csv        the hidden truth (only used for validation)

2. GeoAI grids the point and line data, co-registers all layers on a 100 m
   grid, builds geophysical features (including a reduced-to-pole magnetic
   feature from the local field I = 55°, D = 5°), trains the ANN + RF + GB
   ensemble with spatial cross-validation by well clusters, and ranks drill
   targets with probability, uncertainty and priority classes.

3. The ranked targets are checked against the hidden truth zones, and the
   trained pipeline is saved for example 05 (application to a new block).

Run:   python examples/01_porphyry_survey_from_files.py [--size 150] [--seed 7]
Output: examples/output/01_porphyry/ (CSV, TXT, GeoJSON, HTML report, PNG maps,
        Surfer grids, manifest, model.joblib)
"""
import argparse
import time

from _common import (data_dir, load_survey, output_dir, print_validation, quiet_warnings,
                     truth_zones, validate)

from geoai.synthetic import RealisticGeoDataGenerator, export_survey_dataset


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--size', type=int, default=150, help='grid cells per side (100 m cells)')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--wells', type=int, default=60, help='number of drill holes')
    args = ap.parse_args(argv)
    quiet_warnings()
    t0 = time.time()

    # ── 1. Survey deliverables ───────────────────────────────────────────
    survey = data_dir('porphyry_block_a')
    gen = RealisticGeoDataGenerator(nx=args.size, ny=args.size, cell_size=100.0,
                                    scenario='porphyry_copper', seed=args.seed)
    gen.generate()
    export_survey_dataset(gen, survey, n_wells=args.wells, imbalance_ratio=8.0)

    # ── 2. Load the files exactly as delivered ───────────────────────────
    pipe = load_survey(survey, project_name='Porphyry_Block_A')

    out = output_dir('01_porphyry')
    results = pipe.run(
        target_types='mineral',
        target_nx=args.size, target_ny=args.size,   # 100 m reference grid
        n_targets=12,
        min_prob=None,                              # threshold from cross-validation
        cv_folds=5,
        magnetic_field={'inclination': 55.0, 'declination': 5.0},
        sampler_kwargs={'positive_radius': 250.0, 'negative_radius': 150.0},
        grid_formats=('grd', 'geosoft'),
        output_dir=str(out),
        show_plots=False,
    )

    # ── 3. Validation and model export ───────────────────────────────────
    res = results['mineral']
    print("\nCross-validation (spatial, by well clusters):")
    for model, m in res['cv_results'].items():
        print(f"  {model:9}: ROC-AUC {m['auc_roc']:.3f} | PR-AUC {m['auc_pr']:.3f} | "
              f"balanced Brier {m['brier']:.3f}")
    print(f"  CV groups with positives: {res['cv_info'].get('n_pos_groups')} | "
          f"decision threshold {res['optimal_threshold']:.2f}")
    print("\nTop features (Random Forest):")
    fi = pipe.prospector.models['mineral'].feature_importances
    for name, val in fi.head(8).items():
        print(f"  {name:36} {val:.3f}")

    print_validation(validate(res['targets'], truth_zones(survey), top_k=10))
    model_path = pipe.save(out / 'model.joblib')
    print(f"\nTrained pipeline saved to {model_path} (used by example 05).")
    print(f"Open the HTML report in {out}. Elapsed: {time.time() - t0:.0f} s")
    return pipe, results


if __name__ == '__main__':
    main()
