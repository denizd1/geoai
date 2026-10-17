"""
GeoAI — realistic synthetic data generator (command-line wrapper).

The generator lives in the package as geoai.synthetic; this script keeps the
old entry point and import path working:

    python generate_test_data.py --scenario porphyry_copper --size 200
    python generate_test_data.py --scenario groundwater --survey --output-dir survey_gw

    import generate_test_data as gtd          # same objects as geoai.synthetic
"""
from geoai.synthetic import *  # noqa: F401,F403
from geoai.synthetic import _validate_targets, main  # noqa: F401

if __name__ == '__main__':
    raise SystemExit(main())
