"""Command-line interface tests."""
import glob
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd

from geoai.cli import TEMPLATE, main
from geoai.io.writers import save_surfer_ascii_grd
from geoai.pipeline import GeoAIPipeline

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_template_written_and_not_overwritten(tmp_path, capsys):
    p = tmp_path / 'proj.json'
    assert main(['template', str(p)]) == 0
    assert json.loads(p.read_text(encoding='utf-8')) == TEMPLATE
    assert main(['template', str(p)]) == 1
    assert main(['template', str(p), '--force']) == 0


def test_info_command(tmp_path, capsys):
    g = np.arange(12, dtype=float).reshape(3, 4)
    p = save_surfer_ascii_grd(g, np.arange(4) * 10.0, np.arange(3) * 20.0, tmp_path / 'mag.grd')
    assert main(['info', p]) == 0
    out = capsys.readouterr().out
    assert 'surfer_ascii' in out and '3 rows × 4 columns' in out and 'magnetic' in out
    assert main(['info', str(tmp_path / 'missing.grd')]) == 1


def test_run_with_config(tmp_path, capsys):
    demo = GeoAIPipeline.make_demo_data(nx=40, ny=40)
    data = tmp_path / 'data'
    for name in ('magnetic_tmi', 'resistivity', 'ip_chargeability'):
        save_surfer_ascii_grd(demo['layers'][name], demo['x'], demo['y'], data / f'{name}.grd')
    w = demo['wells']['mineral'].rename(columns={'label': 'result'})
    w.to_csv(data / 'wells.csv', index=False)
    cfg = {
        'project_name': 'CLI_Test',
        'output_dir': 'output',
        'layers': [{'path': f'data/{n}.grd'} for n in ('magnetic_tmi', 'resistivity',
                                                         'ip_chargeability')],
        'wells': [{'path': 'data/wells.csv', 'target_type': 'mineral', 'label_col': 'result'}],
        'run': {'target_types': ['mineral'], 'target_nx': 40, 'target_ny': 40, 'cv_folds': 3,
                'save_plots': False, 'grid_formats': ['grd'], 'verbose': False},
    }
    cfg_path = tmp_path / 'proj.json'
    cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert main(['run', str(cfg_path)]) == 0
    out = capsys.readouterr().out
    assert 'TARGET LIST' in out and '[MINERAL]' in out
    out_dir = tmp_path / 'output'                     # relative to the configuration file
    assert glob.glob(str(out_dir / 'CLI_Test_targets_*.csv'))
    assert os.path.exists(out_dir / 'grids' / 'mineral_probability.grd')
    df = pd.read_csv(glob.glob(str(out_dir / 'CLI_Test_targets_*.csv'))[0])
    assert (df['target_type'] == 'mineral').all()


def test_run_with_invalid_config(tmp_path, capsys):
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'layers': [{'path': 'a.grd', 'colour': 1}]}), encoding='utf-8')
    assert main(['run', str(bad)]) == 1
    assert 'unknown key' in capsys.readouterr().err
    bad.write_text(json.dumps({'layers': []}), encoding='utf-8')
    assert main(['run', str(bad)]) == 1
    bad.write_text('{not json', encoding='utf-8')
    assert main(['run', str(bad)]) == 1
    bad.write_text(json.dumps({'layers': [{'path': 'missing.grd'}]}), encoding='utf-8')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert main(['run', str(bad)]) == 1          # no layer could be loaded


def test_module_entry_point_version():
    r = subprocess.run([sys.executable, '-m', 'geoai', '--version'], cwd=REPO,
                       capture_output=True, text=True, check=True)
    import geoai
    assert r.stdout.strip() == f'geoai {geoai.__version__}'
