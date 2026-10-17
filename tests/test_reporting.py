"""Reporting tests: target tables, text report and run manifest."""
import json

import numpy as np

from geoai.utils.reporting import combine_targets, save_target_report, write_manifest


def _targets(typ, n):
    return [{'rank': i + 1, 'x': 1000.0 * i, 'y': 2000.0, 'peak_x': 1.0, 'peak_y': 2.0,
             'max_probability': 0.9 - 0.1 * i, 'mean_probability': 0.5,
             'max_score': 0.85 - 0.1 * i, 'mean_uncertainty': 0.1, 'area_km2': 0.5,
             'n_pixels': 5, 'priority': 1, 'target_type': typ, 'threshold_used': 0.4}
            for i in range(n)]


def test_combine_targets_keeps_per_type_rank():
    df = combine_targets({'mineral': _targets('mineral', 2), 'groundwater': _targets('groundwater', 3)})
    assert len(df) == 5
    assert sorted(df.loc[df.target_type == 'groundwater', 'rank']) == [1, 2, 3]
    assert sorted(df['global_rank']) == [1, 2, 3, 4, 5]


def test_save_target_report_files(tmp_path):
    paths = save_target_report({'mineral': _targets('mineral', 2), 'geothermal': []},
                               str(tmp_path), project_name='P', cv_results={
                                   'mineral': {'rf': {'auc_roc': 0.9, 'auc_pr': 0.8}}},
                               timestamp='T', modes={'geothermal': 'unsupervised'},
                               return_all=True)
    csv, txt = paths
    assert csv.endswith('P_targets_T.csv') and txt.endswith('P_report_T.txt')
    text = open(txt, encoding='utf-8').read()
    assert '[GEOTHERMAL]  (unsupervised' in text and 'No targets found' in text
    assert 'auc_roc=0.9000' in text
    assert save_target_report({'x': []}, str(tmp_path), timestamp='U') is None


def test_write_manifest_sanitizes(tmp_path):
    p = write_manifest(str(tmp_path / 'm.json'),
                       {'a': np.float32(np.nan), 'b': np.int64(3), 'c': [1.0, float('inf')],
                        'd': np.arange(3), 'e': (1, 2)})
    data = json.load(open(p, encoding='utf-8'))
    assert data == {'a': None, 'b': 3, 'c': [1.0, None], 'd': [0, 1, 2], 'e': [1, 2]}
