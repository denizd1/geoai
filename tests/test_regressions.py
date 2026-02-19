import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from geoai.core.preprocessor import CoRegistrar
from geoai.io.loaders import _find_column, load_well_data, read_geotiff
from geoai.pipeline import GeoAIPipeline
from geoai.viz.maps import plot_target_report


class GeoAIRegressionTests(unittest.TestCase):

    def test_cv_metrics_plot_uses_current_metric_keys(self):
        cv = {
            'ann': {'auc_roc': 0.91, 'auc_pr': 0.87},
            'rf': {'auc_roc': 0.93, 'auc_pr': 0.90},
        }
        fig = plot_target_report(
            targets=[],
            target_type='test',
            cv_results=cv,
            feature_importances=None,
            save_path=None,
            show=False,
        )
        ax = fig.axes[-1]
        heights = [float(p.get_height()) for p in ax.patches[:4]]
        self.assertGreater(heights[0], 0.0)
        self.assertGreater(heights[1], 0.0)
        self.assertGreater(heights[2], 0.0)
        self.assertGreater(heights[3], 0.0)

    def test_find_column_numeric_fallback_index(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        self.assertEqual(_find_column(df, None, ['x'], fallback_index=0), 'a')
        self.assertEqual(_find_column(df, None, ['y'], fallback_index=1), 'b')
        self.assertEqual(_find_column(df, None, ['z'], fallback_index=2), 'c')

    def test_load_well_data_rejects_invalid_label_values(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'wells_invalid.csv'
            p.write_text("X,Y,LABEL\n1,1,2\n2,2,1\n", encoding='utf-8')
            with self.assertRaises(ValueError):
                load_well_data(str(p))

    def test_transform_points_returns_in_bounds_mask(self):
        layers = {
            'g': {
                'x': np.linspace(0, 10, 11),
                'y': np.linspace(0, 10, 11),
                'grid': np.zeros((11, 11), dtype=float),
            }
        }
        cr = CoRegistrar().fit(layers, target_nx=11, target_ny=11)
        rows, cols, mask = cr.transform_points(
            np.array([-1, 5, 12], dtype=float),
            np.array([5, 5, 5], dtype=float),
            return_mask=True,
        )
        self.assertEqual(rows.tolist(), [5, 5, 5])
        self.assertEqual(cols.tolist(), [0, 5, 10])
        self.assertEqual(mask.tolist(), [False, True, False])

    def test_unsupervised_fallback_returns_requested_target_type(self):
        x = np.linspace(0, 100, 30)
        y = np.linspace(0, 100, 30)
        grid = np.random.RandomState(42).randn(30, 30)

        pipe = GeoAIPipeline(project_name='TestUnsup')
        pipe.add_layer_array(grid, x, y, 'magnetic_tmi')

        with tempfile.TemporaryDirectory() as td:
            results = pipe.run(
                target_types=['mineral'],
                target_nx=30,
                target_ny=30,
                n_targets=3,
                cv_folds=2,
                output_dir=td,
                show_plots=False,
                save_plots=False,
                verbose=False,
            )

        self.assertIn('mineral', results)
        self.assertEqual(results['mineral'].get('mode'), 'unsupervised')
        self.assertIn('targets', results['mineral'])

    def test_geotiff_reader_returns_pixel_center_coordinates(self):
        try:
            import rasterio
            from rasterio.transform import from_origin
        except Exception:
            self.skipTest("rasterio yüklü değil")

        with tempfile.TemporaryDirectory() as td:
            tif_path = Path(td) / 'mini.tif'
            data = np.array([[1, 2], [3, 4]], dtype=np.float32)
            transform = from_origin(100.0, 200.0, 1.0, 1.0)
            with rasterio.open(
                tif_path,
                'w',
                driver='GTiff',
                height=2,
                width=2,
                count=1,
                dtype='float32',
                transform=transform,
            ) as dst:
                dst.write(data, 1)

            out = read_geotiff(str(tif_path))
            self.assertAlmostEqual(float(out['x'][0]), 100.5, places=6)
            self.assertAlmostEqual(float(out['y'][0]), 198.5, places=6)
            self.assertTrue(out['x'][0] < out['x'][-1])
            self.assertTrue(out['y'][0] < out['y'][-1])


if __name__ == '__main__':
    unittest.main()
