"""
geoai.utils.reporting
======================
Hedef listesi ve model metriklerini CSV / text raporu olarak kaydeder.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def targets_to_dataframe(targets: list) -> pd.DataFrame:
    """Hedef listesini DataFrame'e dönüştürür."""
    if not targets:
        return pd.DataFrame()
    df = pd.DataFrame(targets)
    col_order = [
        'rank', 'target_type', 'x', 'y',
        'max_probability', 'mean_probability', 'max_score',
        'mean_uncertainty', 'area_km2', 'n_pixels',
    ]
    present = [c for c in col_order if c in df.columns]
    return df[present].sort_values('rank').reset_index(drop=True)


def save_target_report(
    all_targets: dict,
    output_dir: str,
    project_name: str = 'GeoAI',
    cv_results: dict = None,
):
    """
    Tüm hedef tiplerini tek bir Excel/CSV raporuna kaydeder.

    Parametreler
    ------------
    all_targets  : {target_type: list_of_targets}
    output_dir   : Çıkış klasörü
    project_name : Rapor başlığı
    cv_results   : {target_type: {model: metrics}}
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    all_dfs = []
    for typ, targets in all_targets.items():
        if not targets:
            continue
        df = targets_to_dataframe(targets)
        if 'target_type' not in df.columns:
            df.insert(1, 'target_type', typ)
        all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined['rank'] = range(1, len(combined) + 1)

        csv_path = Path(output_dir) / f'{project_name}_targets_{timestamp}.csv'
        combined.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"  ✓ Hedef CSV: {csv_path}")

    # Text raporu
    txt_path = Path(output_dir) / f'{project_name}_report_{timestamp}.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"  {project_name} — JEOFİZİK HEDEF RAPORU\n")
        f.write(f"  Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"{'='*60}\n\n")

        for typ, targets in all_targets.items():
            f.write(f"\n[{typ.upper()}]\n{'─'*40}\n")
            if not targets:
                f.write("  Hedef bulunamadı.\n")
                continue

            f.write(f"{'Sıra':<5} {'X':>12} {'Y':>12} {'P_maks':>8} {'Skor':>8} {'Alan_km2':>10}\n")
            f.write(f"{'─'*55}\n")
            for t in targets:
                f.write(
                    f"  {t['rank']:<4} "
                    f"{t['x']:>12.1f} "
                    f"{t['y']:>12.1f} "
                    f"{t['max_probability']:>8.3f} "
                    f"{t.get('max_score', t['max_probability']):>8.3f} "
                    f"{t['area_km2']:>10.3f}\n"
                )

        if cv_results:
            f.write(f"\n\n{'='*60}\n  MODEL PERFORMANSI\n{'='*60}\n")
            for typ, models in cv_results.items():
                f.write(f"\n[{typ.upper()}]\n")
                for model, metrics in models.items():
                    f.write(f"  {model:12}: " +
                            " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()) +
                            "\n")

    print(f"  ✓ Metin raporu: {txt_path}")
    return str(csv_path) if all_dfs else None
