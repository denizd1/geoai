"""
geoai.viz.maps
==============
Jeofizik veri ve model sonuçlarının görselleştirilmesi.
Tüm grafikler koyu (dark) tema ile bilimsel renk haritaları kullanır.

Not: Bu modül matplotlib backend'ini değiştirmez. Ekransız (headless) Linux
ortamlarında matplotlib otomatik olarak 'Agg' kullanır; etkileşimsiz bir
backend'de show=True verilirse figür sadece kapatılır.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from geoai.core.preprocessor import detect_layer_type


# ─────────────────────────────────────────────────────────────────────────────
# TEMA VE RENKLER
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG = '#0d1117'
PANEL_BG = '#161b22'
TEXT_COLOR = '#e6edf3'
MUTED = '#8b949e'
GRID_COLOR = '#21262d'
ACCENT = '#58a6ff'
NODATA_COLOR = '#3a3f47'

TARGET_COLORS = [
    '#ff4444', '#ff8c00', '#ffd700', '#00ff88',
    '#00cfff', '#bf5fff', '#ff69b4', '#aaffaa',
    '#ffaaaa', '#aaaaff',
]

# Prospectivity için özel renk haritası
_colors_prosp = [
    (0.05, 0.05, 0.15),   # Koyu lacivert: düşük
    (0.10, 0.20, 0.50),   # Mavi
    (0.00, 0.55, 0.65),   # Teal
    (0.10, 0.80, 0.30),   # Yeşil
    (0.90, 0.80, 0.00),   # Sarı
    (1.00, 0.40, 0.00),   # Turuncu
    (0.95, 0.10, 0.10),   # Kırmızı: yüksek
]
PROSP_CMAP = LinearSegmentedColormap.from_list(
    'prospectivity', _colors_prosp).with_extremes(bad=NODATA_COLOR)

# Jeofizik katman renk haritaları (katman tipine göre)
LAYER_CMAPS = {
    'magnetic':     'RdBu_r',
    'gravity':      'seismic',
    'resistivity':  'viridis_r',
    'conductivity': 'viridis',
    'ip':           'hot',
    'chargeability': 'hot',
    'geochemistry': 'YlOrRd',
    'radiometric':  'inferno',
    'seismic':      'Spectral_r',
    'topography':   'terrain',
    'geology':      'tab20',
    'default':      'RdBu_r',
}

_NON_INTERACTIVE = {'agg', 'pdf', 'ps', 'svg', 'pgf', 'cairo', 'template'}


def _get_cmap(layer_name: str):
    cmap = plt.get_cmap(LAYER_CMAPS.get(detect_layer_type(layer_name), LAYER_CMAPS['default']))
    return cmap.with_extremes(bad=NODATA_COLOR)


def _setup_dark_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.4, alpha=0.5)


def _km_label(val, _):
    return f"{val/1000:.1f}"


def _fmt_ax(ax, coregistrar=None):
    """Eksen etiketleri: projeksiyonlu (metre) koordinatlarda km, aksi halde ham birim."""
    use_km = True
    if coregistrar is not None and coregistrar.ref_x is not None:
        span = max(np.ptp(coregistrar.ref_x), np.ptp(coregistrar.ref_y))
        use_km = span >= 5000
    if use_km:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_km_label))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_km_label))
        ax.set_xlabel('X (km)', color=MUTED, fontsize=7)
        ax.set_ylabel('Y (km)', color=MUTED, fontsize=7)
    else:
        ax.set_xlabel('X', color=MUTED, fontsize=7)
        ax.set_ylabel('Y', color=MUTED, fontsize=7)
    ax.set_aspect('equal', adjustable='box')


def _colorbar(fig, mappable, ax, label=None):
    cb = fig.colorbar(mappable, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    if label:
        cb.set_label(label, color=MUTED, fontsize=8)
    return cb


def _robust_limits(grid):
    valid = grid[np.isfinite(grid)]
    if valid.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(valid, [2, 98])
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return float(vmin), float(vmax)


def _finalize(fig, save_path, show):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        print(f"  ✓ Kaydedildi: {save_path}")
    if show and matplotlib.get_backend().lower() not in _NON_INTERACTIVE:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _plot_targets(ax, targets, limit=10):
    for t in targets[:limit]:
        color = TARGET_COLORS[min(t['rank'] - 1, len(TARGET_COLORS) - 1)]
        ax.scatter(t['x'], t['y'], c=color, s=130, marker='*',
                   edgecolors='white', linewidths=0.7, zorder=10)
        ax.annotate(f"T{t['rank']}", (t['x'], t['y']), textcoords='offset points',
                    xytext=(5, 5), color=color, fontsize=8, fontweight='bold', zorder=11)


def _plot_wells(ax, well_df):
    if well_df is None or len(well_df) == 0:
        return False
    pos = well_df[well_df['label'] == 1]
    neg = well_df[well_df['label'] == 0]
    if len(pos):
        ax.scatter(pos['x'], pos['y'], c='lime', s=50, marker='^',
                   edgecolors='white', linewidths=0.7, zorder=8)
    if len(neg):
        ax.scatter(neg['x'], neg['y'], c='red', s=50, marker='v',
                   edgecolors='white', linewidths=0.7, zorder=8, alpha=0.7)
    return True


def _wells_legend(ax):
    handles = [
        Line2D([], [], marker='^', color='none', markerfacecolor='lime',
               markeredgecolor='white', markersize=7, label='Pozitif kuyu'),
        Line2D([], [], marker='v', color='none', markerfacecolor='red',
               markeredgecolor='white', markersize=7, label='Negatif kuyu'),
    ]
    ax.legend(handles=handles, fontsize=7, facecolor=PANEL_BG, labelcolor=MUTED,
              loc='lower left', framealpha=0.8)


# ─────────────────────────────────────────────────────────────────────────────
# GİRİŞ KATMANLARI
# ─────────────────────────────────────────────────────────────────────────────

def plot_input_layers(
    registered_layers: dict,
    coregistrar,
    save_path: str = None,
    show: bool = True,
    max_cols: int = 3,
    figsize_per: tuple = (5, 4),
    masks: dict = None,
):
    """
    Co-register edilmiş giriş katmanlarını çizer.

    registered_layers: {isim: 2D ndarray}
    masks            : {isim: bool ndarray} — True = gerçek veri yok (gri gösterilir)
    """
    names = list(registered_layers.keys())
    n = len(names)
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows + 0.6),
                     facecolor=DARK_BG, layout='constrained')
    fig.suptitle('JEOFİZİK GİRİŞ KATMANLARI — Co-registered',
                 color=TEXT_COLOR, fontsize=13, fontweight='bold')

    X, Y = np.meshgrid(coregistrar.ref_x, coregistrar.ref_y)

    for idx, name in enumerate(names):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        _setup_dark_ax(ax)

        grid = np.asarray(registered_layers[name], dtype=float)
        nodata = ~np.isfinite(grid)
        if masks is not None and name in masks:
            nodata = nodata | masks[name]
        shown = np.ma.masked_where(nodata, grid)
        vmin, vmax = _robust_limits(np.where(nodata, np.nan, grid))

        im = ax.pcolormesh(X, Y, shown, cmap=_get_cmap(name), shading='auto',
                           vmin=vmin, vmax=vmax)
        _colorbar(fig, im, ax)

        pct_nan = 100 * nodata.mean()
        suffix = f" [{pct_nan:.0f}% veri yok]" if pct_nan >= 1 else ""
        ax.set_title(f"{name.replace('_', ' ').upper()} ({detect_layer_type(name)})" + suffix,
                     color=TEXT_COLOR, fontsize=8, fontweight='bold', pad=4)
        _fmt_ax(ax, coregistrar)

    return _finalize(fig, save_path, show)


# ─────────────────────────────────────────────────────────────────────────────
# OLASILIK HARİTASI
# ─────────────────────────────────────────────────────────────────────────────

def plot_prospectivity(
    prob_map: np.ndarray,
    uncertainty_map: np.ndarray,
    targets: list,
    coregistrar,
    target_type: str = 'generic',
    well_df=None,
    reference_layer: np.ndarray = None,
    reference_name: str = 'Referans',
    save_path: str = None,
    show: bool = True,
    uncertainty_penalty: float = 0.3,
    mode: str = 'supervised',
):
    """
    Ana prospectivity sonuç haritası.
    4 panel: Referans | Olasılık | Belirsizlik | Güven-düzeltilmiş skor
    """
    X, Y = np.meshgrid(coregistrar.ref_x, coregistrar.ref_y)
    unsup = mode != 'supervised'
    p_label = 'Elverişlilik indeksi' if unsup else 'Olasılık'

    fig = plt.figure(figsize=(21, 5.6), facecolor=DARK_BG, layout='constrained')
    title = f'GeoAI Prospectivity — {target_type.upper()}'
    if unsup:
        title += '  (denetimsiz mod: kuyu verisi kullanılmadı)'
    fig.suptitle(title, color=TEXT_COLOR, fontsize=14, fontweight='bold')
    axes = fig.subplots(1, 4)

    # ── Panel 1: Referans katman ──────────────────────────────────────────
    ax1 = axes[0]
    _setup_dark_ax(ax1)
    if reference_layer is not None:
        vmin, vmax = _robust_limits(reference_layer)
        im1 = ax1.pcolormesh(X, Y, np.ma.masked_invalid(reference_layer),
                             cmap=_get_cmap(reference_name), shading='auto',
                             vmin=vmin, vmax=vmax)
        _colorbar(fig, im1, ax1)
    ax1.set_title(reference_name, color=TEXT_COLOR, fontsize=9, fontweight='bold')
    if _plot_wells(ax1, well_df):
        _wells_legend(ax1)
    _fmt_ax(ax1, coregistrar)

    # ── Panel 2: Olasılık Haritası ────────────────────────────────────────
    ax2 = axes[1]
    _setup_dark_ax(ax2)
    pm = np.ma.masked_invalid(prob_map)
    im2 = ax2.pcolormesh(X, Y, pm, cmap=PROSP_CMAP, shading='auto', vmin=0, vmax=1)
    _colorbar(fig, im2, ax2, p_label)
    levels = [0.4, 0.6, 0.75, 0.88]
    if np.nanmax(prob_map) > levels[0]:
        try:
            colors_c = ['#aaccff', '#ffdd88', '#ff8844', '#ff2222']
            cs = ax2.contour(X, Y, pm, levels=levels, colors=colors_c,
                             linewidths=[0.6, 0.9, 1.2, 1.8], alpha=0.85)
            ax2.clabel(cs, fmt=lambda v: f"{v*100:.0f}%", fontsize=6.5, inline=True)
        except (ValueError, TypeError):
            pass
    _plot_targets(ax2, targets)
    _plot_wells(ax2, well_df)
    ax2.set_title('HEDEF ' + ('ELVERİŞLİLİK' if unsup else 'OLASILIK') + ' HARİTASI',
                  color=TEXT_COLOR, fontsize=9, fontweight='bold')
    _fmt_ax(ax2, coregistrar)

    # ── Panel 3: Belirsizlik ──────────────────────────────────────────────
    ax3 = axes[2]
    _setup_dark_ax(ax3)
    unc_cmap = plt.get_cmap('Greens_r').with_extremes(bad=NODATA_COLOR)
    im3 = ax3.pcolormesh(X, Y, np.ma.masked_invalid(uncertainty_map),
                         cmap=unc_cmap, shading='auto')
    _colorbar(fig, im3, ax3, 'Belirsizlik (σ)' if not unsup else 'Kanıt uyumsuzluğu')
    ax3.set_title('BELİRSİZLİK\n(Düşük → Güvenilir)', color=TEXT_COLOR,
                  fontsize=9, fontweight='bold')
    _fmt_ax(ax3, coregistrar)

    # ── Panel 4: Güven-Düzeltilmiş Skor ──────────────────────────────────
    ax4 = axes[3]
    _setup_dark_ax(ax4)
    unc = np.where(np.isfinite(uncertainty_map), uncertainty_map, 0)
    score_map = np.clip(prob_map - uncertainty_penalty * unc, 0, 1)
    im4 = ax4.pcolormesh(X, Y, np.ma.masked_invalid(score_map), cmap=PROSP_CMAP,
                         shading='auto', vmin=0, vmax=1)
    _colorbar(fig, im4, ax4, 'Güven Skoru')
    _plot_targets(ax4, targets)
    ax4.set_title(f'GÜVEN-DÜZELTİLMİŞ SKOR\n(P − {uncertainty_penalty:g}·σ)',
                  color=TEXT_COLOR, fontsize=9, fontweight='bold')
    _fmt_ax(ax4, coregistrar)

    return _finalize(fig, save_path, show)


# ─────────────────────────────────────────────────────────────────────────────
# HEDEF RAPORU
# ─────────────────────────────────────────────────────────────────────────────

def plot_target_report(
    targets: list,
    target_type: str = 'generic',
    cv_results: dict = None,
    feature_importances=None,
    save_path: str = None,
    show: bool = True,
):
    """
    Hedef listesi, CV metrikleri ve feature importance rapor paneli.
    """
    n_panels = 2 + (1 if feature_importances is not None else 0) + \
        (1 if cv_results else 0)
    fig = plt.figure(figsize=(5.6 * n_panels, 5.6), facecolor=DARK_BG, layout='constrained')
    axes = np.atleast_1d(fig.subplots(1, n_panels))

    fig.suptitle(f'GeoAI Hedef Raporu — {target_type.upper()}',
                 color=TEXT_COLOR, fontsize=13, fontweight='bold')
    panel = 0

    # ── Panel 0: Hedef Öncelik Barı ───────────────────────────────────────
    ax = axes[panel]
    _setup_dark_ax(ax)
    panel += 1
    ax.set_title('HEDEF ÖNCELİK SIRASI', color=TEXT_COLOR, fontweight='bold')

    if targets:
        ranks = [f"T{t['rank']}" + (f" (Ö{t['priority']})" if 'priority' in t else '')
                 for t in targets]
        probs = [t['max_probability'] for t in targets]
        scores = [t['max_score'] for t in targets]
        uncs = [t['mean_uncertainty'] for t in targets]

        y_pos = np.arange(len(ranks))[::-1]
        bar_colors = [TARGET_COLORS[min(i, len(TARGET_COLORS) - 1)]
                      for i in range(len(ranks))]

        ax.barh(y_pos, probs, color=bar_colors, alpha=0.8, height=0.6,
                label='Maksimum Olasılık', edgecolor='#333')
        ax.barh(y_pos, scores, color=bar_colors, alpha=0.4, height=0.6,
                label='Güven Skoru', edgecolor='none')
        ax.barh(y_pos, uncs, color='gray', alpha=0.6, height=0.2, label='Belirsizlik')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(ranks, color=TEXT_COLOR, fontsize=8)
        ax.set_xlim(0, 1.12)
        ax.set_xlabel('Olasılık / Skor', color=MUTED)
        ax.axvline(0.60, color='yellow', ls='--', lw=1, alpha=0.6)
        ax.axvline(0.75, color='orange', ls='--', lw=1, alpha=0.6)
        ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor=MUTED, loc='lower right')
        for yp, p in zip(y_pos, probs):
            ax.text(p + 0.01, yp, f'{p:.3f}', va='center', color=TEXT_COLOR, fontsize=7.5)
    else:
        ax.text(0.5, 0.5, 'Hedef bulunamadı', ha='center', va='center',
                color=MUTED, transform=ax.transAxes)

    # ── Panel 1: Hedef Konumları Haritası ─────────────────────────────────
    ax = axes[panel]
    _setup_dark_ax(ax)
    panel += 1
    ax.set_title('HEDEF KONUMLARI\n(Daire boyutu ∝ Alan)', color=TEXT_COLOR, fontweight='bold')

    if targets:
        xs = np.array([t['x'] for t in targets])
        ys = np.array([t['y'] for t in targets])
        use_km = max(np.ptp(xs), np.ptp(ys), 0) >= 5000 or np.abs(xs).max() > 1e5
        scale = 1000.0 if use_km else 1.0
        areas = np.array([max(t['area_km2'], 1e-6) for t in targets])
        sizes = 60 + 600 * np.sqrt(areas / areas.max())

        sc = ax.scatter(xs / scale, ys / scale, c=[t['max_probability'] for t in targets],
                        s=sizes, cmap=PROSP_CMAP, vmin=0, vmax=1,
                        edgecolors='white', linewidths=0.7, alpha=0.9)
        _colorbar(fig, sc, ax, 'Olasılık')
        for t in targets:
            ax.annotate(
                f"T{t['rank']}", (t['x'] / scale, t['y'] / scale),
                textcoords='offset points', xytext=(7, 6),
                color=TARGET_COLORS[min(t['rank'] - 1, len(TARGET_COLORS) - 1)],
                fontsize=8.5, fontweight='bold',
            )
        ax.margins(0.2)
        unit = ' (km)' if use_km else ''
        ax.set_xlabel('X' + unit, color=MUTED)
        ax.set_ylabel('Y' + unit, color=MUTED)

    # ── Panel 2: Feature Importance (opsiyonel) ───────────────────────────
    if feature_importances is not None and panel < len(axes):
        ax = axes[panel]
        _setup_dark_ax(ax)
        panel += 1

        top_n = min(15, len(feature_importances))
        top = feature_importances.head(top_n)
        y_pos = np.arange(top_n)[::-1]
        colors_fi = plt.get_cmap('plasma')(np.linspace(0.85, 0.2, top_n))
        ax.barh(y_pos, top.values, color=colors_fi, height=0.65, edgecolor='#333')
        labels = [n.replace('__', ' · ').replace('_', ' ')[:34] for n in top.index]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color=TEXT_COLOR, fontsize=7)
        ax.set_xlabel('Önem Skoru (ortalama impurity azalması)', color=MUTED)
        ax.set_title('EN ÖNEMLİ ÖZELLİKLER\n(Random Forest)', color=TEXT_COLOR,
                     fontweight='bold')

    # ── Panel 3: CV Metrikleri (opsiyonel) ────────────────────────────────
    if cv_results and panel < len(axes):
        ax = axes[panel]
        _setup_dark_ax(ax)
        panel += 1

        model_names = list(cv_results.keys())
        aucs = [cv_results[m].get('auc_roc', cv_results[m].get('auc', 0))
                for m in model_names]
        aps = [cv_results[m].get('auc_pr', cv_results[m].get('avg_precision', 0))
               for m in model_names]

        x_pos = np.arange(len(model_names))
        w = 0.35
        bars1 = ax.bar(x_pos - w/2, aucs, w, label='ROC-AUC',
                       color='#58a6ff', alpha=0.85, edgecolor='#333')
        bars2 = ax.bar(x_pos + w/2, aps, w, label='PR-AUC',
                       color='#ff7c43', alpha=0.85, edgecolor='#333')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.upper() for m in model_names], color=TEXT_COLOR, fontsize=8.5)
        ax.set_ylim(0, 1.1)
        ax.axhline(0.7, color='yellow', ls='--', lw=1, alpha=0.5)
        ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.3)
        ax.set_ylabel('Metrik Değeri (out-of-fold)', color=MUTED)
        ax.set_title('CROSS-VALIDATION METRİKLERİ', color=TEXT_COLOR, fontweight='bold')
        ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=MUTED, loc='lower right')

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
                    ha='center', va='bottom', color=TEXT_COLOR, fontsize=7.5)

    return _finalize(fig, save_path, show)


# ─────────────────────────────────────────────────────────────────────────────
# ÇOKLU HEDEF KARŞILAŞTIRMA
# ─────────────────────────────────────────────────────────────────────────────

def plot_multi_target_comparison(
    multi_results: dict,
    coregistrar,
    save_path: str = None,
    show: bool = True,
):
    """
    Birden fazla hedef tipi için olasılık haritalarını yan yana gösterir.
    multi_results: {target_type: {'prob_map', 'uncertainty_map', 'targets'}}
    """
    n = len(multi_results)
    if n == 0:
        return None

    fig = plt.figure(figsize=(6 * n, 10), facecolor=DARK_BG, layout='constrained')
    axes = np.asarray(fig.subplots(2, n)).reshape(2, n)
    fig.suptitle('GeoAI — Çok Hedefli Karşılaştırma',
                 color=TEXT_COLOR, fontsize=14, fontweight='bold')

    X, Y = np.meshgrid(coregistrar.ref_x, coregistrar.ref_y)
    unc_cmap = plt.get_cmap('Greens_r').with_extremes(bad=NODATA_COLOR)

    for col, (typ, res) in enumerate(multi_results.items()):
        unsup = res.get('mode', 'supervised') != 'supervised'
        tag = ' [denetimsiz]' if unsup else ''

        ax = axes[0][col]
        _setup_dark_ax(ax)
        im = ax.pcolormesh(X, Y, np.ma.masked_invalid(res['prob_map']), cmap=PROSP_CMAP,
                           shading='auto', vmin=0, vmax=1)
        _colorbar(fig, im, ax)
        _plot_targets(ax, res['targets'], limit=6)
        ax.set_title(f'{typ.upper()}{tag}\nOlasılık', color=TEXT_COLOR,
                     fontweight='bold', fontsize=10)
        _fmt_ax(ax, coregistrar)

        ax2 = axes[1][col]
        _setup_dark_ax(ax2)
        im2 = ax2.pcolormesh(X, Y, np.ma.masked_invalid(res['uncertainty_map']),
                             cmap=unc_cmap, shading='auto')
        _colorbar(fig, im2, ax2)
        ax2.set_title(f'{typ.upper()}{tag}\nBelirsizlik', color=TEXT_COLOR,
                      fontweight='bold', fontsize=10)
        _fmt_ax(ax2, coregistrar)

    return _finalize(fig, save_path, show)
