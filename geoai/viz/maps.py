"""
geoai.viz.maps
==============
Jeofizik veri ve model sonuçlarının görselleştirilmesi.
Tüm grafikler koyu (dark) tema ile bilimsel renk haritaları kullanır.
"""

import os
import numpy as np
import matplotlib
if os.name != 'nt' and not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# TEMA VE RENKLERvE
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG = '#0d1117'
PANEL_BG = '#161b22'
TEXT_COLOR = '#e6edf3'
GRID_COLOR = '#21262d'
ACCENT = '#58a6ff'

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
PROSP_CMAP = LinearSegmentedColormap.from_list('prospectivity', _colors_prosp)

# Jeofizik katman renk haritaları
LAYER_CMAPS = {
    'magnetic':     'RdBu_r',
    'gravity':      'seismic',
    'resistivity':  'viridis_r',
    'ip':           'hot',
    'chargeability': 'hot',
    'geochemistry': 'YlOrRd',
    'seismic':      'Spectral_r',
    'geology':      'tab20',
    'default':      'RdBu_r',
}


def _get_cmap(layer_name: str) -> str:
    for k, v in LAYER_CMAPS.items():
        if k in layer_name.lower():
            return v
    return LAYER_CMAPS['default']


def _setup_dark_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linewidth=0.4, alpha=0.5)


def _km_label(val, _):
    return f"{val/1000:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# GRİD GÖRÜNTÜLEME
# ─────────────────────────────────────────────────────────────────────────────

def plot_input_layers(
    registered_layers: dict,
    coregistrar,
    save_path: str = None,
    show: bool = True,
    max_cols: int = 3,
    figsize_per: tuple = (5, 4),
):
    """
    Co-register edilmiş giriş katmanlarını çizer.

    registered_layers: {isim: 2D ndarray}
    """
    names = list(registered_layers.keys())
    n = len(names)
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols

    fig_w = figsize_per[0] * ncols
    fig_h = figsize_per[1] * nrows + 0.8
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=DARK_BG)
    fig.suptitle(
        'JEOFİZİK GİRİŞ KATMANLARI — Co-registered',
        color=TEXT_COLOR, fontsize=13, fontweight='bold', y=1.01
    )

    x = coregistrar.ref_x
    y = coregistrar.ref_y
    X, Y = np.meshgrid(x, y)

    for idx, name in enumerate(names):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        _setup_dark_ax(ax)

        grid = registered_layers[name]
        cmap = _get_cmap(name)

        # Robust color limits (2-98 percentile)
        valid = grid[~np.isnan(grid)]
        if len(valid) > 0:
            vmin, vmax = np.percentile(valid, [2, 98])
        else:
            vmin, vmax = 0, 1

        im = ax.pcolormesh(
            X, Y, grid,
            cmap=cmap, shading='auto',
            vmin=vmin, vmax=vmax,
        )
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.ax.tick_params(colors='#8b949e', labelsize=7)

        # NaN yüzdesi
        n_nan = np.isnan(grid).sum()
        pct_nan = 100 * n_nan / grid.size
        suffix = f" [{pct_nan:.0f}% boş]" if pct_nan > 5 else ""

        ax.set_title(
            name.replace('_', ' ').upper() + suffix,
            color=TEXT_COLOR, fontsize=8, fontweight='bold', pad=4
        )
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_km_label))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_km_label))
        ax.set_xlabel('X (km)', color='#8b949e', fontsize=7)
        ax.set_ylabel('Y (km)', color='#8b949e', fontsize=7)

    # Boş eksenler
    for idx in range(n, nrows * ncols):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        print(f"  ✓ Kaydedildi: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# OLASIM HARİTASI
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
):
    """
    Ana prospectivity sonuç haritası.
    4 panel: Referans | Olasılık | Belirsizlik | Hedef Yakını
    """
    x = coregistrar.ref_x
    y = coregistrar.ref_y
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(20, 5.5), facecolor=DARK_BG)
    fig.suptitle(
        f'GeoAI Prospectivity — {target_type.upper()}',
        color=TEXT_COLOR, fontsize=14, fontweight='bold'
    )

    gs = GridSpec(1, 4, figure=fig, wspace=0.28, left=0.04, right=0.98)

    # ── Panel 1: Referans katman ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    _setup_dark_ax(ax1)
    if reference_layer is not None:
        valid = reference_layer[~np.isnan(reference_layer)]
        vmin, vmax = (np.percentile(valid, [2, 98]) if len(valid) > 0 else (0, 1))
        im1 = ax1.pcolormesh(X, Y, reference_layer, cmap='RdBu_r',
                              shading='auto', vmin=vmin, vmax=vmax)
        cb1 = plt.colorbar(im1, ax=ax1, fraction=0.045, pad=0.03)
        cb1.ax.tick_params(colors='#8b949e', labelsize=7)
    ax1.set_title(reference_name, color=TEXT_COLOR, fontsize=9, fontweight='bold')
    _fmt_ax(ax1)

    # Kuyuları çiz
    if well_df is not None:
        _plot_wells(ax1, well_df)

    # ── Panel 2: Olasılık Haritası ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    _setup_dark_ax(ax2)
    im2 = ax2.pcolormesh(X, Y, prob_map, cmap=PROSP_CMAP,
                          shading='auto', vmin=0, vmax=1)
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.045, pad=0.03)
    cb2.set_label('Olasılık', color='#8b949e', fontsize=8)
    cb2.ax.tick_params(colors='#8b949e', labelsize=7)

    # Konturlar
    try:
        levels = [0.4, 0.6, 0.75, 0.88]
        colors_c = ['#aaccff', '#ffdd88', '#ff8844', '#ff2222']
        cs = ax2.contour(X, Y, prob_map, levels=levels,
                         colors=colors_c, linewidths=[0.6, 0.9, 1.2, 1.8],
                         alpha=0.85)
        ax2.clabel(cs, fmt=lambda v: f"{v*100:.0f}%",
                   colors=colors_c, fontsize=6.5, inline=True)
    except Exception:
        pass

    # Hedef yıldızları
    for t in targets[:10]:
        cidx = min(t['rank'] - 1, len(TARGET_COLORS) - 1)
        color = TARGET_COLORS[cidx]
        ax2.scatter(
            t['x'], t['y'],
            c=color, s=130, marker='*',
            edgecolors='white', linewidths=0.7, zorder=10,
        )
        ax2.annotate(
            f"T{t['rank']}",
            (t['x'], t['y']),
            textcoords='offset points', xytext=(5, 5),
            color=color, fontsize=8, fontweight='bold', zorder=11,
        )

    if well_df is not None:
        _plot_wells(ax2, well_df)

    ax2.set_title('HEDEF OLASILIK HARİTASI', color=TEXT_COLOR,
                  fontsize=9, fontweight='bold')
    _fmt_ax(ax2)

    # ── Panel 3: Belirsizlik ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    _setup_dark_ax(ax3)
    im3 = ax3.pcolormesh(X, Y, uncertainty_map,
                          cmap='Greens_r', shading='auto')
    cb3 = plt.colorbar(im3, ax=ax3, fraction=0.045, pad=0.03)
    cb3.set_label('Belirsizlik (σ)', color='#8b949e', fontsize=8)
    cb3.ax.tick_params(colors='#8b949e', labelsize=7)
    ax3.set_title('BELİRSİZLİK\n(Düşük → Güvenilir)', color=TEXT_COLOR,
                  fontsize=9, fontweight='bold')
    _fmt_ax(ax3)

    # ── Panel 4: Güven-Düzeltilmiş Skor ──────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    _setup_dark_ax(ax4)
    score_map = np.clip(prob_map - 0.3 * uncertainty_map, 0, 1)
    im4 = ax4.pcolormesh(X, Y, score_map, cmap=PROSP_CMAP,
                          shading='auto', vmin=0, vmax=1)
    cb4 = plt.colorbar(im4, ax=ax4, fraction=0.045, pad=0.03)
    cb4.set_label('Güven Skoru', color='#8b949e', fontsize=8)
    cb4.ax.tick_params(colors='#8b949e', labelsize=7)

    for t in targets[:10]:
        cidx = min(t['rank'] - 1, len(TARGET_COLORS) - 1)
        ax4.scatter(
            t['x'], t['y'],
            c=TARGET_COLORS[cidx], s=130, marker='*',
            edgecolors='white', linewidths=0.7, zorder=10,
        )
        ax4.annotate(f"T{t['rank']}", (t['x'], t['y']),
                     textcoords='offset points', xytext=(5, 5),
                     color=TARGET_COLORS[cidx], fontsize=8,
                     fontweight='bold', zorder=11)

    ax4.set_title('GÜVEN-DÜZELTİLMİŞ SKOR\n(P − 0.3·σ)', color=TEXT_COLOR,
                  fontsize=9, fontweight='bold')
    _fmt_ax(ax4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        print(f"  ✓ Kaydedildi: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _fmt_ax(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_km_label))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_km_label))
    ax.set_xlabel('X (km)', color='#8b949e', fontsize=7)
    ax.set_ylabel('Y (km)', color='#8b949e', fontsize=7)


def _plot_wells(ax, well_df):
    if well_df is None or len(well_df) == 0:
        return
    pos = well_df[well_df['label'] == 1]
    neg = well_df[well_df['label'] == 0]
    if len(pos):
        ax.scatter(pos['x'], pos['y'], c='lime', s=50, marker='^',
                   edgecolors='white', linewidths=0.7, zorder=8,
                   label='Pozitif Kuyu')
    if len(neg):
        ax.scatter(neg['x'], neg['y'], c='red', s=50, marker='v',
                   edgecolors='white', linewidths=0.7, zorder=8,
                   label='Negatif Kuyu', alpha=0.7)


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
    fig, axes = plt.subplots(1, n_panels,
                              figsize=(5.5 * n_panels, 5.5),
                              facecolor=DARK_BG)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f'GeoAI Hedef Raporu — {target_type.upper()}',
        color=TEXT_COLOR, fontsize=13, fontweight='bold'
    )

    panel = 0

    # ── Panel 0: Hedef Öncelik Barı ───────────────────────────────────────
    ax = axes[panel]
    _setup_dark_ax(ax)
    panel += 1

    if targets:
        ranks = [f"T{t['rank']}" for t in targets]
        probs = [t['max_probability'] for t in targets]
        scores = [t['max_score'] for t in targets]
        uncs = [t['mean_uncertainty'] for t in targets]

        y_pos = np.arange(len(ranks))
        bar_colors = [TARGET_COLORS[min(i, len(TARGET_COLORS)-1)]
                      for i in range(len(ranks))]

        ax.barh(y_pos[::-1], probs, color=bar_colors, alpha=0.8,
                height=0.6, label='Maksimum Olasılık', edgecolor='#333')
        ax.barh(y_pos[::-1], scores, color=bar_colors, alpha=0.4,
                height=0.6, label='Güven Skoru', edgecolor='none')

        # Belirsizlik bar'ı
        ax.barh(y_pos[::-1], uncs, color='gray', alpha=0.5,
                height=0.2, label='Belirsizlik')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(ranks[::-1], color=TEXT_COLOR, fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel('Olasılık / Skor', color='#8b949e')
        ax.axvline(0.5, color='yellow', ls='--', lw=1, alpha=0.6)
        ax.axvline(0.75, color='orange', ls='--', lw=1, alpha=0.6)
        ax.set_title('HEDEF ÖNCELİK SIRASI', color=TEXT_COLOR, fontweight='bold')
        ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor='#8b949e',
                  loc='lower right')

        for i, (p, s, u) in enumerate(zip(probs[::-1], scores[::-1], uncs[::-1])):
            ax.text(p + 0.01, i, f'{p:.3f}', va='center',
                    color=TEXT_COLOR, fontsize=7.5)

    # ── Panel 1: Hedef Konumları Haritası ─────────────────────────────────
    ax = axes[panel]
    _setup_dark_ax(ax)
    panel += 1

    if targets:
        xs = [t['x'] / 1000 for t in targets]
        ys = [t['y'] / 1000 for t in targets]
        probs = [t['max_probability'] for t in targets]
        areas = [max(t['area_km2'], 0.001) for t in targets]

        sc = ax.scatter(
            xs, ys,
            c=probs, s=[a * 1000 + 80 for a in areas],
            cmap=PROSP_CMAP, vmin=0, vmax=1,
            edgecolors='white', linewidths=0.7, alpha=0.9,
        )
        plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.03,
                     label='Olasılık').ax.tick_params(colors='#8b949e', labelsize=7)

        for t in targets:
            ax.annotate(
                f"T{t['rank']}",
                (t['x'] / 1000, t['y'] / 1000),
                textcoords='offset points', xytext=(6, 5),
                color=TARGET_COLORS[min(t['rank']-1, len(TARGET_COLORS)-1)],
                fontsize=8.5, fontweight='bold',
            )

        ax.set_xlabel('X (km)', color='#8b949e')
        ax.set_ylabel('Y (km)', color='#8b949e')
        ax.set_title('HEDEF KONUMLARI\n(Daire boyutu ∝ Alan)',
                     color=TEXT_COLOR, fontweight='bold')

    # ── Panel 2: Feature Importance (opsiyonel) ───────────────────────────
    if feature_importances is not None and panel < len(axes):
        ax = axes[panel]
        _setup_dark_ax(ax)
        panel += 1

        top_n = min(15, len(feature_importances))
        top = feature_importances.head(top_n)

        y_pos = np.arange(top_n)
        colors_fi = plt.cm.plasma(np.linspace(0.85, 0.2, top_n))

        ax.barh(y_pos[::-1], top.values, color=colors_fi,
                height=0.65, edgecolor='#333')

        labels = [n.replace('__', '\n').replace('_', ' ')[:30] for n in top.index]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels[::-1], color=TEXT_COLOR, fontsize=7)
        ax.set_xlabel('Önem Skoru', color='#8b949e')
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
        ax.set_xticklabels([m.upper() for m in model_names],
                            color=TEXT_COLOR, fontsize=8.5)
        ax.set_ylim(0, 1.08)
        ax.axhline(0.7, color='yellow', ls='--', lw=1, alpha=0.5)
        ax.axhline(0.5, color='red', ls='--', lw=1, alpha=0.3)
        ax.set_ylabel('Metrik Değeri', color='#8b949e')
        ax.set_title('CROSS-VALIDATION METRİKLERİ', color=TEXT_COLOR,
                     fontweight='bold')
        ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor='#8b949e')

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom',
                    color=TEXT_COLOR, fontsize=7.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        print(f"  ✓ Kaydedildi: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


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
        return

    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10), facecolor=DARK_BG)
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        'GeoAI — Çok Hedefli Karşılaştırma',
        color=TEXT_COLOR, fontsize=14, fontweight='bold'
    )

    x = coregistrar.ref_x
    y = coregistrar.ref_y
    X, Y = np.meshgrid(x, y)

    for col, (typ, res) in enumerate(multi_results.items()):
        prob_map = res['prob_map']
        unc_map = res['uncertainty_map']
        targets = res['targets']

        # Üst: Olasılık
        ax = axes[0][col]
        _setup_dark_ax(ax)
        im = ax.pcolormesh(X, Y, prob_map, cmap=PROSP_CMAP,
                            shading='auto', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03).ax.tick_params(
            colors='#8b949e', labelsize=7
        )
        for t in targets[:6]:
            cidx = min(t['rank'] - 1, len(TARGET_COLORS) - 1)
            ax.scatter(t['x'], t['y'], c=TARGET_COLORS[cidx],
                       s=120, marker='*', edgecolors='white',
                       linewidths=0.7, zorder=10)
            ax.annotate(f"T{t['rank']}", (t['x'], t['y']),
                        xytext=(5, 5), textcoords='offset points',
                        color=TARGET_COLORS[cidx], fontsize=8,
                        fontweight='bold', zorder=11)
        ax.set_title(f'{typ.upper()}\nOlasılık', color=TEXT_COLOR,
                     fontweight='bold', fontsize=10)
        _fmt_ax(ax)

        # Alt: Belirsizlik
        ax2 = axes[1][col]
        _setup_dark_ax(ax2)
        im2 = ax2.pcolormesh(X, Y, unc_map, cmap='Greens_r', shading='auto')
        plt.colorbar(im2, ax=ax2, fraction=0.045, pad=0.03).ax.tick_params(
            colors='#8b949e', labelsize=7
        )
        ax2.set_title(f'{typ.upper()}\nBelirsizlik', color=TEXT_COLOR,
                      fontweight='bold', fontsize=10)
        _fmt_ax(ax2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
        print(f"  ✓ Kaydedildi: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
