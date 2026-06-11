"""
Step 2 — Exploratory Data Analysis
-------------------------------------
Plots saved to outputs/figures/:
  01_aqi_distribution.png
  02_missing_values.png
  03_correlation_heatmap.png
  04_feature_boxplots.png
  05_aqi_by_pollutant.png
  06_monthly_aqi_trend.png
  07_taiwan_map.png
"""

import os
import sys
import warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BASE_DIR, DATA_PATH, FIGURES_DIR, RESULTS_DIR,
    NUMERIC_FEATURES, TARGET, CAT_FEATURE,
    POLLUTANT_MAP, POLLUTANT_NONE_LABEL,
)

sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})


# ── helpers ──────────────────────────────────────────────────────────────────

def _save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {name}")


def _load_raw() -> pd.DataFrame:
    """Load raw CSV with minimal cleaning for EDA (keep lat/lon/county)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = pd.read_csv(DATA_PATH, low_memory=False)
        df['date'] = pd.to_datetime(df['date'], format='mixed')
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df[TARGET].notna() & (df[TARGET] >= 0)]
    df[CAT_FEATURE] = df[CAT_FEATURE].map(
        lambda x: POLLUTANT_MAP.get(x, POLLUTANT_NONE_LABEL) if pd.notna(x) else POLLUTANT_NONE_LABEL
    )
    return df


# ── 1. AQI distribution ───────────────────────────────────────────────────────

def plot_aqi_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df[TARGET], bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
    axes[0].set_xlabel('AQI')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('AQI Distribution (Histogram)')
    axes[0].axvline(df[TARGET].mean(), color='red', linestyle='--', label=f"Mean={df[TARGET].mean():.1f}")
    axes[0].axvline(df[TARGET].median(), color='orange', linestyle='--', label=f"Median={df[TARGET].median():.1f}")
    axes[0].legend()

    df[TARGET].plot.kde(ax=axes[1], color='steelblue', linewidth=2)
    axes[1].set_xlabel('AQI')
    axes[1].set_title('AQI Density (KDE)')

    fig.suptitle('AQI Distribution — 2024 Dataset', fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, '01_aqi_distribution.png')


# ── 2. Missing values ─────────────────────────────────────────────────────────

def plot_missing_values(df):
    miss = df[NUMERIC_FEATURES + [TARGET]].isnull().mean() * 100
    miss = miss[miss > 0].sort_values(ascending=False)

    if miss.empty:
        print("  No missing values to plot — skipping 02.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    miss.plot.bar(ax=ax, color='salmon', edgecolor='white')
    ax.set_ylabel('Missing (%)')
    ax.set_title('Missing Value Rate per Feature', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    _save(fig, '02_missing_values.png')


# ── 3. Correlation heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap(df):
    corr_cols = NUMERIC_FEATURES + [TARGET]
    corr = df[corr_cols].corr(method='pearson')

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax,
        annot=True, fmt='.2f', annot_kws={'size': 7},
        cmap='coolwarm', center=0, vmin=-1, vmax=1,
        linewidths=0.3, square=True,
    )
    ax.set_title('Pearson Correlation Heatmap (Numeric Features + AQI)', fontweight='bold', pad=12)
    fig.tight_layout()
    _save(fig, '03_correlation_heatmap.png')


# ── 4. Feature box-plots ──────────────────────────────────────────────────────

def plot_feature_boxplots(df):
    cols = NUMERIC_FEATURES
    n = len(cols)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        data = df[col].dropna()
        axes[i].boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='steelblue'),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='steelblue'),
                        flierprops=dict(marker='o', markersize=1, alpha=0.3))
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(axis='x', bottom=False, labelbottom=False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Feature Box-Plots (Outlier & Scale Overview)', fontweight='bold', y=1.01)
    fig.tight_layout()
    _save(fig, '04_feature_boxplots.png')


# ── 5. AQI by pollutant category ─────────────────────────────────────────────

def plot_aqi_by_pollutant(df):
    order = df.groupby(CAT_FEATURE)[TARGET].median().sort_values(ascending=False).index

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box-plot
    data_by_pol = [df[df[CAT_FEATURE] == cat][TARGET].dropna().values for cat in order]
    bp = axes[0].boxplot(data_by_pol, patch_artist=True, vert=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(order)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[0].set_xticks(range(1, len(order) + 1))
    axes[0].set_xticklabels(order, rotation=30, ha='right', fontsize=8)
    axes[0].set_ylabel('AQI')
    axes[0].set_title('AQI Distribution by Dominant Pollutant')

    # Bar chart of mean AQI
    means = df.groupby(CAT_FEATURE)[TARGET].mean().reindex(order)
    stds  = df.groupby(CAT_FEATURE)[TARGET].std().reindex(order)
    x = range(len(order))
    axes[1].bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor='white')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, rotation=30, ha='right', fontsize=8)
    axes[1].set_ylabel('Mean AQI ± std')
    axes[1].set_title('Mean AQI per Pollutant Category')

    fig.suptitle('AQI by Dominant Pollutant Category', fontweight='bold')
    fig.tight_layout()
    _save(fig, '05_aqi_by_pollutant.png')


# ── 6. Monthly AQI trend ──────────────────────────────────────────────────────

def plot_monthly_trend(df):
    df = df.copy()
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month')[TARGET].agg(['mean', 'std', 'median']).reset_index()
    monthly['month_str'] = monthly['month'].astype(str)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(
        monthly['month_str'],
        monthly['mean'] - monthly['std'],
        monthly['mean'] + monthly['std'],
        alpha=0.2, color='steelblue', label='±1 std'
    )
    ax.plot(monthly['month_str'], monthly['mean'],   'o-', color='steelblue', label='Mean AQI', linewidth=2)
    ax.plot(monthly['month_str'], monthly['median'], 's--', color='orange',   label='Median AQI', linewidth=1.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('AQI')
    ax.set_title('Monthly AQI Trend — Jan to Aug 2024', fontweight='bold')
    ax.legend()
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    _save(fig, '06_monthly_aqi_trend.png')


# ── 7. Taiwan map ─────────────────────────────────────────────────────────────

def plot_taiwan_map():
    map_path = os.path.join(RESULTS_DIR, 'map_data.pkl')
    if not os.path.exists(map_path):
        print("  map_data.pkl not found — run 01_preprocessing.py first.")
        return

    map_df = pd.read_pickle(map_path)
    map_df[TARGET] = pd.to_numeric(map_df[TARGET], errors='coerce')

    county_stats = (
        map_df.groupby('county')
        .agg(avg_aqi=(TARGET, 'mean'), n=(TARGET, 'count'))
        .dropna()
        .reset_index()
    )

    # Shapefile name → dataset county name
    SHP_TO_COUNTY = {
        'Kinmen':        'Kinmen County',
        'Matsu Islands': 'Lienchiang County',
        'Penghu':        'Penghu County',
        'Taoyuan':       'Taoyuan City',
        'Hsinchu':       'Hsinchu County',
        'Hsinchu City':  'Hsinchu City',
        'Miaoli':        'Miaoli County',
        'Taichung City': 'Taichung City',
        'Changhua':      'Changhua County',
        'Yunlin':        'Yunlin County',
        'Chiayi':        'Chiayi County',
        'Chiayi City':   'Chiayi City',
        'Tainan City':   'Tainan City',
        'Kaohsiung City':'Kaohsiung City',
        'Pingtung':      'Pingtung County',
        'Taitung':       'Taitung County',
        'Hualien':       'Hualien County',
        'Yilan':         'Yilan County',
        'New Taipei City':'New Taipei City',
        'Keelung City':  'Keelung City',
        'Nantou':        'Nantou County',
        'Taipei City':   'Taipei City',
    }

    import geopandas as gpd

    shp_path = os.path.join(BASE_DIR, 'Taiwanmap_shp', 'tw.shp')
    gdf = gpd.read_file(shp_path)
    gdf['county'] = gdf['name'].map(SHP_TO_COUNTY)

    # Merge AQI stats into geodataframe
    gdf = gdf.merge(county_stats[['county', 'avg_aqi', 'n']], on='county', how='left')

    vmin = gdf['avg_aqi'].min()
    vmax = gdf['avg_aqi'].max()
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_facecolor('#a8d8ea')   # ocean blue

    # Draw each county filled by avg AQI
    gdf.plot(
        column='avg_aqi',
        cmap=cmap,
        norm=norm,
        linewidth=0.6,
        edgecolor='#444444',
        ax=ax,
        missing_kwds={'color': '#cccccc', 'edgecolor': '#999999', 'label': 'No data'},
    )

    # County name labels at polygon centroid
    for _, row in gdf.iterrows():
        if row.geometry is None:
            continue
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        label = row['county'] if pd.notna(row['county']) else row['name']
        # Shorten long names for readability
        label = (label
                 .replace(' County', ' Co.')
                 .replace(' City', ' City'))
        ax.annotate(
            label, xy=(cx, cy),
            ha='center', va='center',
            fontsize=5.8, fontweight='bold', color='#1a1a1a',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.55, lw=0),
        )

    ax.set_xlim(118.8, 122.8)
    ax.set_ylim(21.8, 25.7)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('Average AQI by County — Taiwan, Jan-Aug 2024',
                 fontsize=13, fontweight='bold', pad=14)
    ax.grid(True, alpha=0.2, linestyle='--')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label('Avg AQI', fontsize=10)

    fig.tight_layout()
    _save(fig, '07_taiwan_map.png')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== EDA ===")
    df = _load_raw()
    print(f"  Working with {len(df):,} rows for EDA\n")

    plot_aqi_distribution(df)
    plot_missing_values(df)
    plot_correlation_heatmap(df)
    plot_feature_boxplots(df)
    plot_aqi_by_pollutant(df)
    plot_monthly_trend(df)
    plot_taiwan_map()

    print("\nEDA complete.\n")


if __name__ == '__main__':
    main()
