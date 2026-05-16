"""
CartShield AI — Customer Risk Segmentation
Clusters customers by behavioral risk patterns using KMeans + PCA visualization.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')


SEGMENT_LABELS = {
    0: ("🟢 Trustworthy",      "#27ae60"),
    1: ("🟡 Moderate Risk",    "#f39c12"),
    2: ("🔴 High Risk",        "#e74c3c"),
    3: ("⚫ Extreme Risk",     "#2c3e50"),
}


def build_customer_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transaction data to customer-level risk features."""
    grp = df.groupby('customer_id')

    profile = pd.DataFrame({
        'total_orders':          grp['order_id'].count(),
        'total_returns':         grp['has_return'].sum(),
        'fraud_count':           grp['is_fraud'].sum(),
        'avg_order_value':       grp['order_value'].mean(),
        'avg_refund_amount':     grp['refund_amount'].mean(),
        'max_refund_frequency':  grp['refund_frequency_30d'].max(),
        'avg_delivery_days':     grp['delivery_days'].mean(),
        'account_age_days':      grp['account_age_days'].first(),
    })

    profile['return_rate']       = profile['total_returns'] / profile['total_orders'].clip(1)
    profile['fraud_rate']        = profile['fraud_count']   / profile['total_orders'].clip(1)
    profile['refund_to_order_r'] = profile['avg_refund_amount'] / profile['avg_order_value'].clip(0.01)

    return profile.reset_index()


def segment_customers(df_transactions: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Cluster customers and assign risk tiers."""
    profile = build_customer_profile(df_transactions)

    feature_cols = [
        'return_rate', 'fraud_rate', 'refund_to_order_r',
        'max_refund_frequency', 'avg_order_value', 'account_age_days',
    ]
    X = profile[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    profile['cluster'] = km.fit_predict(X_scaled)

    # Order clusters by average fraud_rate (ascending → low risk first)
    cluster_fraud = profile.groupby('cluster')['fraud_rate'].mean().sort_values()
    cluster_rank  = {c: i for i, c in enumerate(cluster_fraud.index)}
    profile['risk_segment'] = profile['cluster'].map(cluster_rank)
    profile['segment_label'] = profile['risk_segment'].map(
        lambda x: SEGMENT_LABELS.get(x, ("Unknown", "#7f8c8d"))[0]
    )

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    profile['pca_x'] = coords[:, 0]
    profile['pca_y'] = coords[:, 1]

    _plot_segments(profile)
    return profile


def _plot_segments(profile: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#0f1117')

    # PCA scatter
    ax = axes[0]
    ax.set_facecolor('#0f1117')
    for seg, (label, color) in SEGMENT_LABELS.items():
        mask = profile['risk_segment'] == seg
        ax.scatter(
            profile.loc[mask, 'pca_x'],
            profile.loc[mask, 'pca_y'],
            c=color, alpha=0.7, s=18, label=label
        )
    ax.set_title('Customer Risk Segments (PCA)', color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('PC1', color='#aaa')
    ax.set_ylabel('PC2', color='#aaa')
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)

    # Bar: avg fraud rate per segment
    ax2 = axes[1]
    ax2.set_facecolor('#0f1117')
    seg_stats = profile.groupby('segment_label')['fraud_rate'].mean().sort_values()
    colors_bar = [SEGMENT_LABELS[i][1] for i in range(4) if SEGMENT_LABELS[i][0] in seg_stats.index]
    seg_stats.plot(kind='barh', ax=ax2, color=['#27ae60','#f39c12','#e74c3c','#2c3e50'][:len(seg_stats)])
    ax2.set_title('Avg Fraud Rate by Segment', color='white', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Avg Fraud Rate', color='#aaa')
    ax2.tick_params(colors='#aaa')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333')

    plt.suptitle('CartShield AI — Customer Segmentation', color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()

    out = os.path.join(MODELS_DIR, 'customer_segments.png')
    os.makedirs(MODELS_DIR, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor='#0f1117')
    plt.close()
    print(f"   Segmentation plot → {out}")


if __name__ == '__main__':
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'raw', 'transactions.csv'
    )
    df = pd.read_csv(data_path)
    profile = segment_customers(df)
    print(profile.groupby('segment_label')[['total_orders','return_rate','fraud_rate']].mean().round(3))
