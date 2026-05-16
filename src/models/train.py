"""
CartShield AI — Model Training Pipeline
XGBoost fraud classifier with SHAP explainability + model persistence.
"""

import os
import sys
import joblib
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

warnings.filterwarnings('ignore')

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.data.feature_engineering import FeatureEngineer

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(transactions_path: str = None):
    if transactions_path is None:
        transactions_path = os.path.join(DATA_DIR, 'raw', 'transactions.csv')

    print("📦 Loading data...")
    df = pd.read_csv(transactions_path)
    print(f"   Rows: {len(df):,} | Fraud rate: {df['is_fraud'].mean():.2%}")

    # Only returns can be flagged as fraud in this context
    df_ret = df[df['has_return'] == 1].copy()
    print(f"   Returns only: {len(df_ret):,}")

    print("🔧 Engineering features...")
    fe = FeatureEngineer()
    df_feat = fe.fit_transform(df_ret)

    feature_cols = fe.get_feature_names()
    # Drop columns that might leak the label
    feature_cols = [c for c in feature_cols if c not in ['is_fraud', 'fraud_type', 'has_return']]

    X = df_feat[feature_cols].fillna(0)
    y = df_feat['is_fraud']

    print(f"   Features: {len(feature_cols)}")
    print(f"   Class balance — Legit: {(y==0).sum():,} | Fraud: {(y==1).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Class weight
    weights = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train)
    scale_pos = weights[1] / weights[0]

    print(f"\n🚀 Training XGBoost (scale_pos_weight={scale_pos:.1f})...")
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric='aucpr',
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc  = roc_auc_score(y_test, y_prob)
    pr_auc   = average_precision_score(y_test, y_prob)

    print("\n" + "="*55)
    print("📊 EVALUATION RESULTS")
    print("="*55)
    print(f"   ROC-AUC  : {roc_auc:.4f}")
    print(f"   PR-AUC   : {pr_auc:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    metrics = {
        "roc_auc":  round(roc_auc, 4),
        "pr_auc":   round(pr_auc, 4),
        "n_train":  int(len(X_train)),
        "n_test":   int(len(X_test)),
        "fraud_rate_train": round(float(y_train.mean()), 4),
        "feature_count": len(feature_cols),
        "features": feature_cols,
    }

    # ---------------------------------------------------------------------------
    # Save artefacts
    # ---------------------------------------------------------------------------
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, 'cartshield_xgb.pkl'))
    joblib.dump(fe,    os.path.join(MODELS_DIR, 'feature_engineer.pkl'))

    with open(os.path.join(MODELS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Feature importance plot
    _plot_feature_importance(model, feature_cols)
    # ROC / PR curves
    _plot_curves(model, X_test, y_test)

    print(f"\n✅ Model saved to {MODELS_DIR}/")
    return model, fe, metrics


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def explain(model, X_sample: pd.DataFrame, feature_cols: list, n=200):
    """Generate SHAP values and summary plot."""
    try:
        import shap
    except ImportError:
        print("Install shap: pip install shap")
        return None

    X_s = X_sample[feature_cols].fillna(0).head(n)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_s)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_s, plot_type="bar", show=False,
                      max_display=15, color='#e74c3c')
    plt.title("CartShield AI — Feature Impact (SHAP)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(MODELS_DIR, 'shap_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   SHAP summary saved → {out}")
    return shap_values


# ---------------------------------------------------------------------------
# Scoring utility (used by dashboard)
# ---------------------------------------------------------------------------

def score_transactions(df: pd.DataFrame, model=None, fe=None) -> pd.DataFrame:
    """
    Given a raw transactions DataFrame, return it with a 'fraud_score' column.
    """
    if model is None:
        model = joblib.load(os.path.join(MODELS_DIR, 'cartshield_xgb.pkl'))
    if fe is None:
        fe = joblib.load(os.path.join(MODELS_DIR, 'feature_engineer.pkl'))

    df_feat = fe.transform(df.copy())
    feature_cols = fe.get_feature_names()
    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    X = df_feat[feature_cols].fillna(0)
    df['fraud_score'] = model.predict_proba(X)[:, 1]
    df['risk_tier'] = pd.cut(
        df['fraud_score'],
        bins=[-0.001, 0.30, 0.60, 0.80, 1.001],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    return df


# ---------------------------------------------------------------------------
# Plots (internal helpers)
# ---------------------------------------------------------------------------

def _plot_feature_importance(model, feature_cols):
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(imp)))
    imp.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('CartShield AI — Top 20 Feature Importances', fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'feature_importance.png'), dpi=150)
    plt.close()


def _plot_curves(model, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[0], color='#e74c3c')
    axes[0].set_title('ROC Curve', fontweight='bold')
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[1], color='#2980b9')
    axes[1].set_title('Precision-Recall Curve', fontweight='bold')
    plt.suptitle('CartShield AI — Model Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, 'model_curves.png'), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    train()
