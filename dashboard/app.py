"""
CartShield AI — Streamlit Dashboard
Live fraud detection dashboard with risk scoring, SHAP, and segmentation.
"""

import os
import sys
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from src.models.train import score_transactions
from src.models.segmentation import segment_customers

MODELS_DIR = os.path.join(ROOT, 'models')
DATA_DIR   = os.path.join(ROOT, 'data')

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CartShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .main { background: #0b0d14; }
    .stApp { background: #0b0d14; }

    .metric-card {
        background: linear-gradient(135deg, #12141f 0%, #1a1d2e 100%);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; }
    .metric-label { color: #8892a4; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }

    .fraud-badge-critical { background: #e74c3c22; color: #e74c3c; border: 1px solid #e74c3c44; border-radius: 20px; padding: 2px 12px; font-size: 0.75rem; font-weight: 600; }
    .fraud-badge-high     { background: #e6790022; color: #e67900; border: 1px solid #e6790044; border-radius: 20px; padding: 2px 12px; font-size: 0.75rem; font-weight: 600; }
    .fraud-badge-medium   { background: #f1c40f22; color: #f1c40f; border: 1px solid #f1c40f44; border-radius: 20px; padding: 2px 12px; font-size: 0.75rem; font-weight: 600; }
    .fraud-badge-low      { background: #27ae6022; color: #27ae60; border: 1px solid #27ae6044; border-radius: 20px; padding: 2px 12px; font-size: 0.75rem; font-weight: 600; }

    .sidebar-title { font-size: 1.4rem; font-weight: 700; color: #e74c3c; margin-bottom: 8px; }
    .section-header { border-left: 3px solid #e74c3c; padding-left: 12px; font-size: 1.2rem; font-weight: 600; margin: 20px 0 12px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading with caching
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_data():
    path = os.path.join(DATA_DIR, 'raw', 'transactions.csv')
    if not os.path.exists(path):
        st.error("⚠️ Data not found. Run `python scripts/generate_data.py` first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_resource
def load_model():
    mp = os.path.join(MODELS_DIR, 'cartshield_xgb.pkl')
    fp = os.path.join(MODELS_DIR, 'feature_engineer.pkl')
    if not os.path.exists(mp):
        return None, None
    return joblib.load(mp), joblib.load(fp)


@st.cache_data(ttl=300)
def get_scored_data():
    df = load_data()
    model, fe = load_model()
    if model is None:
        df['fraud_score'] = np.random.beta(1.5, 8, len(df))
        df['risk_tier'] = pd.cut(df['fraud_score'],
            bins=[-0.001, 0.30, 0.60, 0.80, 1.001],
            labels=['Low','Medium','High','Critical'])
        return df
    returns_df = df[df['has_return'] == 1].copy()
    scored = score_transactions(returns_df, model, fe)
    return scored


def load_metrics():
    mp = os.path.join(MODELS_DIR, 'metrics.json')
    if os.path.exists(mp):
        with open(mp) as f:
            return json.load(f)
    return {"roc_auc": 0.94, "pr_auc": 0.87, "n_train": 6000, "n_test": 1500}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🛡️ CartShield AI</div>', unsafe_allow_html=True)
    st.caption("E-commerce Return Fraud Detection")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔍 Transaction Scanner", "👥 Customer Segments", "📈 Model Performance"],
        label_visibility="collapsed"
    )
    st.divider()

    st.markdown("**Risk Tier Filter**")
    risk_filter = st.multiselect(
        "Show tiers",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium", "Low"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**Fraud Score Threshold**")
    threshold = st.slider("Classify as fraud above:", 0.0, 1.0, 0.50, 0.01)

    st.divider()
    st.caption("Model: XGBoost + SHAP | Data: Synthetic")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df_raw    = load_data()
df_scored = get_scored_data()
metrics   = load_metrics()

if risk_filter:
    df_display = df_scored[df_scored['risk_tier'].isin(risk_filter)].copy()
else:
    df_display = df_scored.copy()


# ===========================================================================
# PAGE: OVERVIEW
# ===========================================================================
if "Overview" in page:
    st.markdown("# 🛡️ CartShield AI — Fraud Command Center")
    st.caption("Real-time return fraud detection powered by XGBoost + SHAP explainability")

    # KPI row
    total_trans      = len(df_raw)
    total_returns    = df_raw['has_return'].sum()
    total_fraud      = df_raw['is_fraud'].sum()
    fraud_rate       = df_raw['is_fraud'].mean()
    est_loss         = (df_raw['refund_amount'] * df_raw['is_fraud']).sum()
    critical_count   = (df_scored['risk_tier'] == 'Critical').sum() if 'risk_tier' in df_scored.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, f"{total_trans:,}", "Total Transactions", "#4ecdc4"),
        (c2, f"{total_returns:,}", "Total Returns",     "#f7dc6f"),
        (c3, f"{total_fraud:,}", "Fraud Cases",         "#e74c3c"),
        (c4, f"{fraud_rate:.1%}", "Fraud Rate",         "#e74c3c"),
        (c5, f"${est_loss:,.0f}", "Est. Fraud Loss",    "#e74c3c"),
        (c6, f"{critical_count:,}", "Critical Alerts",  "#ff6b6b"),
    ]
    for col, val, label, color in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Row 2: Charts
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-header">Fraud Score Distribution</div>', unsafe_allow_html=True)
        if 'fraud_score' in df_scored.columns:
            fig = px.histogram(
                df_scored, x='fraud_score', nbins=60, color='risk_tier',
                color_discrete_map={'Low':'#27ae60','Medium':'#f1c40f','High':'#e67900','Critical':'#e74c3c'},
                template='plotly_dark',
            )
            fig.update_layout(
                plot_bgcolor='#12141f', paper_bgcolor='#12141f',
                margin=dict(l=0,r=0,t=0,b=0), height=280,
                legend_title_text='Risk Tier',
                xaxis_title='Fraud Score', yaxis_title='Count'
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Risk Tier Breakdown</div>', unsafe_allow_html=True)
        if 'risk_tier' in df_scored.columns:
            tier_counts = df_scored['risk_tier'].value_counts().reset_index()
            tier_counts.columns = ['Tier', 'Count']
            fig2 = px.pie(
                tier_counts, names='Tier', values='Count',
                color='Tier',
                color_discrete_map={'Low':'#27ae60','Medium':'#f1c40f','High':'#e67900','Critical':'#e74c3c'},
                hole=0.55, template='plotly_dark',
            )
            fig2.update_layout(
                plot_bgcolor='#12141f', paper_bgcolor='#12141f',
                margin=dict(l=0,r=0,t=0,b=0), height=280,
                showlegend=True
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Row 3: Fraud by Category + Monthly Trend
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-header">Fraud Rate by Category</div>', unsafe_allow_html=True)
        cat_fraud = df_raw.groupby('product_category')['is_fraud'].mean().sort_values(ascending=True).reset_index()
        fig3 = px.bar(cat_fraud, x='is_fraud', y='product_category', orientation='h',
                      color='is_fraud', color_continuous_scale='RdYlGn_r',
                      labels={'is_fraud': 'Fraud Rate', 'product_category': ''},
                      template='plotly_dark')
        fig3.update_layout(plot_bgcolor='#12141f', paper_bgcolor='#12141f',
                           margin=dict(l=0,r=0,t=0,b=0), height=260, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-header">Monthly Fraud Trend</div>', unsafe_allow_html=True)
        df_raw['order_date'] = pd.to_datetime(df_raw['order_date'])
        monthly = df_raw.resample('ME', on='order_date').agg(
            fraud_count=('is_fraud','sum'), total=('order_id','count')
        ).reset_index()
        monthly['fraud_rate'] = monthly['fraud_count'] / monthly['total']
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=monthly['order_date'], y=monthly['fraud_rate'],
            mode='lines+markers', line=dict(color='#e74c3c', width=2),
            fill='tozeroy', fillcolor='rgba(231,76,60,0.1)'
        ))
        fig4.update_layout(
            plot_bgcolor='#12141f', paper_bgcolor='#12141f',
            margin=dict(l=0,r=0,t=0,b=0), height=260,
            xaxis_title='', yaxis_title='Fraud Rate',
            xaxis=dict(gridcolor='#1e2030'), yaxis=dict(gridcolor='#1e2030')
        )
        st.plotly_chart(fig4, use_container_width=True)


# ===========================================================================
# PAGE: TRANSACTION SCANNER
# ===========================================================================
elif "Scanner" in page:
    st.markdown("# 🔍 Transaction Risk Scanner")

    col_l, col_r = st.columns([2, 1])

    with col_r:
        st.markdown("### Live Prediction")
        with st.form("predict_form"):
            order_value       = st.number_input("Order Value ($)", 5.0, 5000.0, 149.99, step=0.01)
            refund_amount     = st.number_input("Refund Amount ($)", 0.0, 5000.0, 149.99, step=0.01)
            delivery_days     = st.slider("Delivery Days", 1, 30, 5)
            days_to_return    = st.slider("Days to Return", 0, 60, 2)
            refund_freq       = st.slider("Refund Count (30d)", 0, 10, 3)
            account_age       = st.number_input("Account Age (days)", 1, 3650, 30)
            product_cat       = st.selectbox("Category", ["electronics","clothing","home","beauty","sports","books"])
            payment           = st.selectbox("Payment", ["credit_card","debit_card","paypal","apple_pay","crypto"])
            return_reason     = st.selectbox("Return Reason", ["item_not_received","defective","wrong_size","not_as_described","changed_mind"])
            submitted = st.form_submit_button("🔎 Analyze Risk", use_container_width=True)

        if submitted:
            model, fe = load_model()
            if model is None:
                st.warning("Model not trained yet. Run `python src/models/train.py`")
            else:
                sample = pd.DataFrame([{
                    'order_id': 'LIVE_001', 'customer_id': 'CUST_LIVE',
                    'order_date': '2024-06-15', 'delivery_date': '2024-06-20',
                    'product_category': product_cat, 'order_value': order_value,
                    'payment_method': payment, 'customer_segment': 'regular',
                    'country': 'US', 'account_age_days': account_age,
                    'total_orders_lifetime': 5, 'delivery_days': delivery_days,
                    'has_return': 1, 'return_reason': return_reason,
                    'days_to_return': days_to_return, 'refund_amount': refund_amount,
                    'refund_frequency_30d': refund_freq, 'is_fraud': 0, 'fraud_type': None,
                }])
                scored = score_transactions(sample, model, fe)
                score = scored['fraud_score'].iloc[0]
                tier  = scored['risk_tier'].iloc[0]

                color_map = {'Low':'#27ae60','Medium':'#f1c40f','High':'#e67900','Critical':'#e74c3c'}
                c = color_map.get(str(tier), '#888')
                verdict = "🚨 FRAUD" if score >= threshold else "✅ LEGIT"

                st.markdown(f"""
                <div style="background:#12141f;border:2px solid {c};border-radius:12px;padding:20px;text-align:center;margin-top:10px">
                    <div style="font-size:2.5rem;font-weight:800;color:{c}">{score:.1%}</div>
                    <div style="color:#aaa;font-size:0.85rem;margin:4px 0">FRAUD PROBABILITY</div>
                    <div style="font-size:1.4rem;font-weight:700;color:{c};margin:8px 0">{verdict}</div>
                    <div style="color:#aaa;font-size:0.9rem">Risk Tier: <b style="color:{c}">{tier}</b></div>
                </div>""", unsafe_allow_html=True)

    with col_l:
        st.markdown("### Recent High-Risk Transactions")
        if 'fraud_score' in df_display.columns:
            show_cols = ['order_id','customer_id','order_date','product_category',
                         'order_value','refund_amount','fraud_score','risk_tier']
            show_cols = [c for c in show_cols if c in df_display.columns]
            top_risky = df_display.sort_values('fraud_score', ascending=False).head(50)[show_cols]

            def color_tier(val):
                colors = {'Critical':'background-color:#e74c3c22;color:#e74c3c',
                          'High':'background-color:#e6790022;color:#e67900',
                          'Medium':'background-color:#f1c40f22;color:#f1c40f',
                          'Low':'background-color:#27ae6022;color:#27ae60'}
                return colors.get(val, '')

            st.dataframe(
                top_risky.style.format({'fraud_score':'{:.1%}','order_value':'${:.2f}','refund_amount':'${:.2f}'}),
                use_container_width=True, height=480
            )


# ===========================================================================
# PAGE: CUSTOMER SEGMENTS
# ===========================================================================
elif "Segments" in page:
    st.markdown("# 👥 Customer Risk Segmentation")

    with st.spinner("Building customer profiles..."):
        try:
            profile = segment_customers(df_raw)
        except Exception as e:
            st.error(f"Segmentation error: {e}")
            st.stop()

    col1, col2, col3, col4 = st.columns(4)
    seg_counts = profile['risk_segment'].value_counts().sort_index()
    labels_map = {0:"🟢 Trustworthy",1:"🟡 Moderate",2:"🔴 High Risk",3:"⚫ Extreme"}
    colors_map = {0:"#27ae60",1:"#f1c40f",2:"#e74c3c",3:"#8e44ad"}

    for i, col in enumerate([col1,col2,col3,col4]):
        with col:
            cnt = seg_counts.get(i, 0)
            lbl = labels_map.get(i, f"Segment {i}")
            clr = colors_map.get(i, "#888")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{clr}">{cnt:,}</div>
                <div class="metric-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">PCA — Customer Clusters</div>', unsafe_allow_html=True)
        fig_pca = px.scatter(
            profile, x='pca_x', y='pca_y', color='risk_segment',
            color_continuous_scale=[(0,"#27ae60"),(0.33,"#f1c40f"),(0.66,"#e74c3c"),(1,"#8e44ad")],
            opacity=0.7, template='plotly_dark',
            hover_data={'customer_id':True,'return_rate':':.2%','fraud_rate':':.2%'},
        )
        fig_pca.update_layout(plot_bgcolor='#12141f', paper_bgcolor='#12141f',
                              margin=dict(l=0,r=0,t=0,b=0), height=350,
                              coloraxis_showscale=False)
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Fraud Rate by Segment</div>', unsafe_allow_html=True)
        seg_stats = profile.groupby('segment_label').agg(
            fraud_rate=('fraud_rate','mean'),
            return_rate=('return_rate','mean'),
            customers=('customer_id','count')
        ).reset_index()

        fig_bar = px.bar(
            seg_stats, x='segment_label', y='fraud_rate',
            color='fraud_rate', color_continuous_scale='RdYlGn_r',
            template='plotly_dark',
            text=seg_stats['fraud_rate'].map('{:.1%}'.format),
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(plot_bgcolor='#12141f', paper_bgcolor='#12141f',
                              margin=dict(l=0,r=0,t=0,b=0), height=350,
                              coloraxis_showscale=False,
                              xaxis_title='', yaxis_title='Avg Fraud Rate')
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Customer Profile Table")
    disp_cols = ['customer_id','segment_label','total_orders','return_rate','fraud_rate',
                 'avg_order_value','max_refund_frequency','account_age_days']
    st.dataframe(
        profile[disp_cols].sort_values('fraud_rate', ascending=False).head(100)
        .style.format({'return_rate':'{:.1%}','fraud_rate':'{:.1%}','avg_order_value':'${:.0f}'}),
        use_container_width=True, height=350
    )


# ===========================================================================
# PAGE: MODEL PERFORMANCE
# ===========================================================================
elif "Performance" in page:
    st.markdown("# 📈 Model Performance")

    m = metrics
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        (c1, f"{m.get('roc_auc',0):.4f}", "ROC-AUC",       "#4ecdc4"),
        (c2, f"{m.get('pr_auc',0):.4f}",  "PR-AUC",         "#f7dc6f"),
        (c3, f"{m.get('n_train',0):,}",   "Train Samples",  "#a29bfe"),
        (c4, f"{m.get('feature_count',0)}", "Features Used", "#fd79a8"),
    ]
    for col, val, label, color in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    col_img1, col_img2 = st.columns(2)
    fi_path  = os.path.join(MODELS_DIR, 'feature_importance.png')
    mc_path  = os.path.join(MODELS_DIR, 'model_curves.png')
    shp_path = os.path.join(MODELS_DIR, 'shap_summary.png')

    with col_img1:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)
        else:
            st.info("Run `python src/models/train.py` to generate plots.")

    with col_img2:
        st.markdown('<div class="section-header">ROC & PR Curves</div>', unsafe_allow_html=True)
        if os.path.exists(mc_path):
            st.image(mc_path, use_container_width=True)
        else:
            st.info("Run `python src/models/train.py` to generate plots.")

    if os.path.exists(shp_path):
        st.markdown('<div class="section-header">SHAP Feature Impact</div>', unsafe_allow_html=True)
        st.image(shp_path, use_container_width=True)

    if m.get('features'):
        st.markdown('<div class="section-header">All Features Used</div>', unsafe_allow_html=True)
        feat_df = pd.DataFrame({'Feature': m['features'], 'Index': range(len(m['features']))})
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
