"""
CartShield AI — Feature Engineering Pipeline
Transforms raw transaction data into model-ready features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Builds rich behavioral and transactional features for fraud detection.
    """

    def __init__(self):
        self.label_encoders = {}
        self.cat_cols = ['product_category', 'payment_method', 'customer_segment', 'country', 'return_reason']
        self.feature_names_ = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._temporal_features(df)
        df = self._behavioral_features(df)
        df = self._risk_flag_features(df)
        df = self._encode_categoricals(df, fit=True)
        self.feature_names_ = self._get_feature_cols(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._temporal_features(df)
        df = self._behavioral_features(df)
        df = self._risk_flag_features(df)
        df = self._encode_categoricals(df, fit=False)
        return df

    # ------------------------------------------------------------------
    def _temporal_features(self, df):
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['delivery_date'] = pd.to_datetime(df['delivery_date'])

        df['order_month'] = df['order_date'].dt.month
        df['order_dayofweek'] = df['order_date'].dt.dayofweek
        df['order_quarter'] = df['order_date'].dt.quarter
        df['is_weekend_order'] = (df['order_dayofweek'] >= 5).astype(int)
        df['is_holiday_season'] = df['order_month'].isin([11, 12, 1]).astype(int)

        df['delivery_days'] = df['delivery_days'].fillna(df['delivery_days'].median())
        df['days_to_return'] = df['days_to_return'].fillna(999)

        return df

    def _behavioral_features(self, df):
        # Refund ratio
        df['refund_to_order_ratio'] = np.where(
            df['order_value'] > 0,
            df['refund_amount'] / df['order_value'],
            0
        ).clip(0, 2)

        # Account age buckets
        df['account_age_bucket'] = pd.cut(
            df['account_age_days'],
            bins=[0, 30, 180, 365, 1095, 9999],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

        # Order value z-score approximation (capped)
        df['order_value_log'] = np.log1p(df['order_value'])
        df['is_high_value_order'] = (df['order_value'] > 300).astype(int)
        df['is_very_high_value_order'] = (df['order_value'] > 1000).astype(int)

        # Refund frequency risk tiers
        df['refund_freq_tier'] = pd.cut(
            df['refund_frequency_30d'],
            bins=[-1, 0, 1, 2, 3, 999],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)

        return df

    def _risk_flag_features(self, df):
        # Fast return after delivery — suspicious
        df['is_fast_return'] = (df['days_to_return'] < 3).astype(int)

        # Return within same day of delivery
        df['same_day_return'] = (df['days_to_return'] <= 1).astype(int)

        # High refund + new account
        df['new_account_high_refund'] = (
            (df['account_age_days'] < 90) &
            (df['refund_frequency_30d'] >= 2)
        ).astype(int)

        # Crypto + high value (rare but risky)
        df['crypto_high_value'] = (
            (df['payment_method'] == 'crypto') &
            (df['order_value'] > 200)
        ).astype(int)

        # Repeated returns in electronics
        df['electronics_repeat_return'] = (
            (df['product_category'] == 'electronics') &
            (df['refund_frequency_30d'] >= 2)
        ).astype(int)

        # Refund amount > order value (overpayment fraud)
        df['refund_exceeds_order'] = (
            df['refund_amount'] > df['order_value'] * 1.05
        ).astype(int)

        # Reason: item_not_received  (common fraud vector)
        df['reason_not_received'] = (
            df['return_reason'] == 'item_not_received'
        ).astype(int)

        # Composite risk score (heuristic, pre-ML)
        df['heuristic_risk_score'] = (
            df['refund_freq_tier'] * 2 +
            df['is_fast_return'] * 2 +
            df['new_account_high_refund'] * 3 +
            df['refund_exceeds_order'] * 3 +
            df['reason_not_received'] * 2 +
            df['is_high_value_order'] * 1
        )

        return df

    def _encode_categoricals(self, df, fit=True):
        for col in self.cat_cols:
            if col not in df.columns:
                continue
            df[col] = df[col].fillna('unknown')
            if fit:
                le = LabelEncoder()
                df[col + '_enc'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    df[col + '_enc'] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        return df

    def _get_feature_cols(self, df):
        exclude = ['order_id', 'customer_id', 'order_date', 'delivery_date',
                   'is_fraud', 'fraud_type', 'return_reason', 'payment_method',
                   'product_category', 'customer_segment', 'country']
        return [c for c in df.columns if c not in exclude]

    def get_feature_names(self):
        return self.feature_names_


def load_and_prepare(transactions_path: str, returns_only: bool = False) -> pd.DataFrame:
    df = pd.read_csv(transactions_path)
    if returns_only:
        df = df[df['has_return'] == 1].copy()
    fe = FeatureEngineer()
    df_feat = fe.fit_transform(df)
    return df_feat, fe
