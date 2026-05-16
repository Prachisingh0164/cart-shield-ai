"""
CartShield AI — Unit Tests
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.feature_engineering import FeatureEngineer


def _make_sample_df(n=100):
    np.random.seed(42)
    return pd.DataFrame({
        'order_id': [f'ORD_{i}' for i in range(n)],
        'customer_id': [f'CUST_{i % 20}' for i in range(n)],
        'order_date': pd.date_range('2023-01-01', periods=n, freq='D').strftime('%Y-%m-%d'),
        'delivery_date': pd.date_range('2023-01-05', periods=n, freq='D').strftime('%Y-%m-%d'),
        'product_category': np.random.choice(['electronics','clothing','home'], n),
        'order_value': np.random.uniform(20, 500, n),
        'payment_method': np.random.choice(['credit_card','paypal','crypto'], n),
        'customer_segment': np.random.choice(['budget','regular','premium'], n),
        'country': np.random.choice(['US','UK','IN'], n),
        'account_age_days': np.random.randint(1, 3000, n),
        'total_orders_lifetime': np.random.randint(1, 50, n),
        'delivery_days': np.random.randint(1, 14, n),
        'has_return': np.random.randint(0, 2, n),
        'return_reason': np.random.choice(['wrong_size','defective','item_not_received', None], n),
        'days_to_return': np.where(np.random.randint(0,2,n), np.random.randint(1,30,n), None),
        'refund_amount': np.random.uniform(0, 400, n),
        'refund_frequency_30d': np.random.randint(0, 5, n),
        'is_fraud': np.random.randint(0, 2, n),
        'fraud_type': None,
    })


class TestFeatureEngineer:
    def setup_method(self):
        self.df = _make_sample_df()
        self.fe = FeatureEngineer()

    def test_fit_transform_returns_df(self):
        result = self.fe.fit_transform(self.df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.df)

    def test_temporal_features_created(self):
        result = self.fe.fit_transform(self.df)
        for col in ['order_month', 'order_dayofweek', 'is_weekend_order', 'is_holiday_season']:
            assert col in result.columns, f"Missing: {col}"

    def test_risk_flag_features_created(self):
        result = self.fe.fit_transform(self.df)
        for col in ['is_fast_return', 'refund_exceeds_order', 'heuristic_risk_score']:
            assert col in result.columns, f"Missing: {col}"

    def test_no_nulls_in_risk_flags(self):
        result = self.fe.fit_transform(self.df)
        risk_cols = ['is_fast_return', 'same_day_return', 'new_account_high_refund',
                     'refund_exceeds_order', 'heuristic_risk_score']
        for col in risk_cols:
            assert result[col].isnull().sum() == 0, f"Nulls in {col}"

    def test_encode_categoricals(self):
        result = self.fe.fit_transform(self.df)
        assert 'product_category_enc' in result.columns
        assert 'payment_method_enc' in result.columns

    def test_feature_names_populated(self):
        self.fe.fit_transform(self.df)
        names = self.fe.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 5

    def test_transform_consistency(self):
        df_train = self.df.iloc[:80]
        df_test  = self.df.iloc[80:]
        self.fe.fit_transform(df_train)
        result = self.fe.transform(df_test)
        assert len(result) == 20

    def test_refund_ratio_clipped(self):
        result = self.fe.fit_transform(self.df)
        assert result['refund_to_order_ratio'].max() <= 2.0
        assert result['refund_to_order_ratio'].min() >= 0.0


class TestDataGenerator:
    def test_generate_customers(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from scripts.generate_data import generate_customers
        customers = generate_customers(n=50)
        assert len(customers) == 50
        assert 'customer_id' in customers.columns
        assert 'fraud_propensity' in customers.columns

    def test_generate_transactions(self):
        from scripts.generate_data import generate_customers, generate_transactions
        customers = generate_customers(n=20)
        transactions = generate_transactions(customers, n=100)
        assert len(transactions) == 100
        assert 'is_fraud' in transactions.columns
        assert transactions['is_fraud'].isin([0, 1]).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
