"""
CartShield AI — Synthetic E-commerce Return Fraud Dataset Generator
Generates realistic transaction + return data with fraud labels.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

N_CUSTOMERS = 2000
N_TRANSACTIONS = 50000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')


def generate_customers(n=N_CUSTOMERS):
    customer_ids = [f"CUST_{str(i).zfill(5)}" for i in range(1, n + 1)]

    # Fraud propensity hidden variable
    fraud_propensity = np.random.beta(1.5, 8, size=n)  # Most customers are honest

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "account_age_days": np.random.randint(1, 3650, n),
        "total_orders_lifetime": np.random.poisson(12, n).clip(1),
        "fraud_propensity": fraud_propensity,  # will be dropped before modeling
        "customer_segment": np.random.choice(
            ["budget", "regular", "premium", "vip"], n,
            p=[0.30, 0.40, 0.20, 0.10]
        ),
        "country": np.random.choice(
            ["US", "UK", "CA", "AU", "IN", "DE", "FR"], n,
            p=[0.40, 0.15, 0.10, 0.08, 0.12, 0.08, 0.07]
        ),
    })
    return customers


def generate_transactions(customers, n=N_TRANSACTIONS):
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)

    records = []

    for i in range(n):
        cust = customers.sample(1).iloc[0]
        cust_id = cust["customer_id"]
        fraud_p = cust["fraud_propensity"]

        order_date = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )

        product_category = np.random.choice(
            ["electronics", "clothing", "home", "beauty", "sports", "books"],
            p=[0.20, 0.30, 0.15, 0.15, 0.12, 0.08]
        )

        order_value = {
            "electronics": np.random.lognormal(5.0, 0.8),
            "clothing": np.random.lognormal(4.0, 0.6),
            "home": np.random.lognormal(4.2, 0.7),
            "beauty": np.random.lognormal(3.5, 0.5),
            "sports": np.random.lognormal(4.1, 0.6),
            "books": np.random.lognormal(3.0, 0.4),
        }[product_category]

        order_value = round(max(5.0, order_value), 2)

        delivery_days = np.random.randint(1, 14)
        delivery_date = order_date + timedelta(days=delivery_days)

        # Was there a return?
        base_return_prob = 0.15
        return_prob = base_return_prob + fraud_p * 0.5
        has_return = np.random.random() < return_prob

        # Fraud logic
        is_fraud = False
        fraud_type = None
        refund_amount = 0.0
        return_reason = None
        days_to_return = None

        if has_return:
            return_reasons_legit = ["wrong_size", "defective", "not_as_described", "changed_mind", "gift_return"]
            return_reasons_fraud = ["item_not_received", "not_as_described", "defective", "wrong_item"]

            if fraud_p > 0.6 and np.random.random() < 0.7:
                is_fraud = True
                fraud_type = np.random.choice([
                    "wardrobing", "false_item_not_received",
                    "price_arbitrage", "counterfeit_return", "duplicate_refund"
                ])
                return_reason = np.random.choice(return_reasons_fraud)
                refund_amount = round(order_value * np.random.uniform(0.9, 1.1), 2)
                days_to_return = np.random.randint(1, 5)  # Fraudsters return quickly
            else:
                return_reason = np.random.choice(return_reasons_legit)
                refund_amount = round(order_value * np.random.uniform(0.5, 1.0), 2)
                days_to_return = np.random.randint(3, 30)

        refund_frequency_30d = round(np.random.poisson(fraud_p * 3), 0)
        payment_method = np.random.choice(
            ["credit_card", "debit_card", "paypal", "apple_pay", "crypto"],
            p=[0.45, 0.25, 0.18, 0.10, 0.02]
        )

        records.append({
            "order_id": f"ORD_{str(i + 1).zfill(7)}",
            "customer_id": cust_id,
            "order_date": order_date.strftime("%Y-%m-%d"),
            "delivery_date": delivery_date.strftime("%Y-%m-%d"),
            "product_category": product_category,
            "order_value": order_value,
            "payment_method": payment_method,
            "customer_segment": cust["customer_segment"],
            "country": cust["country"],
            "account_age_days": int(cust["account_age_days"]),
            "total_orders_lifetime": int(cust["total_orders_lifetime"]),
            "delivery_days": delivery_days,
            "has_return": int(has_return),
            "return_reason": return_reason,
            "days_to_return": days_to_return,
            "refund_amount": refund_amount if has_return else 0.0,
            "refund_frequency_30d": int(refund_frequency_30d),
            "is_fraud": int(is_fraud),
            "fraud_type": fraud_type,
        })

    return pd.DataFrame(records)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating customers...")
    customers = generate_customers()

    print("Generating transactions...")
    transactions = generate_transactions(customers)

    # Save
    customers_out = customers.drop(columns=["fraud_propensity"])
    customers_out.to_csv(os.path.join(OUTPUT_DIR, "customers.csv"), index=False)
    transactions.to_csv(os.path.join(OUTPUT_DIR, "transactions.csv"), index=False)

    # Sample for quick testing
    sample = transactions.sample(n=500, random_state=42)
    sample.to_csv(os.path.join(OUTPUT_DIR, '..', 'sample', 'sample_transactions.csv'), index=False)

    print(f"\n✅ Generated {len(customers)} customers and {len(transactions)} transactions")
    print(f"   Fraud rate: {transactions['is_fraud'].mean():.2%}")
    print(f"   Return rate: {transactions['has_return'].mean():.2%}")
    print(f"   Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
