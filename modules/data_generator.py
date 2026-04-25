import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

PRODUCTS = [
    "Solar Panel 400W",
    "Solar Panel 550W",
    "Solar Inverter 5kW",
    "Solar Battery 10kWh",
    "Solar Charge Controller",
]

DISTRICTS = ["North District", "South District", "East District"]
CUSTOMERS = [f"Customer_{i:03d}" for i in range(1, 21)]

BASE_PRICES = {
    "Solar Panel 400W": 350,
    "Solar Panel 550W": 480,
    "Solar Inverter 5kW": 1200,
    "Solar Battery 10kWh": 3500,
    "Solar Charge Controller": 180,
}


def generate_sample_data(seed: int = 42) -> pd.DataFrame:
    try:
        np.random.seed(seed)
        random.seed(seed)
        records = []
        base_date = datetime(2025, 9, 1)

        for month_offset in range(6):
            month_date = base_date + timedelta(days=30 * month_offset)
            trend_mult = 1 + 0.05 * month_offset

            for _ in range(np.random.randint(30, 60)):
                product = random.choice(PRODUCTS)
                district = random.choice(DISTRICTS)
                customer = random.choice(CUSTOMERS)
                qty = int(np.random.randint(1, 10))
                base_price = BASE_PRICES[product]
                price = base_price * trend_mult * np.random.uniform(0.9, 1.1)
                revenue = qty * price
                sale_date = month_date + timedelta(days=int(np.random.randint(0, 28)))

                records.append({
                    "date": sale_date.strftime("%Y-%m-%d"),
                    "month": month_date.strftime("%Y-%m"),
                    "product": product,
                    "district": district,
                    "customer": customer,
                    "quantity": qty,
                    "unit_price": round(float(price), 2),
                    "revenue": round(float(revenue), 2),
                    "stock_level": int(np.random.randint(5, 100)),
                })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    except Exception as e:
        return pd.DataFrame({
            "date": pd.date_range("2025-09-01", periods=30, freq="D"),
            "month": ["2025-09"] * 30,
            "product": ["Solar Panel 400W"] * 30,
            "district": ["North District"] * 30,
            "customer": [f"Customer_{i:03d}" for i in range(1, 31)],
            "quantity": [1] * 30,
            "unit_price": [350.0] * 30,
            "revenue": [350.0] * 30,
            "stock_level": [50] * 30,
        })
