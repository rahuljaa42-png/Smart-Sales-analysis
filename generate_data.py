"""
Step 0: Generate a realistic (and deliberately messy) sales dataset in Excel.

Covers Oct 2024 - Sep 2026 (24 months), 4 regions, 5 categories.
Baked-in stories for the chatbot to discover:
  * North region drops sharply in the last 3 months (lost 2 big customers,
    Electronics collapse, heavier discounting).
  * South grows steadily.
  * A few extreme orders (anomalies) and dirty-data problems.

Run:  python generate_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

from config import RAW_DATA_PATH

rng = np.random.default_rng(42)

REGIONS = ["North", "South", "East", "West"]
PRODUCTS = {
    "Electronics": [("Laptop", 55000), ("Smartphone", 22000), ("Headphones", 3500), ("Monitor", 14000)],
    "Furniture":   [("Office Chair", 8500), ("Study Table", 12000), ("Bookshelf", 6500)],
    "Stationery":  [("Notebook Pack", 450), ("Pen Set", 300), ("Printer Paper", 650)],
    "Appliances":  [("Air Purifier", 16000), ("Microwave", 9500), ("Water Purifier", 13000)],
    "Clothing":    [("Formal Shirt", 1800), ("Jacket", 3800), ("Sneakers", 4200)],
}
FIRST = ["Aarav", "Ishita", "Rohan", "Priya", "Kabir", "Ananya", "Vikram", "Sneha", "Arjun", "Meera",
         "Rahul", "Pooja", "Aditya", "Kavya", "Nikhil", "Riya", "Siddharth", "Tanvi", "Karan", "Neha"]
LAST = ["Traders", "Enterprises", "Retail", "Stores", "Mart", "Distributors", "& Sons", "Supplies"]

# Region-level growth per month (South grows, others flat-ish)
REGION_BASE_ORDERS = {"North": 150, "South": 115, "East": 105, "West": 125}
REGION_GROWTH = {"North": 0.004, "South": 0.020, "East": 0.006, "West": 0.002}


def make_customers(n=158):
    key_names = ["Vikram Distributors", "Meera Enterprises"]
    all_names = [f"{f} {l}" for f in FIRST for l in LAST if f"{f} {l}" not in key_names]
    names = rng.permutation(all_names)[:n]                       # unique names
    customers = []
    for i, name in enumerate(names):
        region = REGIONS[i % 4]
        # a few "key accounts" buy much more often
        weight = rng.choice([1, 1, 1, 2, 3, 8], p=[0.35, 0.2, 0.15, 0.15, 0.1, 0.05])
        customers.append({"CustomerID": f"C{1000 + i}", "CustomerName": str(name),
                          "Region": region, "weight": float(weight)})
    # Two big North key accounts that will churn in the last 3 months
    for j, name in enumerate(key_names):
        customers.append({"CustomerID": f"C{1000 + n + j}", "CustomerName": name,
                          "Region": "North", "weight": 12.0})
    return pd.DataFrame(customers)


def main():
    customers = make_customers()
    churned = set(customers[(customers.Region == "North") & (customers.weight == 12.0)].CustomerID)

    months = pd.date_range("2024-10-01", "2026-09-01", freq="MS")
    rows, order_no = [], 100000
    for m_idx, month in enumerate(months):
        last_3 = m_idx >= len(months) - 3
        season = 1 + 0.15 * np.sin(2 * np.pi * (month.month - 10) / 12)   # festive peak ~Oct/Nov
        for region in REGIONS:
            n_orders = REGION_BASE_ORDERS[region] * (1 + REGION_GROWTH[region]) ** m_idx * season
            if region == "North" and last_3:
                n_orders *= 0.72                                           # fewer orders
            n_orders = int(rng.poisson(n_orders))

            pool = customers[customers.Region == region].copy()
            if region == "North" and last_3:
                pool = pool[~pool.CustomerID.isin(churned)]               # key accounts gone
            probs = pool.weight / pool.weight.sum()

            for _ in range(n_orders):
                cust = pool.iloc[rng.choice(len(pool), p=probs)]
                cat_probs = np.array([0.30, 0.18, 0.22, 0.15, 0.15])
                if region == "North" and last_3:
                    cat_probs = np.array([0.08, 0.20, 0.32, 0.18, 0.22])  # electronics collapse
                category = rng.choice(list(PRODUCTS), p=cat_probs / cat_probs.sum())
                product, price = PRODUCTS[category][rng.integers(len(PRODUCTS[category]))]
                qty = int(rng.integers(2, 9)) if cust.weight >= 8 else int(rng.integers(1, 5))
                unit_price = round(price * rng.uniform(0.92, 1.08), 2)
                discount = round(float(rng.choice([0, 0.05, 0.10, 0.15], p=[0.5, 0.25, 0.15, 0.10])), 2)
                if region == "North" and last_3:
                    discount = round(min(discount + 0.08, 0.30), 2)       # desperate discounting
                day = int(rng.integers(1, month.days_in_month + 1))
                order_no += 1
                rows.append({
                    "OrderID": f"ORD{order_no}",
                    "OrderDate": month.replace(day=day),
                    "CustomerID": cust.CustomerID,
                    "CustomerName": cust.CustomerName,
                    "Region": region,
                    "Category": category,
                    "Product": product,
                    "Quantity": qty,
                    "UnitPrice": unit_price,
                    "Discount": discount,
                })

    df = pd.DataFrame(rows)
    df["Sales"] = (df.Quantity * df.UnitPrice * (1 - df.Discount)).round(2)

    # ---------- inject dirty-data problems (the cleaning step must fix these) ----------
    idx = rng.choice(len(df), size=40, replace=False)
    df.loc[idx[:15], "Region"] = df.loc[idx[:15], "Region"].str.upper()           # "NORTH"
    df.loc[idx[15:25], "Region"] = df.loc[idx[15:25], "Region"].str.lower() + " "  # "north "
    df.loc[idx[25:32], "Discount"] = np.nan                                        # missing discount
    df.loc[idx[32:36], "Quantity"] = -df.loc[idx[32:36], "Quantity"]               # typo negatives
    df.loc[idx[36:40], "Sales"] = np.nan                                           # missing sales
    df = pd.concat([df, df.sample(12, random_state=1)], ignore_index=True)         # duplicates

    # ---------- genuine anomalies (bulk / suspicious orders) ----------
    anomaly_idx = rng.choice(len(df), size=6, replace=False)
    df.loc[anomaly_idx, "Quantity"] = rng.integers(60, 120, size=6)
    df.loc[anomaly_idx, "Sales"] = (df.loc[anomaly_idx, "Quantity"] * df.loc[anomaly_idx, "UnitPrice"]
                                    * (1 - df.loc[anomaly_idx, "Discount"].fillna(0))).round(2)

    df = df.sample(frac=1, random_state=7).reset_index(drop=True)                  # shuffle
    Path(RAW_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(RAW_DATA_PATH, index=False, sheet_name="Sales")
    print(f"Wrote {len(df):,} rows -> {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
