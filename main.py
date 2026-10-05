"""
Smart Sales Analyst - run the whole pipeline.

    python main.py                          # uses data/sales_data.xlsx
    python main.py --input my_sales.xlsx    # your own sheet (same column names)

Steps:
  1. Excel        -> load the sales sheet
  2. scikit-learn -> clean, segment customers (KMeans), flag anomalies (IsolationForest)
  3. PyTorch      -> forecast next month's sales per region
  4. (chat.py)    -> LangChain chatbot uses the saved results as tools
  5. Excel        -> final report workbook with charts
"""
import argparse
import json
import time
import warnings

import pandas as pd

import config
from src.data_prep import clean_sales, load_sales
from src.forecast import run_forecast
from src.report import build_report
from src.segmentation import detect_anomalies, segment_customers

warnings.filterwarnings("ignore", category=UserWarning)


def step(msg):
    print(f"\n==> {msg}")
    return time.time()


def key_findings(clean, rfm, seg_summary, anomalies, fc):
    """A few plain-English headlines for the Summary sheet."""
    normal = clean[~clean.IsAnomaly]
    months = sorted(normal.Month.unique())
    recent, prev = months[-3:], months[-6:-3]
    by_region = pd.DataFrame({
        "recent": normal[normal.Month.isin(recent)].groupby("Region").Sales.sum(),
        "prev": normal[normal.Month.isin(prev)].groupby("Region").Sales.sum()})
    by_region["chg"] = (by_region.recent - by_region.prev) / by_region.prev
    worst, best = by_region.chg.idxmin(), by_region.chg.idxmax()

    w = normal[normal.Region == worst]
    lost = set(w[w.Month.isin(prev)].CustomerName) - set(w[w.Month.isin(recent)].CustomerName)
    lost_value = w[w.Month.isin(prev) & w.CustomerName.isin(lost)].Sales.sum()
    drop = by_region.loc[worst, "prev"] - by_region.loc[worst, "recent"]
    top_lost = (w[w.Month.isin(prev) & w.CustomerName.isin(lost)]
                .groupby("CustomerName").Sales.sum().sort_values(ascending=False).head(2).index.tolist())

    total = fc["forecast"].set_index("Region").loc["TOTAL"]
    champ = seg_summary[seg_summary.Segment == "Champions"].iloc[0]
    champ_share = champ.TotalSales / seg_summary.TotalSales.sum()
    risk = rfm[rfm.ChurnRisk]
    findings = [
        f"{worst} is the weakest region: sales {by_region.loc[worst, 'chg']:+.0%} in the last 3 months vs the "
        f"3 before. {len(lost)} customers stopped buying there"
        + (f" (incl. {', '.join(top_lost)}), explaining {lost_value / drop:.0%} of the drop." if drop > 0 and top_lost else "."),
        f"{best} is the strongest region ({by_region.loc[best, 'chg']:+.0%}).",
        f"{fc['info']['forecast_month']} forecast: ₹{total.NextMonthForecast / 1e7:.2f} Cr "
        f"({total['ChangeVsLastMonth%']:+.1f}% vs last month) using {fc['info']['backend']}.",
        f"Just {int(champ.Customers)} 'Champions' customers bring in {champ_share:.0%} of all sales "
        f"(₹{champ.TotalSales / 1e7:.2f} Cr). Losing even one hurts.",
        f"{len(risk)} high-value customers are at churn risk (no order in 60+ days)"
        + (f": {', '.join(risk.CustomerName.head(3))}." if len(risk) else "."),
        f"{len(anomalies)} unusual one-off orders were flagged and excluded from trends and the forecast.",
    ]
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(config.RAW_DATA_PATH))
    args = parser.parse_args()
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    step(f"1/5  Loading {args.input}")
    raw = load_sales(args.input)
    print(f"     {len(raw):,} rows, {raw.shape[1]} columns")

    step("2/5  Cleaning, segmenting customers, detecting anomalies (scikit-learn)")
    clean, cleaning_log = clean_sales(raw)
    for entry in cleaning_log:
        print(f"     {entry['Step']:<24} {entry['Rows affected']:>6}  {entry['Detail']}")
    rfm, seg_summary, seg_info = segment_customers(clean, k_range=config.K_RANGE)
    print(f"     KMeans picked k={seg_info['best_k']}  silhouette={seg_info['silhouette_scores']}")
    print(seg_summary[["Segment", "Customers", "TotalSales"]].to_string(index=False))
    anomalies, is_anomaly = detect_anomalies(clean)
    clean["IsAnomaly"] = is_anomaly.values
    print(f"     {len(anomalies)} unusual orders flagged")

    step("3/5  Forecasting next month (PyTorch)")
    fc = run_forecast(clean, exclude_mask=clean["IsAnomaly"], window=config.WINDOW,
                      epochs=config.EPOCHS, lr=config.LEARNING_RATE, holdout=config.HOLDOUT_MONTHS)
    print(f"     Backend: {fc['info']['backend']}")
    print(f"     Back-test MAPE: model {fc['info']['overall_accuracy']['ModelMAPE%']}% "
          f"vs naive {fc['info']['overall_accuracy']['NaiveMAPE%']}%")
    print(fc["forecast"].to_string(index=False))

    step("4/5  Saving results for the chatbot")
    a = config.ARTIFACTS_DIR
    clean.to_pickle(a / "clean_sales.pkl")
    rfm.to_pickle(a / "customers_rfm.pkl")
    seg_summary.to_pickle(a / "segment_summary.pkl")
    anomalies.to_pickle(a / "anomalies.pkl")
    fc["forecast"].to_pickle(a / "forecast.pkl")
    fc["accuracy"].to_pickle(a / "forecast_accuracy.pkl")
    (a / "run_info.json").write_text(json.dumps({"segmentation": seg_info, "forecast": fc["info"]}, indent=2))
    print(f"     Saved to {a}")

    step("5/5  Building Excel report")
    findings = key_findings(clean, rfm, seg_summary, anomalies, fc)
    build_report(config.REPORT_PATH, clean, cleaning_log, rfm, seg_summary, seg_info, anomalies, fc, findings)
    print(f"     Report: {config.REPORT_PATH}")
    print("\nKey findings:\n" + "\n".join(f"  - {f}" for f in findings))
    print(f"\nDone in {time.time() - t0:.1f}s. Now chat with it:  python chat.py")


if __name__ == "__main__":
    main()
