"""Turn Locust CSVs into the README table.  Usage: python locust/summarize.py results_1 results_2 results_4"""
import sys

import pandas as pd

print("| Run | Requests | Failures | Requests/s | Median (ms) | Avg (ms) | 95th pct (ms) |")
print("|---|---|---|---|---|---|---|")
for prefix in sys.argv[1:]:
    df = pd.read_csv(f"{prefix}_stats.csv")
    row = df[df["Name"] == "Aggregated"].iloc[0]
    print(f"| {prefix} | {int(row['Request Count'])} | {int(row['Failure Count'])} | {row['Requests/s']:.1f} | "
          f"{row['Median Response Time']:.0f} | {row['Average Response Time']:.0f} | {row['95%']:.0f} |")
