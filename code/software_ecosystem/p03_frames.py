import numpy as np
import pandas as pd
import polars as pl
import duckdb

rng = np.random.default_rng(0)
n = 100_000
df = pd.DataFrame({
    "date": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 180, n), unit="D"),
    "site": rng.choice(["Culham", "Sunbury", "Reading"], n),
    "power_kw": rng.gamma(4.0, 25.0, n),
})

# pandas: monthly mean per site, then month-on-month change
m = (df.assign(month=df["date"].dt.to_period("M"))
       .groupby(["site", "month"], as_index=False)["power_kw"].mean())
m["mom_pct"] = m.groupby("site")["power_kw"].pct_change().mul(100).round(2)
print(m.head(3).to_string(index=False))

# polars: the same query, expression API (lazy -> optimised plan, multithreaded)
p = (pl.from_pandas(df).lazy()
       .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
       .group_by("site", "month").agg(pl.col("power_kw").mean())
       .sort("site", "month")
       .with_columns((pl.col("power_kw").pct_change().over("site") * 100).round(2).alias("mom_pct"))
       .collect())
print(p.head(3))

# DuckDB: plain SQL directly on the pandas DataFrame in memory (window function)
q = duckdb.sql("""
    SELECT site, date_trunc('month', date) AS month, avg(power_kw) AS power_kw,
           round(100 * (avg(power_kw) / lag(avg(power_kw)) OVER (PARTITION BY site ORDER BY date_trunc('month', date)) - 1), 2) AS mom_pct
    FROM df GROUP BY site, month ORDER BY site, month LIMIT 3
""").df()
print(q.to_string(index=False))
