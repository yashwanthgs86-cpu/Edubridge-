"""EDA + cleaning + features + 1-min candles for trades_orderbook_STUDENT_dirty.csv (synthetic BTCUSDT)."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns, os
os.makedirs("charts", exist_ok=True)
log = []                                   # cleaning decision log
def note(msg): log.append(msg); print(msg)

# 1-3. LOAD, INSPECT, FIND PROBLEMS
raw = pd.read_csv("trades_orderbook_STUDENT_dirty.csv", dtype={"price": str, "timestamp": str})
df = raw.copy(); n0 = len(df)
note(f"Loaded {n0} rows x {df.shape[1]} cols")

# 4. CLEAN
d = df.duplicated("trade_id").sum(); df = df.drop_duplicates("trade_id")
note(f"Dropped {d} duplicate trade_ids (exact repeats of the same trade).")

df["symbol"] = "BTCUSDT"
note("Standardized symbol (btcusdt, BTC/USDT, BTC-USDT) -> BTCUSDT: same instrument.")
df["side"] = df["side"].str.lower().str.strip()
note("Standardized side to lowercase buy/sell.")

df["price"] = pd.to_numeric(df["price"].str.replace(r"[$,]", "", regex=True), errors="coerce")
note("Converted price text ($, commas) to float.")

def parse_ts(s):
    s = str(s).strip()
    if s.isdigit():
        return pd.to_datetime(int(s), unit="ms" if len(s) >= 13 else "s")
    if "/" in s:
        return pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    return pd.to_datetime(s, errors="coerce")
df["timestamp"] = df["timestamp"].map(parse_ts)
note(f"Parsed 3 timestamp formats (ISO, DD/MM/YYYY, Unix) to one datetime (UTC assumed). Unparseable: {df.timestamp.isna().sum()}")
df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
note("Sorted by time.")

# Required book fields: cannot be invented -> drop
need = ["bid_qty_1", "ask_price_1", "quantity"]
b = len(df); df = df.dropna(subset=need); note(f"Dropped {b-len(df)} rows missing bid_qty_1 / ask_price_1 / quantity (cannot be reliably recovered).")
b = len(df); df = df[df.quantity > 0]; note(f"Dropped {b-len(df)} rows with zero quantity (no real trade).")
b = len(df); df = df[df.bid_price_1 < df.ask_price_1]; note(f"Dropped {b-len(df)} crossed-book rows (bid >= ask: invalid snapshot).")

# Trade price: in this data buys execute at ask_1, sells at bid_1 -> use to validate/repair
expected = np.where(df.side == "buy", df.ask_price_1, df.bid_price_1)
# 5. OUTLIERS: price <=0, missing, or far from rolling median of mid
mid = (df.bid_price_1 + df.ask_price_1) / 2
rmed = mid.rolling(21, center=True, min_periods=5).median()
bad = df.price.isna() | (df.price <= 0) | ((df.price / rmed - 1).abs() > 0.05)
fix = bad & df.side.notna()
df.loc[fix, "price"] = expected[fix]
note(f"Outliers/missing/negative price: {bad.sum()} rows; repaired {fix.sum()} from the order book (buy->ask_1, sell->bid_1).")
unfixed = bad & ~fix; df = df[~unfixed].reset_index(drop=True)
note(f"Dropped {unfixed.sum()} bad-price rows with no side label (cannot tell whether to use bid or ask).")
# Missing side: infer from price vs book
ms = df.side.isna()
df.loc[ms, "side"] = np.where(df.loc[ms, "price"] >= df.loc[ms, "ask_price_1"], "buy", "sell")
note(f"Inferred {ms.sum()} missing side labels (price at ask -> buy, else sell).")
df = df.dropna(subset=["price", "side"]).reset_index(drop=True)
note(f"Final clean rows: {len(df)}")

# 6. FEATURES (no look-ahead: only current/past data)
df["spread"] = df.ask_price_1 - df.bid_price_1
df["mid_price"] = (df.ask_price_1 + df.bid_price_1) / 2
df["order_imbalance"] = (df.bid_qty_1 - df.ask_qty_1) / (df.bid_qty_1 + df.ask_qty_1)
df["return"] = df.price.pct_change()
df["volatility_20"] = df["return"].rolling(20).std()
df["ma_20"] = df.price.rolling(20).mean()
df.to_csv("trades_orderbook_clean_features.csv", index=False)

# 7. RESAMPLE to 1-min OHLCV
t = df.set_index("timestamp")
c = t.price.resample("1min").ohlc()
c["volume"] = t.quantity.resample("1min").sum()
c["trades"] = t.price.resample("1min").count()
c["avg_imbalance"] = t.order_imbalance.resample("1min").mean()
c = c.dropna(subset=["close"]); c.to_csv("candles_1min.csv")

# 8. VISUALIZE
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[0,0].plot(df.timestamp, df.price); ax[0,0].set_title("Price")
ax[0,1].hist(df.quantity, bins=50); ax[0,1].set_title("Quantity (volume)")
ax[0,2].hist(df.spread, bins=30); ax[0,2].set_title("Spread distribution")
df.side.value_counts().plot.bar(ax=ax[1,0], title="Buy vs sell")
sns.heatmap(df[["price","quantity","spread","mid_price","order_imbalance","return","volatility_20"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax[1,1])
ax[1,2].plot(c.index, c.close); ax[1,2].set_title("1-min close")
plt.tight_layout(); plt.savefig("charts/eda_overview.png", dpi=110)

# 9. TARGET + TIME SPLIT
c["target_up_5m"] = (c.close.shift(-5) > c.close).astype(int)
c = c.iloc[:-5]; cut = int(len(c) * 0.7)
c.iloc[:cut].to_csv("train_candles.csv"); c.iloc[cut:].to_csv("test_candles.csv")
note(f"Time split: train {cut} candles / test {len(c)-cut} candles (no shuffling).")
open("cleaning_log.txt", "w").write("\n".join(log))
print(df.describe().T[["min","mean","50%","max"]].head(8))
