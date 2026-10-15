# Trading Data EDA Task (Synthetic BTCUSDT)

**File:** `trades_orderbook_STUDENT_dirty.csv` (2,035 rows, deliberately messy)
Each row = one trade + the top 3 bid/ask levels of the order book at that moment.
All data is synthetic. Educational material, not financial advice.

## Goal
Take the dirty file through the full EDA process: clean it, build features, make 1-minute candles, visualize, and prepare it for a model. **Document every decision.**

## Rules
1. Never shuffle time-series data. Split by time only.
2. Avoid look-ahead bias: no feature may use future data.
3. Write down why you dropped, filled, or fixed each problem.

## Steps

### 1. Load and inspect
- Read the CSV with pandas. Run `head()`, `shape`, `info()`, `dtypes`.
- Note column types and row count.

### 2. Summarize
- Run `describe()`; run `value_counts()` on `side` and `symbol`.
- Report min, max, mean, median of `price` and `quantity`.

### 3. Find the problems (report counts for each)
- Duplicate rows
- Missing values (price, quantity, side, bid_qty_1, ask_price_1)
- Mixed timestamp formats
- Inconsistent `side` and `symbol` labels
- Prices stored as text
- Outlier prices and negative prices
- Zero quantities
- Crossed books (bid_price_1 >= ask_price_1)
- Rows out of time order

### 4. Clean
- Drop duplicates by `trade_id`.
- Standardize `side` (buy/sell) and `symbol` (BTCUSDT).
- Convert price text (`$`, commas) to numbers.
- Parse all timestamps into one datetime format.
- Fix or drop bad rows, and write down why.

### 5. Handle outliers
- Find extreme prices with a rolling median or IQR, then remove or correct them.

### 6. Engineer features
- `spread = ask_price_1 - bid_price_1`
- `mid_price = (ask_price_1 + bid_price_1) / 2`
- `order_imbalance = (bid_qty_1 - ask_qty_1) / (bid_qty_1 + ask_qty_1)`
- Returns and rolling volatility
- Optional: RSI, moving averages

### 7. Resample
- Convert ticks to 1-minute OHLCV candles, sorted by time.

### 8. Visualize
- Price line chart, volume histogram, spread distribution, buy vs sell counts, correlation heatmap.

### 9. Prepare for a model
- Define a target, e.g. "price up or down in the next 5 minutes".
- Split by time (first 70% train, last 30% test).
- Scale features if needed.
- Save the cleaned file.

## Deliverables
1. `trades_orderbook_clean.csv` (cleaned data with features)
2. `candles_1min.csv`
3. Your notebook or script
4. A short **cleaning log**: each problem found, how many rows, what you did, why
5. Charts from step 8

## Bonus: 3Commas demo
Cleaned data, then features, then a simple rule (e.g. buy when `order_imbalance > 0.3`), then a backtest, then buy/sell signals to 3Commas by webhook, **paper trading first**.

## Scoring idea
Compare your cleaned file to the answer key: row count, missing values, matching prices, and quality of your decision log.
