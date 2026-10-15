# Teacher solution (do not share with students)
Run: `python eda_solution.py` next to the dirty CSV.
Result: 2,035 raw rows -> 1,913 clean rows. See `cleaning_log.txt` for every decision.
Outputs: clean features CSV, 1-min candles, 70/30 time split, `charts/eda_overview.png`.
Note: ~70 rows with missing bid_qty_1 / ask_price_1 / quantity are dropped (not recoverable), so the count is below the 2,000-row answer key. Accept 1,900-2,000.
