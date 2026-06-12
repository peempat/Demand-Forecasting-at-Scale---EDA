# ExpresSo NB Coffee Chain — Demand Forecasting Hackathon

## Problem
Forecast `units_sold` per `(store_id, category, date)` at **3 horizons: 1d / 7d / 1m**  
Window: 2024-11-01 → 2024-12-31 · 20 stores × 7 categories × 61 days × 3 horizons = **25,620 rows**  
Metric: **MAE** (lower is better)  
Public LB = November, Private LB = December (split by date, NOT random)

## Target construction
```python
target = (TRANSACTION
          .merge(ORDER[["order_id","store_id","date"]], on="order_id")
          .merge(PRODUCT[["product_id","category"]], on="product_id")
          .groupby(["store_id","category","date"])["units_sold"]
          .sum())
```
Zero-sales days = 0 in target. Full grid rate: **1.6% zeros**.

## Data tables (train/)
| Table | Rows | Notes |
|-------|------|-------|
| TRANSACTION | 2,858,050 | units_sold, revenue, discount_applied (0–100 %) |
| ORDER | 1,376,133 | store_id, date, hour, customer_id (60% null = walk-ins), payment_method |
| INVENTORY | 804,000 | is_stockout per product-day; use only for stockout flag NOT as target |
| PRODUCT | 60 | 7 categories, base_price |
| STORE | 20 | neighborhood_type, seating_capacity, staff_count, has_drive_through |
| DATE_DIM | 731 | is_weekend, is_holiday, is_payday, is_school_break, is_rainy_season |
| PROMOTION | 31,314 | start/end date, promo_type (Thai), discount_pct |
| LOCAL_EVENT | 1,440 | store-level event dates, event_type |
| CUSTOMER | 6,000 | is_member, preferred_store_id |

Train date range: **2023-01-01 → 2024-10-31**

## EDA Key Findings

### Target distribution
- Mean: 42.1, Median: 23, Max: 1607, Skewness: **4.15** (right-skewed)
- Zero-sales rate: 1.6%
- Stockout truncates demand → down-weight stockout rows (weight ≈ 0.3)

### Category breakdown (% of total units)
Coffee 49.2% · Tea 15.4% · Bakery 12.1% · Savory Bakery 9.5% · Chocolate & Milk 7.0% · Juice & Smoothie 4.2% · Merchandise 2.5%

### Calendar effects (mean units_sold)
- Weekend: **45.6** vs Weekday: 40.7 (+12%)
- Holiday: **64.8** vs Non-holiday: 41.2 (+57%)
- Payday: **51.6** vs Normal: 41.4 (+25%)
- School break: **47.7** vs Normal: 38.9 (+23%)
- Rainy season: 32.0 vs Dry: 45.9 (−30% — hot drinks?)

### Promotion effects
- With promo: **51.4** vs Without: 36.0 (+43%)
- Promo types: แต้มx2 (double points), สมาชิกใหม่ (new member), ลดราคา (discount), ซื้อ1แถม1 (BOGO), ชุดคู่ลด (bundle)

### Local events
- With event: **64.1** vs Without: 39.9 (+61%)
- Event types: market, cultural, food_festival, sports, convention, music_festival, book_fair, concert

### Stockout
- Train store-cat-day stockout rate: **34.6%** (very high!)
- Stockout days: mean 63.7 vs non-stockout: 30.7 (stockout days paradoxically show higher recorded demand — high-demand periods run out)
- Category stockout rates: Coffee 58.1% · Tea 45.7% · Chocolate & Milk 40.2%
- Test stockout rate 8.2% vs train 6.6% (slight drift)

### Autocorrelation
- Lag 1: 0.252, Lag 7: 0.162, Lag 14: 0.172 (weekly pattern visible)

### Store attributes
- Neighborhood types: university(5), tourist(4), hospital(3), gas_station(3), urban_residential(2), transit(2), mall(1), office(1)
- Only stores 11 and 13 have drive-through

## Submission format
`id = {store_id}_{category}_{forecast_date}_{horizon}` e.g. `1_Bakery_2024-11-01_1d`
Horizons: **1d, 7d, 1m** (next-day, week-ahead, month-ahead)

## Data integrity gotchas
1. `discount_applied` is percent (0–100), NOT fraction
2. `INVENTORY.units_sold` is noisy (~12% match TXN) — use only `is_stockout`
3. 60% walk-in orders have `customer_id = null` — structural, not missing
4. `test/` has NO TRANSACTION/ORDER/INVENTORY — don't reference them
5. PROMOTION/LOCAL_EVENT/DATE_DIM are byte-identical between train and test

## Feature engineering priority
1. **Lag features**: lag_1, lag_7, lag_14, lag_28 (per store×category)
2. **Rolling stats**: rolling_7_mean, rolling_7_std, rolling_14_mean, rolling_28_mean
3. **Date features**: day_of_week, month, is_weekend, is_holiday, is_payday, is_school_break
4. **Store features**: neighborhood_type, seating_capacity, staff_count, has_drive_through
5. **Promo features**: has_promo, avg_discount_pct, promo_type dummies
6. **Event flag**: has_local_event (per store-day)
7. **Stockout lag**: prior-day stockout flag
8. **Identity**: store_id, category (target encode)
9. **Horizon**: 1d / 7d / 1m as feature (to train single model for all horizons)

## Existing models
AutoGluon models saved in `c:\Users\Panuwit\Downloads\expresso3\coffee_demand_outputs\`:
- `autogluon_valid_1d` — 1d horizon (CatBoost ensemble)
- `autogluon_valid_7d` — 7d horizon (CatBoost ensemble)

## Project path
Data: `c:\Users\Panuwit\Downloads\edacoffee\super-ai-engineer-season-6-coffee-chain-hackathon\`
Python env: `.venv\Scripts\python.exe` (Python 3.12, pandas 3.0, numpy 2.4, seaborn 0.13)
EDA notebook: `eda_coffee.ipynb` (executed: `eda_coffee_executed.ipynb`)
