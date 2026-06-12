import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
data_path = Path(r'c:\Users\Panuwit\Downloads\edacoffee\super-ai-engineer-season-6-coffee-chain-hackathon\train')
output_path = Path(r'c:\Users\Panuwit\Downloads\edacoffee\eda_insights')

sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.dpi'] = 100

# Load data
print("Loading data...")
transaction = pd.read_csv(data_path / 'TRANSACTION.csv')
order = pd.read_csv(data_path / 'ORDER.csv')
product = pd.read_csv(data_path / 'PRODUCT.csv')
store = pd.read_csv(data_path / 'STORE.csv')
date_dim = pd.read_csv(data_path / 'DATE_DIM.csv')
promotion = pd.read_csv(data_path / 'PROMOTION.csv')
local_event = pd.read_csv(data_path / 'LOCAL_EVENT.csv')
customer = pd.read_csv(data_path / 'CUSTOMER.csv')

# Prepare data
order['date'] = pd.to_datetime(order['date'])
date_dim['date'] = pd.to_datetime(date_dim['date'])
promotion['start_date'] = pd.to_datetime(promotion['start_date'])
promotion['end_date'] = pd.to_datetime(promotion['end_date'])
local_event['date'] = pd.to_datetime(local_event['date'])

# Main dataframe
print("Building master dataframe...")
df = (transaction
      .merge(order[['order_id', 'store_id', 'date', 'hour', 'customer_id']], on='order_id')
      .merge(product[['product_id', 'category']], on='product_id')
      .merge(store[['store_id', 'neighborhood_type', 'seating_capacity', 'staff_count', 'has_drive_through']], on='store_id')
      .merge(date_dim[['date', 'is_weekend', 'is_holiday', 'is_payday', 'is_school_break', 'is_rainy_season']], on='date'))

df['day_of_week'] = df['date'].dt.day_name()
df['month'] = df['date'].dt.month

# Target construction
target = (transaction
          .merge(order[['order_id', 'store_id', 'date']], on='order_id')
          .merge(product[['product_id', 'category']], on='product_id')
          .groupby(['store_id', 'category', 'date'])['units_sold'].sum())

target_df = target.reset_index()
target_df = target_df.merge(date_dim[['date', 'is_weekend', 'is_holiday', 'is_payday', 'is_school_break', 'is_rainy_season']], on='date')
target_df = target_df.merge(store[['store_id', 'neighborhood_type']], on='store_id')
target_df['month'] = target_df['date'].dt.month

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
categories = df['category'].unique()

plot_num = 0

# ===== SECTION 6: STOCKOUT (FIXED) =====
print("6. Stockout analysis...")
plot_num += 1
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Stockout Analysis', fontsize=16, fontweight='bold')

# Use the columns that actually exist
from pathlib import Path as P
inv_path = data_path / 'INVENTORY.csv'
inventory = pd.read_csv(inv_path)
inventory['date'] = pd.to_datetime(inventory['date'])

# Create simple stockout aggregate
inv_agg = inventory.groupby('date')['is_stockout'].agg(['mean', 'sum', 'count'])

axes[0,0].plot(inv_agg.index, inv_agg['mean']*100, color='red', linewidth=2)
axes[0,0].set_title('Overall Daily Stockout Rate (%)')
axes[0,0].set_xlabel('Date')
axes[0,0].set_ylabel('Stockout %')

# By product
prod_stockout = inventory.merge(product[['product_id', 'category']], on='product_id')
cat_stock = prod_stockout.groupby('category')['is_stockout'].mean().sort_values(ascending=False)
axes[0,1].barh(cat_stock.index, cat_stock.values*100, color='red', alpha=0.7)
axes[0,1].set_title('Stockout Rate by Category')
axes[0,1].set_xlabel('Stockout %')

# Stockout count
axes[0,2].hist(inventory.groupby('date')['is_stockout'].mean()*100, bins=30, color='orange', edgecolor='black')
axes[0,2].set_title('Distribution of Daily Stockout Rates')
axes[0,2].set_xlabel('Stockout %')

# By product type
prod_stockout = inventory.merge(product[['product_id', 'category']], on='product_id')
by_cat = prod_stockout.groupby('category')['is_stockout'].sum()
axes[1,0].barh(by_cat.index[:5], by_cat.values[:5], color='darkred', alpha=0.7)
axes[1,0].set_title('Stockout Count by Category (Top 5)')
axes[1,0].set_xlabel('Count')

# Weekly trend
inv_agg['week'] = inv_agg.index.isocalendar().week
weekly_stock = inv_agg.groupby('week')['mean'].mean()
axes[1,1].bar(weekly_stock.index, weekly_stock.values*100, color='red', alpha=0.7)
axes[1,1].set_title('Weekly Avg Stockout Rate')
axes[1,1].set_xlabel('Week')
axes[1,1].set_ylabel('Stockout %')

axes[1,2].axis('off')

plt.tight_layout()
plt.savefig(output_path / '06_stockout_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== SECTION 7: STORE ANALYSIS =====
print("7. Store analysis...")
plot_num += 1
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Store Characteristics', fontsize=16, fontweight='bold')

nb_count = store['neighborhood_type'].value_counts()
axes[0,0].barh(nb_count.index, nb_count.values, color='steelblue')
axes[0,0].set_title('Stores by Neighborhood')
axes[0,0].set_xlabel('Count')

drive_thru = store['has_drive_through'].value_counts()
colors_pie = ['lightblue', 'gold']
axes[0,1].pie(drive_thru.values, labels=['No', 'Yes'], autopct='%1.1f%%', colors=colors_pie)
axes[0,1].set_title('Drive-Through Availability')

axes[0,2].hist(store['seating_capacity'], bins=10, color='green', edgecolor='black')
axes[0,2].set_title('Seating Capacity')
axes[0,2].set_xlabel('Seats')

axes[1,0].hist(store['staff_count'], bins=10, color='orange', edgecolor='black')
axes[1,0].set_title('Staff Count')
axes[1,0].set_xlabel('Staff')

store_sales = target_df.groupby('store_id')['units_sold'].mean()
axes[1,1].bar(range(len(store_sales)), sorted(store_sales.values), color='steelblue', alpha=0.7)
axes[1,1].set_title('Store Sales Ranking')
axes[1,1].set_xlabel('Store')
axes[1,1].set_ylabel('Avg Units')

nb_sales = target_df.groupby('neighborhood_type')['units_sold'].mean()
axes[1,2].bar(range(len(nb_sales)), nb_sales.values, color='coral', alpha=0.7)
axes[1,2].set_xticks(range(len(nb_sales)))
axes[1,2].set_xticklabels(nb_sales.index, rotation=45, ha='right', fontsize=8)
axes[1,2].set_title('Avg Sales by Neighborhood')
axes[1,2].set_ylabel('Units')

plt.tight_layout()
plt.savefig(output_path / '07_store_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== SECTION 8: TIMESERIES =====
print("8. Time series analysis...")
plot_num += 1
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Time Series Patterns', fontsize=16, fontweight='bold')

daily_sales = target_df.groupby('date')['units_sold'].sum().sort_index()
axes[0,0].plot(daily_sales.index, daily_sales.values, color='steelblue', alpha=0.7, linewidth=1)
axes[0,0].set_title('Daily Total Sales')
axes[0,0].set_xlabel('Date')
axes[0,0].tick_params(axis='x', rotation=45)

weekly_sales = target_df.groupby(target_df['date'].dt.isocalendar().week)['units_sold'].sum()
axes[0,1].plot(weekly_sales.index, weekly_sales.values, marker='o', color='coral', markersize=4)
axes[0,1].set_title('Weekly Sales')
axes[0,1].set_xlabel('Week')

monthly_sales = target_df.groupby('month')['units_sold'].sum()
axes[0,2].bar(monthly_sales.index, monthly_sales.values, color='green', alpha=0.7)
axes[0,2].set_title('Monthly Sales')
axes[0,2].set_xlabel('Month')

window = 7
rolling_mean = daily_sales.rolling(window).mean()
axes[1,0].plot(daily_sales.index, daily_sales.values, alpha=0.3, color='gray', label='Daily')
axes[1,0].plot(rolling_mean.index, rolling_mean.values, color='red', linewidth=2, label='7-day MA')
axes[1,0].set_title('Rolling Average (7-day)')
axes[1,0].legend()
axes[1,0].tick_params(axis='x', rotation=45)

weekly_pattern = target_df.groupby(target_df['date'].dt.day_name())['units_sold'].mean().reindex(day_order)
axes[1,1].bar(range(7), weekly_pattern.values, color='purple', alpha=0.7)
axes[1,1].set_xticks(range(7))
axes[1,1].set_xticklabels(['M','T','W','Th','F','Sa','Su'])
axes[1,1].set_title('Day of Week Pattern')

monthly_pattern = target_df.groupby('month')['units_sold'].mean()
axes[1,2].bar(monthly_pattern.index, monthly_pattern.values, color='orange', alpha=0.7)
axes[1,2].set_title('Monthly Average')
axes[1,2].set_xlabel('Month')

plt.tight_layout()
plt.savefig(output_path / '08_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== SECTION 9: HOURLY & TRANSACTIONS =====
print("9. Hourly patterns...")
plot_num += 1
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Hourly & Transaction Patterns', fontsize=16, fontweight='bold')

hourly = df.groupby('hour')['units_sold'].agg(['mean', 'count'])
axes[0,0].bar(hourly.index, hourly['mean'], color='steelblue', alpha=0.7)
axes[0,0].set_title('Avg Units by Hour')
axes[0,0].set_xlabel('Hour')
axes[0,0].set_xlim(-0.5, 23.5)

axes[0,1].bar(hourly.index, hourly['count']/1000, color='coral', alpha=0.7)
axes[0,1].set_title('Transactions by Hour')
axes[0,1].set_xlabel('Hour')
axes[0,1].set_xlim(-0.5, 23.5)

peak_hours = hourly.nlargest(5, 'mean')
axes[0,2].barh(peak_hours.index.astype(str), peak_hours['mean'], color='gold')
axes[0,2].set_title('Top 5 Peak Hours')

axes[1,0].hist(df['revenue'], bins=40, color='green', edgecolor='black', alpha=0.7)
axes[1,0].set_title('Revenue Distribution')

axes[1,1].scatter(df['units_sold'], df['revenue'], alpha=0.2, s=5)
axes[1,1].set_title('Units vs Revenue')
axes[1,1].set_xlabel('Units')
axes[1,1].set_ylabel('Revenue')

discount_by_units = df.groupby(df['discount_applied']//10)['units_sold'].mean()
axes[1,2].bar(discount_by_units.index, discount_by_units.values, color='red', alpha=0.7)
axes[1,2].set_title('Units by Discount Bracket')
axes[1,2].set_xlabel('Discount % (binned)')

plt.tight_layout()
plt.savefig(output_path / '09_hourly_transactions.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== SECTION 10: ADVANCED INSIGHTS =====
print("10. Advanced insights...")
plot_num += 1
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Advanced Insights & Interactions', fontsize=16, fontweight='bold')

# Correlation
corr_data = target_df[['units_sold', 'is_weekend', 'is_holiday', 'is_payday', 'is_school_break', 'is_rainy_season']].astype(float)
corr = corr_data.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0,0], square=True)
axes[0,0].set_title('Correlation Matrix')

# Category by neighborhood
cat_nb = target_df.pivot_table(values='units_sold', index='category', columns='neighborhood_type', aggfunc='mean')
if cat_nb.shape[0] > 0 and cat_nb.shape[1] > 0:
    sns.heatmap(cat_nb.iloc[:5, :], annot=True, fmt='.0f', cmap='YlGnBu', ax=axes[0,1], cbar_kws={'shrink': 0.8})
    axes[0,1].set_title('Top Categories × Neighborhoods')

# Multi-factor: Weekend + Holiday
weekend_holiday = target_df.groupby(['is_weekend', 'is_holiday'])['units_sold'].mean().unstack()
if weekend_holiday.shape[1] >= 2:
    x = np.arange(2)
    axes[0,2].bar(x-0.2, weekend_holiday.iloc[:, 0], 0.4, label='Non-Holiday', color='lightblue')
    axes[0,2].bar(x+0.2, weekend_holiday.iloc[:, 1], 0.4, label='Holiday', color='salmon')
    axes[0,2].set_xticks(x)
    axes[0,2].set_xticklabels(['Weekday', 'Weekend'])
    axes[0,2].set_title('Weekend × Holiday Interaction')
    axes[0,2].legend()
else:
    axes[0,2].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[0,2].transAxes)

# Neighborhood performance
nb_perf = target_df.groupby('neighborhood_type')['units_sold'].agg(['mean', 'std'])
axes[1,0].errorbar(range(len(nb_perf)), nb_perf['mean'], yerr=nb_perf['std'], fmt='o', markersize=8, capsize=5)
axes[1,0].set_xticks(range(len(nb_perf)))
axes[1,0].set_xticklabels(nb_perf.index, rotation=45, ha='right', fontsize=8)
axes[1,0].set_title('Neighborhood Performance ± Std')
axes[1,0].set_ylabel('Avg Units')

# Category performance
cat_perf = target_df.groupby('category')['units_sold'].agg(['mean', 'count']).sort_values('mean', ascending=False)
colors_cat = plt.cm.viridis(np.linspace(0, 1, len(cat_perf)))
axes[1,1].barh(cat_perf.index, cat_perf['mean'], color=colors_cat)
axes[1,1].set_title('Category Performance')
axes[1,1].set_xlabel('Avg Units')

# Target distribution by quantile
quantiles = pd.qcut(target_df['units_sold'], q=5, duplicates='drop')
q_counts = quantiles.value_counts().sort_index()
colors_q = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(q_counts)))
axes[1,2].bar(range(len(q_counts)), q_counts.values, color=colors_q)
axes[1,2].set_xticks(range(len(q_counts)))
axes[1,2].set_xticklabels([f'Q{i+1}' for i in range(len(q_counts))])
axes[1,2].set_title('Distribution by Quantile')

plt.tight_layout()
plt.savefig(output_path / '10_advanced_insights.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*60}")
print(f"[DONE] FAST EDA COMPLETE!")
print(f"{'='*60}")
print(f"Total figures generated: {plot_num + 5}  (Figures 1-5 already created)")
print(f"Total plots created: ~60+ plots across all dimensions")
print(f"\nOutput directory: {output_path}")
print(f"\nFiles generated:")
for i, file in enumerate(sorted(output_path.glob('*.png')), 1):
    print(f"  {i}. {file.name}")

print(f"\n{'='*60}")
print(f"Key Dimensions Covered:")
print(f"  1. Target Distribution & Statistics")
print(f"  2. Category Analysis (units, trends, distribution)")
print(f"  3. Calendar Effects (weekends, holidays, paydays, seasons)")
print(f"  4. Promotion Impact Analysis")
print(f"  5. Local Events Effect")
print(f"  6. Stockout Patterns")
print(f"  7. Store Characteristics & Performance")
print(f"  8. Time Series & Seasonality")
print(f"  9. Hourly Patterns & Transactions")
print(f"  10. Correlation & Advanced Interactions")
print(f"{'='*60}")
