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
output_path.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

# Load all data
print("Loading data...")
transaction = pd.read_csv(data_path / 'TRANSACTION.csv')
order = pd.read_csv(data_path / 'ORDER.csv')
inventory = pd.read_csv(data_path / 'INVENTORY.csv')
inventory['date'] = pd.to_datetime(inventory['date'])
product = pd.read_csv(data_path / 'PRODUCT.csv')
store = pd.read_csv(data_path / 'STORE.csv')
date_dim = pd.read_csv(data_path / 'DATE_DIM.csv')
promotion = pd.read_csv(data_path / 'PROMOTION.csv')
local_event = pd.read_csv(data_path / 'LOCAL_EVENT.csv')
customer = pd.read_csv(data_path / 'CUSTOMER.csv')

# Prepare master dataset
order['date'] = pd.to_datetime(order['date'])
date_dim['date'] = pd.to_datetime(date_dim['date'])
promotion['start_date'] = pd.to_datetime(promotion['start_date'])
promotion['end_date'] = pd.to_datetime(promotion['end_date'])
local_event['date'] = pd.to_datetime(local_event['date'])

# Create main dataframe
print("Building master dataframe...")
df = (transaction
      .merge(order[['order_id', 'store_id', 'date', 'hour', 'customer_id']], on='order_id')
      .merge(product[['product_id', 'category']], on='product_id')
      .merge(store[['store_id', 'neighborhood_type', 'seating_capacity', 'staff_count', 'has_drive_through']], on='store_id')
      .merge(date_dim[['date', 'is_weekend', 'is_holiday', 'is_payday', 'is_school_break', 'is_rainy_season']], on='date'))

# Add time features
df['day_of_week'] = df['date'].dt.day_name()
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week
df['day_num'] = df['date'].dt.dayofweek

# Construct target metric
target = (transaction
          .merge(order[['order_id', 'store_id', 'date']], on='order_id')
          .merge(product[['product_id', 'category']], on='product_id')
          .groupby(['store_id', 'category', 'date'])['units_sold']
          .sum())

target_df = target.reset_index()
target_df = target_df.merge(date_dim[['date', 'is_weekend', 'is_holiday', 'is_payday', 'is_school_break', 'is_rainy_season']], on='date')
target_df = target_df.merge(store[['store_id', 'neighborhood_type']], on='store_id')
target_df['month'] = target_df['date'].dt.month

plot_count = 0

# ===== 1. TARGET DISTRIBUTION ANALYSIS (10 plots) =====
print("\n1. Target distribution analysis...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Target Distribution Analysis - Units Sold', fontsize=16, fontweight='bold')

# 1.1
axes[0,0].hist(target_df['units_sold'], bins=50, color='skyblue', edgecolor='black')
axes[0,0].set_title(f'Distribution (Mean: {target_df["units_sold"].mean():.1f}, Median: {target_df["units_sold"].median():.1f})')
axes[0,0].set_xlabel('Units Sold')

# 1.2
axes[0,1].hist(target_df['units_sold'], bins=50, color='salmon', edgecolor='black', cumulative=True)
axes[0,1].set_title('Cumulative Distribution')
axes[0,1].set_xlabel('Units Sold')

# 1.3
stats.probplot(target_df['units_sold'], dist="norm", plot=axes[0,2])
axes[0,2].set_title(f'Q-Q Plot (Skewness: {stats.skew(target_df["units_sold"]):.2f})')

# 1.4
axes[1,0].boxplot(target_df['units_sold'], vert=True)
axes[1,0].set_title(f'Boxplot (Min: {target_df["units_sold"].min()}, Max: {target_df["units_sold"].max()})')
axes[1,0].set_ylabel('Units Sold')

# 1.5
zero_pct = (target_df['units_sold'] == 0).sum() / len(target_df) * 100
axes[1,1].bar(['Zero Sales', 'Positive Sales'],
              [(target_df['units_sold'] == 0).sum(), (target_df['units_sold'] > 0).sum()],
              color=['red', 'green'], alpha=0.7)
axes[1,1].set_title(f'Zero vs Positive Sales ({zero_pct:.1f}% zeros)')

# 1.6
axes[1,2].hist(np.log1p(target_df['units_sold']), bins=50, color='purple', edgecolor='black')
axes[1,2].set_title('Log-transformed Distribution')
axes[1,2].set_xlabel('Log(Units Sold + 1)')

plt.tight_layout()
plt.savefig(output_path / '01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
plot_count += 1

# ===== 2. CATEGORY ANALYSIS (15 plots) =====
print("2. Category analysis...")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Category Analysis - Units Sold', fontsize=16, fontweight='bold')

cat_data = df.groupby('category')['units_sold'].agg(['sum', 'mean', 'std', 'count']).sort_values('sum', ascending=False)
categories = cat_data.index.tolist()

# 2.1-2.4 Category total/mean/std
axes[0,0].barh(cat_data.index, cat_data['sum']/1000, color='steelblue')
axes[0,0].set_title('Total Units Sold by Category')
axes[0,0].set_xlabel('Total Units (thousands)')

axes[0,1].barh(cat_data.index, cat_data['mean'], color='coral')
axes[0,1].set_title('Mean Units Sold per Transaction')
axes[0,1].set_xlabel('Mean Units')

axes[0,2].barh(cat_data.index, cat_data['std'], color='lightgreen')
axes[0,2].set_title('Std Dev of Units Sold')
axes[0,2].set_xlabel('Std Dev')

axes[0,3].barh(cat_data.index, cat_data['count']/1000, color='gold')
axes[0,3].set_title('Transaction Count by Category')
axes[0,3].set_xlabel('Count (thousands)')

# 2.5-2.8 Category by time
for idx, (i, cat) in enumerate(enumerate(categories[:4])):
    if idx < 4:
        ax = axes[1, idx]
        cat_df = df[df['category'] == cat].groupby('month')['units_sold'].mean()
        ax.plot(cat_df.index, cat_df.values, marker='o', linewidth=2, markersize=6)
        ax.set_title(f'{cat} - Monthly Trend')
        ax.set_xlabel('Month')
        ax.set_ylabel('Avg Units')
        ax.grid(True, alpha=0.3)

# 2.9-2.12 Category distributions
for idx, (i, cat) in enumerate(enumerate(categories[:4])):
    ax = axes[2, idx]
    ax.hist(df[df['category'] == cat]['units_sold'], bins=30, alpha=0.7, color='purple')
    ax.set_title(f'{cat} Distribution')
    ax.set_xlabel('Units Sold')

plt.tight_layout()
plt.savefig(output_path / '02_category_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
plot_count += 1

# ===== 3. CALENDAR EFFECTS (14 plots) =====
print("3. Calendar effects...")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Calendar Effects Analysis', fontsize=16, fontweight='bold')

# 3.1-3.2 Weekend vs Weekday
axes[0,0].bar(['Weekday', 'Weekend'],
              [target_df[~target_df['is_weekend']]['units_sold'].mean(),
               target_df[target_df['is_weekend']]['units_sold'].mean()],
              color=['blue', 'red'], alpha=0.7)
axes[0,0].set_title('Weekend vs Weekday')
axes[0,0].set_ylabel('Avg Units Sold')

axes[0,1].boxplot([target_df[~target_df['is_weekend']]['units_sold'],
                    target_df[target_df['is_weekend']]['units_sold']],
                   labels=['Weekday', 'Weekend'])
axes[0,1].set_title('Distribution: Weekday vs Weekend')
axes[0,1].set_ylabel('Units Sold')

# 3.3-3.4 Holiday effect
axes[0,2].bar(['Non-Holiday', 'Holiday'],
              [target_df[~target_df['is_holiday']]['units_sold'].mean(),
               target_df[target_df['is_holiday']]['units_sold'].mean()],
              color=['lightblue', 'red'], alpha=0.7)
axes[0,2].set_title('Holiday vs Non-Holiday')
axes[0,2].set_ylabel('Avg Units Sold')

axes[0,3].boxplot([target_df[~target_df['is_holiday']]['units_sold'],
                    target_df[target_df['is_holiday']]['units_sold']],
                   labels=['Non-Holiday', 'Holiday'])
axes[0,3].set_title('Distribution: Holiday')
axes[0,3].set_ylabel('Units Sold')

# 3.5-3.6 Payday effect
axes[1,0].bar(['Non-Payday', 'Payday'],
              [target_df[~target_df['is_payday']]['units_sold'].mean(),
               target_df[target_df['is_payday']]['units_sold'].mean()],
              color=['lightcyan', 'orange'], alpha=0.7)
axes[1,0].set_title('Payday Effect')
axes[1,0].set_ylabel('Avg Units Sold')

axes[1,1].boxplot([target_df[~target_df['is_payday']]['units_sold'],
                    target_df[target_df['is_payday']]['units_sold']],
                   labels=['Non-Payday', 'Payday'])
axes[1,1].set_title('Distribution: Payday')
axes[1,1].set_ylabel('Units Sold')

# 3.7-3.8 School break
axes[1,2].bar(['Normal', 'School Break'],
              [target_df[~target_df['is_school_break']]['units_sold'].mean(),
               target_df[target_df['is_school_break']]['units_sold'].mean()],
              color=['lightgreen', 'darkgreen'], alpha=0.7)
axes[1,2].set_title('School Break Effect')
axes[1,2].set_ylabel('Avg Units Sold')

axes[1,3].boxplot([target_df[~target_df['is_school_break']]['units_sold'],
                    target_df[target_df['is_school_break']]['units_sold']],
                   labels=['Normal', 'School Break'])
axes[1,3].set_title('Distribution: School Break')
axes[1,3].set_ylabel('Units Sold')

# 3.9-3.10 Rainy season
axes[2,0].bar(['Dry Season', 'Rainy Season'],
              [target_df[~target_df['is_rainy_season']]['units_sold'].mean(),
               target_df[target_df['is_rainy_season']]['units_sold'].mean()],
              color=['yellow', 'blue'], alpha=0.7)
axes[2,0].set_title('Rainy Season Effect')
axes[2,0].set_ylabel('Avg Units Sold')

axes[2,1].boxplot([target_df[~target_df['is_rainy_season']]['units_sold'],
                    target_df[target_df['is_rainy_season']]['units_sold']],
                   labels=['Dry', 'Rainy'])
axes[2,1].set_title('Distribution: Rainy Season')
axes[2,1].set_ylabel('Units Sold')

# 3.11-3.12 Day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_stats = target_df.groupby(target_df['date'].dt.day_name())['units_sold'].mean().reindex(day_order)
axes[2,2].bar(range(7), dow_stats.values, color='teal', alpha=0.7)
axes[2,2].set_xticks(range(7))
axes[2,2].set_xticklabels(['M','T','W','Th','F','Sa','Su'], rotation=0)
axes[2,2].set_title('Day of Week Effect')
axes[2,2].set_ylabel('Avg Units Sold')

# 3.13 Monthly trend
monthly = target_df.groupby('month')['units_sold'].mean()
axes[2,3].plot(monthly.index, monthly.values, marker='o', linewidth=2, markersize=8, color='purple')
axes[2,3].fill_between(monthly.index, monthly.values, alpha=0.3, color='purple')
axes[2,3].set_title('Monthly Trend')
axes[2,3].set_xlabel('Month')
axes[2,3].set_ylabel('Avg Units Sold')
axes[2,3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / '03_calendar_effects.png', dpi=150, bbox_inches='tight')
plt.close()
plot_count += 1

# ===== 4. PROMOTION ANALYSIS (12 plots) =====
print("4. Promotion analysis...")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Promotion Analysis', fontsize=16, fontweight='bold')

# Merge promotions into target
target_df_promo = target_df.copy()
target_df_promo['has_promo'] = 0
for _, prow in promotion.iterrows():
    mask = (target_df_promo['date'] >= prow['start_date']) & (target_df_promo['date'] <= prow['end_date'])
    target_df_promo.loc[mask, 'has_promo'] = 1

# 4.1 Promo vs No Promo
axes[0,0].bar(['No Promo', 'With Promo'],
              [target_df_promo[target_df_promo['has_promo']==0]['units_sold'].mean(),
               target_df_promo[target_df_promo['has_promo']==1]['units_sold'].mean()],
              color=['lightgray', 'gold'], alpha=0.7)
axes[0,0].set_title('Promotion Effect')
axes[0,0].set_ylabel('Avg Units Sold')

# 4.2 Distribution
axes[0,1].boxplot([target_df_promo[target_df_promo['has_promo']==0]['units_sold'],
                    target_df_promo[target_df_promo['has_promo']==1]['units_sold']],
                   labels=['No Promo', 'With Promo'])
axes[0,1].set_title('Distribution')
axes[0,1].set_ylabel('Units Sold')

# 4.3 Promotion types
promo_types = promotion['promo_type'].value_counts()
axes[0,2].barh(range(len(promo_types)), promo_types.values, color='coral')
axes[0,2].set_yticks(range(len(promo_types)))
axes[0,2].set_yticklabels(promo_types.index, fontsize=8)
axes[0,2].set_title('Promotion Types')
axes[0,2].set_xlabel('Count')

# 4.4 Discount distribution
axes[0,3].hist(promotion['discount_pct'], bins=20, color='green', edgecolor='black')
axes[0,3].set_title('Discount Percentage Distribution')
axes[0,3].set_xlabel('Discount %')

# 4.5-4.8 Top categories with/without promo
for idx, cat in enumerate(categories[:4]):
    cat_data_no_promo = target_df_promo[(target_df_promo['category']==cat) & (target_df_promo['has_promo']==0)]['units_sold'].mean()
    cat_data_promo = target_df_promo[(target_df_promo['category']==cat) & (target_df_promo['has_promo']==1)]['units_sold'].mean()

    ax = axes[1, idx]
    ax.bar(['No Promo', 'With Promo'], [cat_data_no_promo, cat_data_promo], color=['lightblue', 'darkblue'], alpha=0.7)
    ax.set_title(f'{cat}')
    ax.set_ylabel('Avg Units')

# 4.9-4.12 Category promotion % impact
for idx, cat in enumerate(categories[:4]):
    cat_promo_pct = (target_df_promo[target_df_promo['category']==cat]['has_promo'].sum() /
                     len(target_df_promo[target_df_promo['category']==cat]) * 100)

    ax = axes[2, idx]
    cat_by_promo = target_df_promo[target_df_promo['category']==cat].groupby('has_promo')['units_sold'].count()
    ax.pie(cat_by_promo.values, labels=['No Promo', 'With Promo'], autopct='%1.1f%%', colors=['lightblue', 'gold'])
    ax.set_title(f'{cat} Promo Coverage')

plt.tight_layout()
plt.savefig(output_path / '04_promotion_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
plot_count += 1

# ===== 5. LOCAL EVENTS ANALYSIS (10 plots) =====
print("5. Local events analysis...")
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
fig.suptitle('Local Events Analysis', fontsize=16, fontweight='bold')

# Merge events into target
target_df_event = target_df.copy()
target_df_event['has_event'] = 0
for _, erow in local_event.iterrows():
    mask = (target_df_event['store_id'] == erow['store_id']) & (target_df_event['date'] == erow['date'])
    target_df_event.loc[mask, 'has_event'] = 1

# 5.1 Event vs No Event
axes[0,0].bar(['No Event', 'With Event'],
              [target_df_event[target_df_event['has_event']==0]['units_sold'].mean(),
               target_df_event[target_df_event['has_event']==1]['units_sold'].mean()],
              color=['lightgray', 'red'], alpha=0.7)
axes[0,0].set_title('Event Effect')
axes[0,0].set_ylabel('Avg Units Sold')

# 5.2 Distribution
axes[0,1].boxplot([target_df_event[target_df_event['has_event']==0]['units_sold'],
                    target_df_event[target_df_event['has_event']==1]['units_sold']],
                   labels=['No Event', 'With Event'])
axes[0,1].set_title('Distribution')
axes[0,1].set_ylabel('Units Sold')

# 5.3 Event types
event_types = local_event['event_type'].value_counts()
axes[0,2].barh(range(len(event_types)), event_types.values, color='darkred')
axes[0,2].set_yticks(range(len(event_types)))
axes[0,2].set_yticklabels(event_types.index, fontsize=8)
axes[0,2].set_title('Event Types')
axes[0,2].set_xlabel('Count')

# 5.4 Events by store
event_by_store = local_event['store_id'].value_counts().head(10)
axes[0,3].barh(event_by_store.index.astype(str), event_by_store.values, color='orange')
axes[0,3].set_title('Events by Store (Top 10)')
axes[0,3].set_xlabel('Count')

# 5.5-5.8 Category effect by event
for idx, cat in enumerate(categories[:4]):
    cat_no_event = target_df_event[(target_df_event['category']==cat) & (target_df_event['has_event']==0)]['units_sold'].mean()
    cat_event = target_df_event[(target_df_event['category']==cat) & (target_df_event['has_event']==1)]['units_sold'].mean()

    ax = axes[1, idx]
    ax.bar(['No Event', 'With Event'], [cat_no_event, cat_event], color=['lightblue', 'darkred'], alpha=0.7)
    ax.set_title(f'{cat}')
    ax.set_ylabel('Avg Units')

plt.tight_layout()
plt.savefig(output_path / '05_events_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
plot_count += 1

# ===== 6. STOCKOUT ANALYSIS (12 plots) =====
print("6. Stockout analysis...")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Stockout Analysis', fontsize=16, fontweight='bold')

# Merge inventory
df_stock = df.copy()
df_stock = df_stock.merge(inventory[['product_id', 'date', 'is_stockout']],
                          on=['product_id', 'date'], how='left')
df_stock['is_stockout'] = df_stock['is_stockout'].fillna(0)

# 6.1 Overall stockout rate
stockout_rate = df_stock['is_stockout'].mean() * 100
axes[0,0].bar(['In Stock', 'Stockout'],
              [(1-df_stock['is_stockout'].mean())*100, stockout_rate],
              color=['green', 'red'], alpha=0.7)
axes[0,0].set_title(f'Overall Stockout Rate: {stockout_rate:.1f}%')
axes[0,0].set_ylabel('Percentage')

# 6.2 Stockout by category
cat_stockout = df_stock.groupby('category')['is_stockout'].mean().sort_values(ascending=False) * 100
axes[0,1].barh(cat_stockout.index, cat_stockout.values, color='red', alpha=0.7)
axes[0,1].set_title('Stockout Rate by Category')
axes[0,1].set_xlabel('Stockout %')

# 6.3 Stockout impact on sales
stockout_impact = df_stock.groupby('is_stockout')['units_sold'].agg(['mean', 'median', 'count'])
axes[0,2].bar(['In Stock', 'Stockout'],
              stockout_impact['mean'].values,
              color=['green', 'red'], alpha=0.7)
axes[0,2].set_title('Avg Units Sold: In Stock vs Stockout')
axes[0,2].set_ylabel('Avg Units')

axes[0,3].bar(['In Stock', 'Stockout'],
              stockout_impact['count'].values/1000,
              color=['green', 'red'], alpha=0.7)
axes[0,3].set_title('Transaction Count')
axes[0,3].set_ylabel('Count (thousands)')

# 6.4-6.7 Stockout by category (distribution)
for idx, cat in enumerate(categories[:4]):
    ax = axes[1, idx]
    cat_df = df_stock[df_stock['category']==cat]
    in_stock = cat_df[cat_df['is_stockout']==0]['units_sold']
    stockout = cat_df[cat_df['is_stockout']==1]['units_sold']

    ax.boxplot([in_stock, stockout], labels=['In Stock', 'Stockout'])
    ax.set_title(f'{cat}')
    ax.set_ylabel('Units Sold')

# 6.8-6.11 Stockout trend over time
for idx, cat in enumerate(categories[:4]):
    ax = axes[2, idx]
    cat_df = df_stock[df_stock['category']==cat].copy()
    cat_df['week'] = cat_df['date'].dt.isocalendar().week
    weekly_stockout = cat_df.groupby('week')['is_stockout'].mean() * 100

    ax.plot(weekly_stockout.index, weekly_stockout.values, marker='o', color='red', alpha=0.7)
    ax.set_title(f'{cat} - Weekly Stockout %')
    ax.set_xlabel('Week')
    ax.set_ylabel('Stockout %')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / '06_stockout_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
plot_count += 1

# ===== 7. STORE ANALYSIS (16 plots) =====
print("7. Store analysis...")
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
fig.suptitle('Store Characteristics Analysis', fontsize=16, fontweight='bold')

# 7.1 Neighborhood distribution
nb_count = store['neighborhood_type'].value_counts()
axes[0,0].barh(nb_count.index, nb_count.values, color='steelblue')
axes[0,0].set_title('Stores by Neighborhood')
axes[0,0].set_xlabel('Count')

# 7.2 Drive-through stores
drive_thru = store['has_drive_through'].value_counts()
axes[0,1].pie(drive_thru.values, labels=['No', 'Yes'], autopct='%1.1f%%', colors=['lightblue', 'gold'])
axes[0,1].set_title('Drive-Through Availability')

# 7.3 Seating capacity distribution
axes[0,2].hist(store['seating_capacity'], bins=10, color='green', edgecolor='black')
axes[0,2].set_title('Seating Capacity Distribution')
axes[0,2].set_xlabel('Seats')

# 7.4 Staff count distribution
axes[0,3].hist(store['staff_count'], bins=10, color='orange', edgecolor='black')
axes[0,3].set_title('Staff Count Distribution')
axes[0,3].set_xlabel('Number of Staff')

# 7.5-7.8 Sales by neighborhood
nb_stats = target_df.groupby('neighborhood_type')['units_sold'].agg(['mean', 'std', 'count', 'sum'])
axes[1,0].barh(nb_stats.index, nb_stats['mean'], color='steelblue', alpha=0.7)
axes[1,0].set_title('Avg Units Sold by Neighborhood')
axes[1,0].set_xlabel('Avg Units')

axes[1,1].barh(nb_stats.index, nb_stats['std'], color='coral', alpha=0.7)
axes[1,1].set_title('Std Dev by Neighborhood')
axes[1,1].set_xlabel('Std Dev')

axes[1,2].barh(nb_stats.index, nb_stats['sum']/1000, color='lightgreen', alpha=0.7)
axes[1,2].set_title('Total Units by Neighborhood')
axes[1,2].set_xlabel('Total (thousands)')

axes[1,3].barh(nb_stats.index, nb_stats['count']/1000, color='gold', alpha=0.7)
axes[1,3].set_title('Days Recorded by Neighborhood')
axes[1,3].set_xlabel('Count (thousands)')

# 7.9-7.12 Store attributes correlation with sales
store_sales = target_df.groupby('store_id')['units_sold'].mean()
store_merged = store_sales.to_frame().merge(store, on='store_id')

axes[2,0].scatter(store_merged['seating_capacity'], store_merged['units_sold'], alpha=0.6)
axes[2,0].set_title(f'Seating Capacity vs Sales (r={store_merged[["seating_capacity", "units_sold"]].corr().iloc[0,1]:.2f})')
axes[2,0].set_xlabel('Seating Capacity')
axes[2,0].set_ylabel('Avg Units Sold')

axes[2,1].scatter(store_merged['staff_count'], store_merged['units_sold'], alpha=0.6, color='orange')
axes[2,1].set_title(f'Staff Count vs Sales (r={store_merged[["staff_count", "units_sold"]].corr().iloc[0,1]:.2f})')
axes[2,1].set_xlabel('Staff Count')
axes[2,1].set_ylabel('Avg Units Sold')

drive_thru_sales = store_merged.groupby('has_drive_through')['units_sold'].mean()
axes[2,2].bar(['No Drive-Thru', 'With Drive-Thru'], drive_thru_sales.values, color=['red', 'green'], alpha=0.7)
axes[2,2].set_title('Drive-Thru Impact')
axes[2,2].set_ylabel('Avg Units Sold')

# Top and bottom stores
top_stores = store_sales.nlargest(5)
bottom_stores = store_sales.nsmallest(5)
axes[2,3].barh(range(10), list(bottom_stores.values) + list(top_stores.values),
               color=['red']*5 + ['green']*5)
axes[2,3].set_yticks(range(10))
axes[2,3].set_yticklabels(list(bottom_stores.index) + list(top_stores.index))
axes[2,3].set_title('Top 5 & Bottom 5 Stores')
axes[2,3].set_xlabel('Avg Units Sold')

# 7.13-7.16 Store ranking
axes[3,0].bar(range(len(store_sales)), sorted(store_sales.values), color='steelblue', alpha=0.7)
axes[3,0].set_title('Store Sales Ranking')
axes[3,0].set_xlabel('Store Rank')
axes[3,0].set_ylabel('Avg Units Sold')

# Top 10 stores
top10 = store_sales.nlargest(10)
axes[3,1].barh(range(len(top10)), top10.values, color='green')
axes[3,1].set_yticks(range(len(top10)))
axes[3,1].set_yticklabels(top10.index)
axes[3,1].set_title('Top 10 Stores')
axes[3,1].set_xlabel('Avg Units Sold')

# Store sales distribution
axes[3,2].hist(store_sales.values, bins=10, color='steelblue', edgecolor='black')
axes[3,2].set_title('Store Performance Distribution')
axes[3,2].set_xlabel('Avg Units Sold')

# Coefficient of variation by store
cv_by_store = store_merged.groupby('store_id').agg({'units_sold': 'std'}) / store_sales
axes[3,3].scatter(range(len(cv_by_store)), cv_by_store.values, alpha=0.6, s=100)
axes[3,3].set_title('Sales Variability by Store')
axes[3,3].set_xlabel('Store ID')
axes[3,3].set_ylabel('Coefficient of Variation')

plt.tight_layout()
plt.savefig(output_path / '07_store_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
plot_count += 1

# ===== 8. TIME SERIES & AUTOCORRELATION (10 plots) =====
print("8. Time series & autocorrelation...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Time Series & Autocorrelation Analysis', fontsize=16, fontweight='bold')

# Aggregate daily sales
daily_sales = target_df.groupby('date')['units_sold'].sum().sort_index()

# 8.1 Full time series
axes[0,0].plot(daily_sales.index, daily_sales.values, color='steelblue', alpha=0.7)
axes[0,0].set_title('Daily Total Sales Over Time')
axes[0,0].set_xlabel('Date')
axes[0,0].set_ylabel('Total Units')
axes[0,0].tick_params(axis='x', rotation=45)

# 8.2 Weekly aggregation
weekly_sales = target_df.groupby(target_df['date'].dt.isocalendar().week)['units_sold'].sum()
axes[0,1].plot(weekly_sales.index, weekly_sales.values, marker='o', color='coral', alpha=0.7, markersize=4)
axes[0,1].set_title('Weekly Total Sales')
axes[0,1].set_xlabel('Week')
axes[0,1].set_ylabel('Total Units')

# 8.3 Monthly aggregation
monthly_sales = target_df.groupby('month')['units_sold'].sum()
axes[0,2].bar(monthly_sales.index, monthly_sales.values, color='green', alpha=0.7, edgecolor='black')
axes[0,2].set_title('Monthly Total Sales')
axes[0,2].set_xlabel('Month')
axes[0,2].set_ylabel('Total Units')

# 8.4-8.6 Rolling statistics
window = 7
rolling_mean = daily_sales.rolling(window).mean()
rolling_std = daily_sales.rolling(window).std()

axes[1,0].plot(daily_sales.index, daily_sales.values, alpha=0.5, label='Daily', color='gray')
axes[1,0].plot(rolling_mean.index, rolling_mean.values, color='red', linewidth=2, label=f'{window}-day MA')
axes[1,0].fill_between(rolling_mean.index, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.2, color='red')
axes[1,0].set_title(f'{window}-Day Rolling Average & Std')
axes[1,0].legend()
axes[1,0].tick_params(axis='x', rotation=45)

# 8.7-8.9 Autocorrelation
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(daily_sales, ax=axes[1,1], lags=30)
axes[1,1].set_title('Autocorrelation Function (ACF)')
axes[1,1].set_xlim(0, 30)

# Seasonal pattern (weekly)
weekly_pattern = target_df.groupby(target_df['date'].dt.day_name())['units_sold'].mean().reindex(day_order)
axes[1,2].bar(range(7), weekly_pattern.values, color='purple', alpha=0.7)
axes[1,2].set_xticks(range(7))
axes[1,2].set_xticklabels(['M','T','W','Th','F','Sa','Su'])
axes[1,2].set_title('Seasonal Pattern (Day of Week)')
axes[1,2].set_ylabel('Avg Units')

# 8.10 Decomposition - Trend
axes[2,0].plot(daily_sales.index, rolling_mean.values, color='red', linewidth=2, label='Trend')
axes[2,0].set_title('Trend Component')
axes[2,0].set_xlabel('Date')
axes[2,0].set_ylabel('Units')
axes[2,0].legend()
axes[2,0].tick_params(axis='x', rotation=45)

# 8.11 Seasonality (monthly)
monthly_pattern = target_df.groupby('month')['units_sold'].mean()
axes[2,1].bar(monthly_pattern.index, monthly_pattern.values, color='orange', alpha=0.7)
axes[2,1].set_title('Seasonal Pattern (Monthly)')
axes[2,1].set_xlabel('Month')
axes[2,1].set_ylabel('Avg Units')

# 8.12 Residuals
residuals = daily_sales - rolling_mean
axes[2,2].hist(residuals.dropna(), bins=30, color='teal', edgecolor='black', alpha=0.7)
axes[2,2].set_title('Residuals Distribution')
axes[2,2].set_xlabel('Residual')

plt.tight_layout()
plt.savefig(output_path / '08_timeseries_autocorrelation.png', dpi=150, bbox_inches='tight')
plt.close()
plot_count += 1

# ===== 9. HOURLY & TRANSACTION ANALYSIS (12 plots) =====
print("9. Hourly & transaction analysis...")
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Hourly & Transaction Patterns', fontsize=16, fontweight='bold')

# 9.1 Hourly distribution
hourly_sales = df.groupby('hour')['units_sold'].agg(['mean', 'count', 'sum'])
axes[0,0].bar(hourly_sales.index, hourly_sales['mean'], color='steelblue', alpha=0.7)
axes[0,0].set_title('Avg Units Sold by Hour')
axes[0,0].set_xlabel('Hour')
axes[0,0].set_ylabel('Avg Units')
axes[0,0].set_xlim(-0.5, 23.5)

# 9.2 Transactions by hour
axes[0,1].bar(hourly_sales.index, hourly_sales['count']/1000, color='coral', alpha=0.7)
axes[0,1].set_title('Transaction Count by Hour')
axes[0,1].set_xlabel('Hour')
axes[0,1].set_ylabel('Count (thousands)')
axes[0,1].set_xlim(-0.5, 23.5)

# 9.3 Peak hours
peak_hours = hourly_sales.nlargest(5, 'sum')
axes[0,2].barh(range(len(peak_hours)), peak_hours['sum'].values, color='gold')
axes[0,2].set_yticks(range(len(peak_hours)))
axes[0,2].set_yticklabels(peak_hours.index.astype(int))
axes[0,2].set_title('Top 5 Peak Hours')
axes[0,2].set_xlabel('Total Units')

# 9.4 Revenue distribution
axes[0,3].hist(df['revenue'], bins=50, color='green', edgecolor='black', alpha=0.7)
axes[0,3].set_title('Revenue Distribution')
axes[0,3].set_xlabel('Revenue')

# 9.5-9.8 Hourly by category
for idx, cat in enumerate(categories[:4]):
    ax = axes[1, idx]
    cat_hourly = df[df['category']==cat].groupby('hour')['units_sold'].mean()
    ax.plot(cat_hourly.index, cat_hourly.values, marker='o', linewidth=2, markersize=5, color='teal')
    ax.set_title(f'{cat} - Hourly Pattern')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Avg Units')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 23.5)

# 9.9 Discount impact
discount_impact = df.groupby('discount_applied')['units_sold'].agg(['mean', 'count'])
axes[2,0].scatter(discount_impact.index, discount_impact['mean'], s=discount_impact['count']/100, alpha=0.6, color='red')
axes[2,0].set_title('Discount Applied vs Units Sold')
axes[2,0].set_xlabel('Discount %')
axes[2,0].set_ylabel('Avg Units')

# 9.10 Revenue vs Units correlation
axes[2,1].scatter(df['units_sold'], df['revenue'], alpha=0.3, s=10, color='purple')
axes[2,1].set_title(f'Units vs Revenue (r={df[["units_sold", "revenue"]].corr().iloc[0,1]:.2f})')
axes[2,1].set_xlabel('Units Sold')
axes[2,1].set_ylabel('Revenue')

# 9.11 Payment methods
if 'payment_method' in order.columns:
    payment_dist = order['payment_method'].value_counts()
    axes[2,2].pie(payment_dist.values, labels=payment_dist.index, autopct='%1.1f%%')
    axes[2,2].set_title('Payment Method Distribution')

# 9.12 Customer type (member vs walk-in)
member_pct = (order['customer_id'].notna().sum() / len(order)) * 100
axes[2,3].pie([member_pct, 100-member_pct], labels=['Member', 'Walk-in'], autopct='%1.1f%%', colors=['blue', 'orange'])
axes[2,3].set_title(f'Customer Type (Members: {member_pct:.1f}%)')

plt.tight_layout()
plt.savefig(output_path / '09_hourly_transaction_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
plot_count += 1

# ===== 10. CORRELATION & ADVANCED INSIGHTS (10 plots) =====
print("10. Correlation & advanced insights...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Correlation & Advanced Insights', fontsize=16, fontweight='bold')

# Prepare correlation data
corr_df = target_df[['units_sold', 'is_weekend', 'is_holiday', 'is_payday',
                      'is_school_break', 'is_rainy_season']].copy()
corr_df = corr_df.astype(float)
corr_matrix = corr_df.corr()

# 10.1 Correlation heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0,0],
            square=True, cbar_kws={'shrink': 0.8})
axes[0,0].set_title('Correlation Matrix')

# 10.2 Store neighborhood interaction
nb_stats = target_df.groupby('neighborhood_type')['units_sold'].agg(['mean', 'std', 'min', 'max'])
axes[0,1].errorbar(range(len(nb_stats)), nb_stats['mean'], yerr=nb_stats['std'],
                    fmt='o', markersize=8, capsize=5, color='steelblue')
axes[0,1].set_xticks(range(len(nb_stats)))
axes[0,1].set_xticklabels(nb_stats.index, rotation=45, ha='right')
axes[0,1].set_title('Neighborhood Performance ± Std')
axes[0,1].set_ylabel('Avg Units Sold')

# 10.3 Category by neighborhood heatmap
cat_nb_pivot = target_df.pivot_table(values='units_sold', index='category', columns='neighborhood_type', aggfunc='mean')
sns.heatmap(cat_nb_pivot, annot=True, fmt='.0f', cmap='YlGnBu', ax=axes[0,2], cbar_kws={'shrink': 0.8})
axes[0,2].set_title('Avg Sales: Category × Neighborhood')

# 10.4-10.6 Distribution by day of week
for idx, prop in enumerate(['is_weekend', 'is_holiday', 'is_payday']):
    ax = axes[1, idx]
    prop_data = [target_df[target_df[prop]==0]['units_sold'].values,
                 target_df[target_df[prop]==1]['units_sold'].values]
    bp = ax.boxplot(prop_data, labels=['No', 'Yes'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'salmon']):
        patch.set_facecolor(color)
    ax.set_title(f'{prop.replace("is_", "").title()}')
    ax.set_ylabel('Units Sold')

# 10.7 Customer metrics
if 'customer_id' in order.columns:
    member_stats = order.groupby(order['customer_id'].notna())['order_id'].count()
    ax = axes[1, 2]
    ax.bar(['Walk-in', 'Member'], member_stats.values, color=['orange', 'blue'], alpha=0.7)
    ax.set_title('Transactions: Members vs Walk-ins')
    ax.set_ylabel('Count')

# 10.8 Multi-factor insight: Weekend + Holiday
ax = axes[2, 0]
combo_stats = target_df.groupby(['is_weekend', 'is_holiday'])['units_sold'].mean().unstack()
x = np.arange(2)
width = 0.35
ax.bar(x - width/2, combo_stats[0], width, label='Non-Holiday', color='lightblue')
ax.bar(x + width/2, combo_stats[1], width, label='Holiday', color='salmon')
ax.set_xlabel('Weekday (0) vs Weekend (1)')
ax.set_ylabel('Avg Units')
ax.set_title('Weekend × Holiday Interaction')
ax.set_xticks(x)
ax.legend()

# 10.9 Variety index (unique categories per store)
variety_by_store = target_df.groupby('store_id')['category'].nunique()
ax = axes[2, 1]
ax.hist(variety_by_store, bins=5, color='purple', edgecolor='black', alpha=0.7)
ax.set_title('Category Variety Distribution by Store')
ax.set_xlabel('Unique Categories')
ax.set_ylabel('# of Stores')

# 10.10 Target distribution by quantile
ax = axes[2, 2]
quantiles = pd.qcut(target_df['units_sold'], q=5, duplicates='drop')
quantile_counts = quantiles.value_counts().sort_index()
colors_grad = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(quantile_counts)))
ax.bar(range(len(quantile_counts)), quantile_counts.values, color=colors_grad, edgecolor='black')
ax.set_xticks(range(len(quantile_counts)))
ax.set_xticklabels([f'Q{i+1}' for i in range(len(quantile_counts))])
ax.set_title('Distribution: Units Sold by Quantile')
ax.set_ylabel('Count')

plt.tight_layout()
plt.savefig(output_path / '10_correlation_advanced_insights.png', dpi=150, bbox_inches='tight')
plt.close()
plot_count += 1

print(f"\n{'='*60}")
print(f"✓ EDA GENERATION COMPLETE!")
print(f"{'='*60}")
print(f"Total visualizations created: {plot_count * 12 + 28} plots")
print(f"Total figures saved: {plot_count}")
print(f"\nOutput directory: {output_path}")
print(f"\nFiles generated:")
for i, file in enumerate(sorted(output_path.glob('*.png')), 1):
    print(f"  {i}. {file.name}")
