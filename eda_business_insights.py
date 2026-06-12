import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
data_path = Path(r'c:\Users\Panuwit\Downloads\edacoffee\super-ai-engineer-season-6-coffee-chain-hackathon\train')
output_path = Path(r'c:\Users\Panuwit\Downloads\edacoffee\eda_business_insights')
output_path.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams['figure.dpi'] = 100

# Load data
print("Loading data...")
transaction = pd.read_csv(data_path / 'TRANSACTION.csv')
order = pd.read_csv(data_path / 'ORDER.csv')
product = pd.read_csv(data_path / 'PRODUCT.csv')
store = pd.read_csv(data_path / 'STORE.csv')
date_dim = pd.read_csv(data_path / 'DATE_DIM.csv')
customer = pd.read_csv(data_path / 'CUSTOMER.csv')
promotion = pd.read_csv(data_path / 'PROMOTION.csv')
local_event = pd.read_csv(data_path / 'LOCAL_EVENT.csv')

# Prepare data
order['date'] = pd.to_datetime(order['date'])
date_dim['date'] = pd.to_datetime(date_dim['date'])
promotion['start_date'] = pd.to_datetime(promotion['start_date'])
promotion['end_date'] = pd.to_datetime(promotion['end_date'])
local_event['date'] = pd.to_datetime(local_event['date'])

# Build main dataframe
df = (transaction
      .merge(order[['order_id', 'store_id', 'date', 'hour', 'customer_id', 'payment_method']], on='order_id')
      .merge(product[['product_id', 'category', 'base_price']], on='product_id')
      .merge(store[['store_id', 'neighborhood_type', 'seating_capacity', 'staff_count', 'has_drive_through']], on='store_id')
      .merge(date_dim[['date', 'is_weekend', 'is_holiday', 'is_payday', 'is_school_break', 'is_rainy_season']], on='date'))

df['day_of_week'] = df['date'].dt.day_name()
df['month'] = df['date'].dt.month

# ===== BUSINESS INSIGHT 1: REVENUE CONTRIBUTION BY CATEGORY =====
print("\n1. Revenue Contribution by Category (Pareto Analysis)...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Business Insight 1: Revenue by Category (Pareto Analysis)', fontsize=14, fontweight='bold')

# Revenue by category
revenue_by_cat = df.groupby('category')['revenue'].sum().sort_values(ascending=False)
cumsum = revenue_by_cat.cumsum() / revenue_by_cat.sum() * 100

colors = ['#d62728' if x > 80 else '#ff7f0e' if x > 50 else '#2ca02c' for x in cumsum.values]
axes[0].bar(range(len(revenue_by_cat)), revenue_by_cat.values/1e6, color=colors, alpha=0.8, edgecolor='black')
axes[0].set_xticks(range(len(revenue_by_cat)))
axes[0].set_xticklabels(revenue_by_cat.index, rotation=45, ha='right')
axes[0].set_ylabel('Revenue (Million Baht)', fontsize=11)
axes[0].set_title('Total Revenue by Category')
axes[0].grid(axis='y', alpha=0.3)

# Cumulative % contribution
axes[1].plot(range(len(cumsum)), cumsum.values, marker='o', linewidth=3, markersize=8, color='darkblue', label='Cumulative %')
axes[1].axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% Threshold')
axes[1].fill_between(range(len(cumsum)), cumsum.values, alpha=0.2, color='darkblue')
axes[1].set_ylabel('Cumulative % of Revenue', fontsize=11)
axes[1].set_xlabel('Category (Ranked)', fontsize=11)
axes[1].set_title('Pareto Curve: Revenue Concentration')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 105)

plt.tight_layout()
plt.savefig(output_path / 'B01_revenue_pareto.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== BUSINESS INSIGHT 2: CUSTOMER LOYALTY & MEMBER VALUE =====
print("2. Customer Loyalty & Member Value...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Business Insight 2: Customer Loyalty & Member Value Analysis', fontsize=14, fontweight='bold')

# Member vs Walk-in transactions
member_orders = order[order['customer_id'].notna()]
walkin_orders = order[order['customer_id'].isna()]

member_pct = len(member_orders) / len(order) * 100
axes[0].pie([member_pct, 100-member_pct],
           labels=[f'Members\n({member_pct:.1f}%)', f'Walk-ins\n({100-member_pct:.1f}%)'],
           colors=['#2ca02c', '#d62728'], autopct='%1.1f%%', startangle=90,
           explode=(0.05, 0))
axes[0].set_title('Transaction Mix: Members vs Walk-ins')

# Revenue per customer type
member_rev = transaction.merge(member_orders[['order_id', 'customer_id']], on='order_id')['revenue'].sum()
walkin_rev = transaction.merge(walkin_orders[['order_id']], on='order_id')['revenue'].sum()

rev_by_type = [member_rev/1e6, walkin_rev/1e6]
axes[1].bar(['Members', 'Walk-ins'], rev_by_type, color=['#2ca02c', '#d62728'], alpha=0.8, edgecolor='black', width=0.5)
axes[1].set_ylabel('Revenue (Million Baht)', fontsize=11)
axes[1].set_title('Revenue Contribution by Customer Type')
axes[1].grid(axis='y', alpha=0.3)

# Member purchase frequency
member_freq = member_orders.groupby('customer_id')['order_id'].count()
axes[2].hist(member_freq, bins=30, color='#2ca02c', alpha=0.7, edgecolor='black')
axes[2].axvline(member_freq.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {member_freq.mean():.1f}')
axes[2].axvline(member_freq.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {member_freq.median():.0f}')
axes[2].set_xlabel('Number of Transactions per Member', fontsize=11)
axes[2].set_ylabel('Number of Members', fontsize=11)
axes[2].set_title('Member Purchase Frequency Distribution')
axes[2].legend()
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / 'B02_customer_loyalty.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== BUSINESS INSIGHT 3: STORE PROFITABILITY MATRIX =====
print("3. Store Profitability Matrix...")
fig, ax = plt.subplots(figsize=(12, 8))

store_metrics = df.groupby('store_id').agg({
    'revenue': 'sum',
    'units_sold': 'sum',
    'customer_id': 'nunique'
}).reset_index()

store_metrics['avg_transaction_value'] = store_metrics['revenue'] / store_metrics['customer_id']
store_metrics = store_metrics.merge(store[['store_id', 'neighborhood_type']], on='store_id')

# Create bubble chart
colors_map = {'university': '#1f77b4', 'tourist': '#ff7f0e', 'hospital': '#2ca02c',
              'gas_station': '#d62728', 'urban_residential': '#9467bd', 'transit': '#8c564b',
              'mall': '#e377c2', 'office': '#7f7f7f'}
colors = [colors_map.get(nb, '#999999') for nb in store_metrics['neighborhood_type']]

scatter = ax.scatter(store_metrics['units_sold']/1000,
                     store_metrics['avg_transaction_value'],
                     s=store_metrics['revenue']/5000,
                     c=colors,
                     alpha=0.6,
                     edgecolors='black',
                     linewidth=1.5)

ax.set_xlabel('Sales Volume (Thousands Units)', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Transaction Value (Baht)', fontsize=12, fontweight='bold')
ax.set_title('Store Profitability Matrix\n(Bubble size = Total Revenue)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add store labels
for idx, row in store_metrics.iterrows():
    ax.annotate(f"S{row['store_id']}",
               (row['units_sold']/1000, row['avg_transaction_value']),
               fontsize=8, ha='center', va='center')

# Create legend for neighborhoods
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors_map[nb], edgecolor='black', label=nb)
                   for nb in store_metrics['neighborhood_type'].unique()]
ax.legend(handles=legend_elements, loc='best', title='Neighborhood', fontsize=9)

plt.tight_layout()
plt.savefig(output_path / 'B03_store_profitability.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== BUSINESS INSIGHT 4: PROMOTIONAL EFFECTIVENESS ROI =====
print("4. Promotional Effectiveness & ROI...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Business Insight 4: Promotional Effectiveness & ROI', fontsize=14, fontweight='bold')

# Build promo periods
promo_periods = []
for _, prow in promotion.iterrows():
    date_range = pd.date_range(start=prow['start_date'], end=prow['end_date'], freq='D')
    for d in date_range:
        promo_periods.append(d)
promo_dates = set(promo_periods)

# Sales with/without promo
df_with_promo_flag = df.copy()
df_with_promo_flag['has_promo'] = df_with_promo_flag['date'].isin(promo_dates)

promo_daily = df_with_promo_flag.groupby(['date', 'has_promo'])['revenue'].sum().reset_index()
daily_avg = promo_daily.groupby('has_promo')['revenue'].mean()

axes[0].bar(['No Promotion', 'With Promotion'], daily_avg.values/1e6,
           color=['#d62728', '#2ca02c'], alpha=0.8, edgecolor='black', width=0.5)
axes[0].set_ylabel('Daily Revenue (Million Baht)', fontsize=11)
axes[0].set_title('Average Daily Revenue: Promo Impact')
axes[0].grid(axis='y', alpha=0.3)

# Promo type effectiveness
promo_type_effect = promotion['promo_type'].value_counts()
axes[1].barh(promo_type_effect.index, promo_type_effect.values, color='steelblue', alpha=0.8, edgecolor='black')
axes[1].set_xlabel('Number of Promotions', fontsize=11)
axes[1].set_title('Promotion Type Frequency')
axes[1].grid(axis='x', alpha=0.3)

# Discount depth analysis
discount_analysis = promotion.groupby(pd.cut(promotion['discount_pct'], bins=5))['discount_pct'].agg(['count', 'mean'])
discount_bins = [f"{int(x.left)}-{int(x.right)}%" for x in discount_analysis.index]
axes[2].bar(range(len(discount_bins)), discount_analysis['count'].values,
           color='coral', alpha=0.8, edgecolor='black')
axes[2].set_xticks(range(len(discount_bins)))
axes[2].set_xticklabels(discount_bins, rotation=45, ha='right')
axes[2].set_ylabel('Count of Promotions', fontsize=11)
axes[2].set_title('Discount Depth Distribution')
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / 'B04_promo_effectiveness.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== BUSINESS INSIGHT 5: PEAK DEMAND & OPERATIONAL PLANNING =====
print("5. Peak Demand & Operational Planning...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Business Insight 5: Peak Demand & Operational Planning', fontsize=14, fontweight='bold')

# Peak hours
hourly_revenue = df.groupby('hour')['revenue'].sum()
hourly_volume = df.groupby('hour')['units_sold'].sum()

axes[0,0].bar(hourly_revenue.index, hourly_revenue.values/1e6, color='steelblue', alpha=0.8, edgecolor='black')
axes[0,0].set_xlabel('Hour of Day', fontsize=11)
axes[0,0].set_ylabel('Revenue (Million Baht)', fontsize=11)
axes[0,0].set_title('Hourly Revenue Distribution')
axes[0,0].grid(axis='y', alpha=0.3)
axes[0,0].set_xlim(-0.5, 23.5)

# Workload distribution
peak_hours = hourly_volume[hourly_volume > hourly_volume.quantile(0.75)].index.tolist()
off_peak_hours = hourly_volume[hourly_volume <= hourly_volume.quantile(0.25)].index.tolist()

axes[0,1].bar(['Peak Hours\n(Top 25%)', 'Off-Peak Hours\n(Bottom 25%)', 'Mid Hours'],
             [hourly_volume[peak_hours].sum(), hourly_volume[off_peak_hours].sum(),
              hourly_volume[~hourly_volume.index.isin(peak_hours + off_peak_hours)].sum()],
             color=['#d62728', '#2ca02c', '#ff7f0e'], alpha=0.8, edgecolor='black')
axes[0,1].set_ylabel('Total Units Sold', fontsize=11)
axes[0,1].set_title('Workload Distribution Across Periods')
axes[0,1].grid(axis='y', alpha=0.3)

# Day of week patterns for staffing
dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_revenue = df.groupby(df['date'].dt.day_name())['revenue'].mean().reindex(dow_order)
dow_count = df.groupby(df['date'].dt.day_name())['order_id'].count().reindex(dow_order)

axes[1,0].bar(range(7), dow_revenue.values/1e6, color='coral', alpha=0.8, edgecolor='black')
axes[1,0].set_xticks(range(7))
axes[1,0].set_xticklabels(['M','T','W','Th','F','Sa','Su'])
axes[1,0].set_ylabel('Avg Daily Revenue (Million Baht)', fontsize=11)
axes[1,0].set_title('Revenue by Day of Week (Staffing Guide)')
axes[1,0].grid(axis='y', alpha=0.3)

# Monthly seasonality for planning
monthly_revenue = df.groupby('month')['revenue'].mean()
axes[1,1].plot(monthly_revenue.index, monthly_revenue.values/1e6, marker='o', linewidth=3,
              markersize=10, color='darkgreen')
axes[1,1].fill_between(monthly_revenue.index, monthly_revenue.values/1e6, alpha=0.3, color='darkgreen')
axes[1,1].set_xlabel('Month', fontsize=11)
axes[1,1].set_ylabel('Avg Daily Revenue (Million Baht)', fontsize=11)
axes[1,1].set_title('Seasonal Revenue Pattern')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / 'B05_peak_demand_ops.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== BUSINESS INSIGHT 6: DISCOUNT STRATEGY ANALYSIS =====
print("6. Discount Strategy Impact...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Business Insight 6: Discount Strategy & Price Elasticity', fontsize=14, fontweight='bold')

# Discount distribution across categories
discount_by_cat = df.groupby('category')['discount_applied'].agg(['mean', 'max', 'min', 'count'])
discount_by_cat = discount_by_cat.sort_values('mean', ascending=False)

axes[0,0].barh(discount_by_cat.index, discount_by_cat['mean'], color='orange', alpha=0.8, edgecolor='black')
axes[0,0].set_xlabel('Average Discount Applied (%)', fontsize=11)
axes[0,0].set_title('Discount Strategy by Category')
axes[0,0].grid(axis='x', alpha=0.3)

# Discount vs Units Sold relationship
discount_bins = pd.cut(df['discount_applied'], bins=[0, 10, 20, 30, 40, 100])
discount_units = df.groupby(discount_bins)['units_sold'].agg(['mean', 'count'])

bin_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '>40%']
axes[0,1].bar(bin_labels, discount_units['mean'].values,
             color='lightcoral', alpha=0.8, edgecolor='black')
axes[0,1].set_ylabel('Avg Units Sold per Transaction', fontsize=11)
axes[0,1].set_title('Elasticity: Discount vs Units')
axes[0,1].grid(axis='y', alpha=0.3)

# Discount vs Revenue
revenue_bins = pd.cut(df['discount_applied'], bins=[0, 10, 20, 30, 40, 100])
revenue_impact = df.groupby(revenue_bins)['revenue'].agg(['mean', 'sum'])

axes[1,0].bar(bin_labels, revenue_impact['mean'].values,
             color='lightblue', alpha=0.8, edgecolor='black')
axes[1,0].set_ylabel('Avg Revenue per Transaction (Baht)', fontsize=11)
axes[1,0].set_title('Revenue Impact of Discount Levels')
axes[1,0].grid(axis='y', alpha=0.3)

# Profit margin perspective (estimate)
df['estimated_cost'] = df['units_sold'] * (df['base_price'] * 0.4)  # Assume 60% margin
df['gross_profit'] = df['revenue'] - df['estimated_cost']

margin_by_discount = df.groupby(discount_bins)['gross_profit'].mean()
axes[1,1].bar(bin_labels, margin_by_discount.values,
             color='green', alpha=0.7, edgecolor='black')
axes[1,1].set_ylabel('Estimated Gross Profit (Baht)', fontsize=11)
axes[1,1].set_title('Profit Impact of Discounting')
axes[1,1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / 'B06_discount_strategy.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== BUSINESS INSIGHT 7: EVENT MARKETING IMPACT =====
print("7. Event Marketing Impact...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Business Insight 7: Local Events Marketing Impact', fontsize=14, fontweight='bold')

# Build event flag
event_dates = set(local_event['date'].unique())
df['has_event'] = df['date'].isin(event_dates)

event_impact = df.groupby('has_event').agg({
    'revenue': ['sum', 'mean', 'count'],
    'units_sold': ['sum', 'mean']
}).round(0)

# Events vs non-events
event_labels = ['No Event', 'With Event']
event_revenue = df.groupby('has_event')['revenue'].mean()
axes[0,0].bar(event_labels, event_revenue.values/1e6,
             color=['#d62728', '#2ca02c'], alpha=0.8, edgecolor='black', width=0.5)
axes[0,0].set_ylabel('Avg Daily Revenue (Million Baht)', fontsize=11)
axes[0,0].set_title('Event Impact on Revenue')
axes[0,0].grid(axis='y', alpha=0.3)

# Event type breakdown
event_type_freq = local_event['event_type'].value_counts()
axes[0,1].barh(event_type_freq.index, event_type_freq.values,
              color='steelblue', alpha=0.8, edgecolor='black')
axes[0,1].set_xlabel('Number of Events', fontsize=11)
axes[0,1].set_title('Event Types Held')
axes[0,1].grid(axis='x', alpha=0.3)

# Store with most events
store_events = local_event['store_id'].value_counts().head(10)
axes[1,0].barh(store_events.index.astype(str), store_events.values,
              color='coral', alpha=0.8, edgecolor='black')
axes[1,0].set_xlabel('Number of Events', fontsize=11)
axes[1,0].set_title('Top 10 Stores by Event Count')
axes[1,0].grid(axis='x', alpha=0.3)

# Event ROI (estimated)
event_incremental = df.groupby('has_event')['units_sold'].mean()
lift = (event_incremental[True] - event_incremental[False]) / event_incremental[False] * 100

axes[1,1].bar(['Unit Sales\nLift', 'Revenue\nLift'], [lift, lift * 0.8],
             color=['#2ca02c', '#ff7f0e'], alpha=0.8, edgecolor='black', width=0.5)
axes[1,1].set_ylabel('% Increase', fontsize=11)
axes[1,1].set_title(f'Event Marketing ROI (Est. {lift:.1f}% lift)')
axes[1,1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / 'B07_event_marketing.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== BUSINESS INSIGHT 8: PRODUCT MIX & CONTRIBUTION =====
print("8. Product Mix & Contribution Analysis...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Business Insight 8: Product Mix & Margin Contribution', fontsize=14, fontweight='bold')

# Units vs Revenue contribution
product_mix = df.groupby('category').agg({
    'units_sold': 'sum',
    'revenue': 'sum',
    'order_id': 'count'
}).round(0)
product_mix['avg_price'] = product_mix['revenue'] / product_mix['units_sold']
product_mix = product_mix.sort_values('revenue', ascending=False)

# Units contribution
units_pct = product_mix['units_sold'] / product_mix['units_sold'].sum() * 100
axes[0].pie(units_pct.values, labels=product_mix.index, autopct='%1.1f%%',
           colors=plt.cm.Set3(range(len(product_mix))))
axes[0].set_title('Units Sold by Category (%)')

# Revenue contribution
revenue_pct = product_mix['revenue'] / product_mix['revenue'].sum() * 100
axes[1].pie(revenue_pct.values, labels=product_mix.index, autopct='%1.1f%%',
           colors=plt.cm.Set2(range(len(product_mix))))
axes[1].set_title('Revenue by Category (%)')

# Avg price point positioning
axes[2].barh(product_mix.index, product_mix['avg_price'].values,
            color='mediumpurple', alpha=0.8, edgecolor='black')
axes[2].set_xlabel('Average Price per Unit (Baht)', fontsize=11)
axes[2].set_title('Price Point by Category')
axes[2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / 'B08_product_mix.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== BUSINESS INSIGHT 9: PAYMENT METHOD ANALYSIS =====
print("9. Payment Method Preferences...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Business Insight 9: Payment Method & Customer Behavior', fontsize=14, fontweight='bold')

# Payment method distribution
payment_dist = order['payment_method'].value_counts()
colors_payment = plt.cm.Pastel1(range(len(payment_dist)))

axes[0].pie(payment_dist.values, labels=payment_dist.index, autopct='%1.1f%%',
           colors=colors_payment)
axes[0].set_title('Payment Method Preference')

# Revenue by payment method
payment_revenue = order.merge(transaction[['order_id', 'revenue']], on='order_id').groupby('payment_method')['revenue'].sum()
axes[1].bar(payment_revenue.index, payment_revenue.values/1e6,
           color=colors_payment, alpha=0.8, edgecolor='black')
axes[1].set_ylabel('Revenue (Million Baht)', fontsize=11)
axes[1].set_title('Revenue by Payment Method')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

# Avg transaction by payment
payment_avg = order.merge(transaction[['order_id', 'revenue']], on='order_id').groupby('payment_method')['revenue'].mean()
axes[2].bar(payment_avg.index, payment_avg.values,
           color=colors_payment, alpha=0.8, edgecolor='black')
axes[2].set_ylabel('Avg Transaction Value (Baht)', fontsize=11)
axes[2].set_title('Avg Basket Size by Payment Method')
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / 'B09_payment_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== BUSINESS INSIGHT 10: BUSINESS PERFORMANCE SCORECARD =====
print("10. Business Performance Scorecard...")
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

fig.suptitle('Business Insight 10: Executive Performance Scorecard', fontsize=16, fontweight='bold')

# KPI metrics
total_revenue = df['revenue'].sum() / 1e9
avg_daily_revenue = df.groupby('date')['revenue'].sum().mean() / 1e6
total_units = df['units_sold'].sum() / 1e6
avg_transaction = df['revenue'].mean() / 1000
member_pct = (order['customer_id'].notna().sum() / len(order) * 100)
growth_indicator = ((df[df['month'] > 9]['revenue'].sum() - df[df['month'] <= 9]['revenue'].sum()) /
                   df[df['month'] <= 9]['revenue'].sum() * 100)

# 1. Total Revenue
ax1 = fig.add_subplot(gs[0, 0])
ax1.text(0.5, 0.6, f'{total_revenue:.2f}B', ha='center', va='center', fontsize=28, fontweight='bold', transform=ax1.transAxes)
ax1.text(0.5, 0.2, 'Total Revenue', ha='center', va='center', fontsize=12, transform=ax1.transAxes)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis('off')
ax1.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='steelblue', linewidth=3, transform=ax1.transAxes))

# 2. Avg Daily Revenue
ax2 = fig.add_subplot(gs[0, 1])
ax2.text(0.5, 0.6, f'{avg_daily_revenue:.2f}M', ha='center', va='center', fontsize=24, fontweight='bold', transform=ax2.transAxes)
ax2.text(0.5, 0.2, 'Avg Daily Revenue', ha='center', va='center', fontsize=11, transform=ax2.transAxes)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.axis('off')
ax2.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='coral', linewidth=3, transform=ax2.transAxes))

# 3. Total Units
ax3 = fig.add_subplot(gs[0, 2])
ax3.text(0.5, 0.6, f'{total_units:.2f}M', ha='center', va='center', fontsize=24, fontweight='bold', transform=ax3.transAxes)
ax3.text(0.5, 0.2, 'Total Units Sold', ha='center', va='center', fontsize=11, transform=ax3.transAxes)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.axis('off')
ax3.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='green', linewidth=3, transform=ax3.transAxes))

# 4. Avg Transaction
ax4 = fig.add_subplot(gs[1, 0])
ax4.text(0.5, 0.6, f'{avg_transaction:.0f} ฿', ha='center', va='center', fontsize=24, fontweight='bold', transform=ax4.transAxes)
ax4.text(0.5, 0.2, 'Avg Transaction Value', ha='center', va='center', fontsize=11, transform=ax4.transAxes)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')
ax4.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='purple', linewidth=3, transform=ax4.transAxes))

# 5. Member Percentage
ax5 = fig.add_subplot(gs[1, 1])
ax5.text(0.5, 0.6, f'{member_pct:.1f}%', ha='center', va='center', fontsize=24, fontweight='bold', transform=ax5.transAxes)
ax5.text(0.5, 0.2, 'Member Transactions', ha='center', va='center', fontsize=11, transform=ax5.transAxes)
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis('off')
color_member = '#2ca02c' if member_pct > 30 else '#ff7f0e' if member_pct > 20 else '#d62728'
ax5.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor=color_member, linewidth=3, transform=ax5.transAxes))

# 6. Growth Indicator
ax6 = fig.add_subplot(gs[1, 2])
growth_symbol = '↑' if growth_indicator > 0 else '↓'
growth_color = '#2ca02c' if growth_indicator > 0 else '#d62728'
ax6.text(0.5, 0.65, f'{growth_symbol}', ha='center', va='center', fontsize=40, color=growth_color, transform=ax6.transAxes)
ax6.text(0.5, 0.35, f'{abs(growth_indicator):.1f}%', ha='center', va='center', fontsize=18, fontweight='bold', transform=ax6.transAxes)
ax6.text(0.5, 0.05, 'Growth vs Early Year', ha='center', va='center', fontsize=10, transform=ax6.transAxes)
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)
ax6.axis('off')
ax6.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor=growth_color, linewidth=3, transform=ax6.transAxes))

# 7-9. Mini charts
ax7 = fig.add_subplot(gs[2, :])
monthly_rev = df.groupby('month')['revenue'].sum() / 1e6
ax7.plot(monthly_rev.index, monthly_rev.values, marker='o', linewidth=3, markersize=10,
        color='darkblue', label='Monthly Revenue')
ax7.fill_between(monthly_rev.index, monthly_rev.values, alpha=0.2, color='darkblue')
ax7.set_xlabel('Month', fontsize=11)
ax7.set_ylabel('Revenue (Million Baht)', fontsize=11)
ax7.set_title('Monthly Revenue Trend', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.legend(loc='upper left')

plt.savefig(output_path / 'B10_performance_scorecard.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n{'='*60}")
print(f"[DONE] BUSINESS INSIGHTS EDA COMPLETE!")
print(f"{'='*60}")
print(f"Total Business Insights Generated: 10 figures")
print(f"\nOutput directory: {output_path}")
print(f"\nBusiness Visualizations Generated:")
files = sorted(output_path.glob('*.png'))
for i, file in enumerate(files, 1):
    print(f"  {i}. {file.name}")

print(f"\n{'='*60}")
print(f"Key Business Dimensions Covered:")
print(f"  1. Revenue Concentration (Pareto Analysis)")
print(f"  2. Customer Loyalty & Member Value")
print(f"  3. Store Profitability Matrix")
print(f"  4. Promotional Effectiveness & ROI")
print(f"  5. Peak Demand & Operational Planning")
print(f"  6. Discount Strategy & Price Elasticity")
print(f"  7. Local Events Marketing Impact")
print(f"  8. Product Mix & Margin Contribution")
print(f"  9. Payment Method & Customer Behavior")
print(f"  10. Executive Performance Scorecard")
print(f"{'='*60}")
