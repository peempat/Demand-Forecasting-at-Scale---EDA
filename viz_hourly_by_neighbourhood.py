import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Data path
data_path = Path(r'c:\Users\Panuwit\Downloads\edacoffee\super-ai-engineer-season-6-coffee-chain-hackathon\train')

# Load data
print("Loading data...")
transaction = pd.read_csv(data_path / 'TRANSACTION.csv')
order = pd.read_csv(data_path / 'ORDER.csv')
store = pd.read_csv(data_path / 'STORE.csv')

# Convert date columns
order['date'] = pd.to_datetime(order['date'])

# Merge data
print("Merging data...")
df = (transaction
      .merge(order[['order_id', 'store_id', 'date', 'hour']], on='order_id')
      .merge(store[['store_id', 'neighborhood_type']], on='store_id'))

# Create day of week
df['day_of_week'] = df['date'].dt.day_name()
df['day_num'] = df['date'].dt.dayofweek

# Order days properly
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)

# Aggregate: units_sold by hour, neighbourhood, day_of_week
agg_data = (df.groupby(['hour', 'neighborhood_type', 'day_of_week'])
            .agg({'units_sold': ['mean', 'median', 'std', 'min', 'max']})
            .reset_index())

agg_data.columns = ['hour', 'neighborhood_type', 'day_of_week', 'mean', 'median', 'std', 'min', 'max']
agg_data['std'] = agg_data['std'].fillna(0)

print(f"Aggregated data shape: {agg_data.shape}")
print(f"Neighborhoods: {agg_data['neighborhood_type'].unique()}")

# Create candlestick plot (OHLC-style using min/median/max)
# Using median as "close", mean as reference, min/max as wicks
fig, axes = plt.subplots(
    nrows=len(day_order),
    ncols=1,
    figsize=(16, 14),
    sharex=True
)

neighborhoods = sorted(agg_data['neighborhood_type'].unique())
colors = sns.color_palette("husl", len(neighborhoods))
color_map = {nb: colors[i] for i, nb in enumerate(neighborhoods)}

for ax_idx, day in enumerate(day_order):
    ax = axes[ax_idx]
    day_data = agg_data[agg_data['day_of_week'] == day].copy()

    if day_data.empty:
        ax.text(0.5, 0.5, f'No data for {day}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{day} - No Data', fontsize=12, fontweight='bold')
        continue

    # Plot candlestick for each neighborhood
    for nb in neighborhoods:
        nb_data = day_data[day_data['neighborhood_type'] == nb].sort_values('hour')

        if nb_data.empty:
            continue

        # Plot wicks (min to max)
        ax.vlines(nb_data['hour'], nb_data['min'], nb_data['max'],
                 colors=color_map[nb], alpha=0.3, linewidth=1, label=f'{nb} range')

        # Plot body (median as center, std as width approximation)
        for _, row in nb_data.iterrows():
            # Candlestick body
            body_height = max(row['std'], 1)  # Minimum height for visibility
            ax.bar(row['hour'] + (neighborhoods.index(nb) * 0.08 - 0.3),
                  body_height,
                  width=0.06,
                  bottom=row['median'] - body_height/2,
                  color=color_map[nb],
                  alpha=0.7,
                  edgecolor='black',
                  linewidth=0.5)

        # Plot mean line
        ax.plot(nb_data['hour'], nb_data['mean'], marker='o',
               color=color_map[nb], alpha=0.8, linewidth=2,
               markersize=4, label=f'{nb} mean')

    ax.set_ylabel('Units Sold', fontsize=10)
    ax.set_title(f'{day}', fontsize=12, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 24)
    ax.set_xticks(range(0, 24, 2))

axes[-1].set_xlabel('Hour of Day', fontsize=11)

# Create legend
handles = [plt.Line2D([0], [0], color=color_map[nb], lw=2, marker='o', markersize=6)
          for nb in neighborhoods]
fig.legend(handles, neighborhoods, loc='upper center', bbox_to_anchor=(0.5, 0.98),
          ncol=4, fontsize=10, title='Neighborhood Type')

plt.suptitle('Hourly Unit Sales by Neighborhood - Split by Day of Week\n(Candlestick: Line=min/max, Bar=std around median, Dots=mean)',
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(r'c:\Users\Panuwit\Downloads\edacoffee\hourly_sales_by_neighbourhood.png', dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: hourly_sales_by_neighbourhood.png")
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
summary = (agg_data.groupby('neighborhood_type')
          .agg({
              'mean': ['min', 'mean', 'max'],
              'median': ['min', 'mean', 'max'],
              'max': 'max'
          })
          .round(2))
print(summary)
