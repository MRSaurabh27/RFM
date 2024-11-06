import pandas as pd
import numpy as np
import joblib
import plotnine as pn
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load data
cdnow_raw_df = pd.read_csv('cdnow.csv', names=["customer_id", "date", "quantity", "price"])
print(cdnow_raw_df.head())

# Convert 'date' column to datetime
cdnow_raw_df['date'] = pd.to_datetime(cdnow_raw_df['date'], errors='coerce')

# Step 3: Find minimum and maximum dates
min_date = cdnow_raw_df['date'].min()
max_date = cdnow_raw_df['date'].max()
print("\nMinimum date of first purchase:", min_date)
print("\nMaximum date of first purchase:", max_date)

# Set 'date' as the index and ensure 'price' is numeric
cdnow_raw_df.set_index('date', inplace=True)
cdnow_raw_df['price'] = pd.to_numeric(cdnow_raw_df['price'], errors='coerce')

# Plot the monthly sum of prices
plt.figure(figsize=(12, 6))
cdnow_raw_df[['price']].resample('MS').sum().plot()
plt.title('Monthly Sum of Prices')
plt.xlabel('Date')
plt.ylabel('Sum of Prices')
plt.grid(True)
plt.show()

# Get unique customer IDs and select first 20
ids = cdnow_raw_df['customer_id'].unique()
ids_selected = ids[:20]

# Filter for selected customers and group by 'customer_id' and 'date'
cdnow_cust_id_subset_df = cdnow_raw_df \
    .loc[cdnow_raw_df['customer_id'].isin(ids_selected)] \
    .groupby(['customer_id', 'date']) \
    .sum() \
    .reset_index()
print(cdnow_cust_id_subset_df.head())

# Create the plot using plotnine
plot = (
    pn.ggplot(data=cdnow_cust_id_subset_df) +
    pn.aes(x='date', y='price', group='customer_id') +
    pn.geom_line() +
    pn.geom_point() +
    pn.facet_wrap('~customer_id') +
    pn.scale_x_date(date_breaks="1 year", date_labels="%Y")
)

# Display the plot in plotnine (this will save and open the plot as an image)
plot.save("customer_price_plot.png")
print("Plot saved as 'customer_price_plot.png'.")

# Time-based filtering
n_days = 90
max_date = cdnow_raw_df.index.max()
cutoff = max_date - pd.to_timedelta(n_days, unit="d")
print("Cutoff date:", cutoff)

# Split data based on the cutoff date
temporal_in_df = cdnow_raw_df[cdnow_raw_df.index <= cutoff]
temporal_out_df = cdnow_raw_df[cdnow_raw_df.index > cutoff]

# Create targets_df from temporal_out_df
targets_df = temporal_out_df \
    .drop('quantity', axis=1) \
    .groupby('customer_id') \
    .agg({'price': 'sum'}) \
    .rename(columns={'price': 'spend_90_total'}) \
    .reset_index()

print("\nTarget DataFrame:\n", targets_df.head())
