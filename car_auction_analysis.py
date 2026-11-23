import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf

# 1. Load the datasets
sales = pd.read_csv('used_car_sales.csv')
auction = pd.read_csv('auction_lots.csv')

# 2. Split used_car_sales.csv: 70% training (168 cars), 30% testing (72 cars)
train, test = train_test_split(sales, test_size=0.30, random_state=42)

# 3. Fit initial full model vs. reduced model (v1)
full_model = smf.ols(
    'Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident) + Num_Owners',
    data=train,
).fit()
print('=== FULL MODEL SUMMARY ===')
print(full_model.summary())

# Num_Owners has a high p-value (p = 0.865) and adds no predictive power.
# Drop Num_Owners and refit as v1:
model_v1 = smf.ols(
    'Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident)', data=train
).fit()
print('\n=== MODEL V1 SUMMARY (DROPPED NUM_OWNERS) ===')
print(model_v1.summary())

# 4. Evaluate on unseen test data (72 cars)
y_test = test['Sale_Price']
baseline_pred_value = train['Sale_Price'].mean()
y_pred_base = np.full_like(y_test, baseline_pred_value, dtype=float)
y_pred_v1 = model_v1.predict(test)

print('\n=== TEST SET PERFORMANCE (72 UNSEEN CARS) ===')
print(f'Baseline Average Prediction: ${baseline_pred_value:,.2f}')
print(
    f'Baseline -> R2: {r2_score(y_test, y_pred_base):.3f} | RMSE:'
    f' ${root_mean_squared_error(y_test, y_pred_base):,.0f}'
)
print(
    f'Model v1 -> R2: {r2_score(y_test, y_pred_v1):.3f} | RMSE:'
    f' ${root_mean_squared_error(y_test, y_pred_v1):,.0f}'
)


# 5. Dealer Decision Rule Function
def should_buy_car(
    mileage_k, age_years, horsepower, accident, auction_price, model
):
  """Returns 'buy' if predicted retail price exceeds auction price + $1,500 prep cost."""
  car_features = pd.DataFrame({
      'Mileage_K': [mileage_k],
      'Age_Years': [age_years],
      'Horsepower': [horsepower],
      'Accident': [accident],
  })
  predicted_sale = model.predict(car_features).iloc[0]
  return 'buy' if (auction_price + 1500) < predicted_sale else 'pass'


# Apply both rules to the 40 auction lots
auction['Baseline_Pred'] = baseline_pred_value
auction['v1_Pred'] = auction.apply(
    lambda r: model_v1.predict(pd.DataFrame([r])).iloc[0], axis=1
)

auction['Baseline_Decision'] = np.where(
    (auction['Auction_Price'] + 1500) < auction['Baseline_Pred'], 'buy', 'pass'
)
auction['v1_Decision'] = auction.apply(
    lambda r: should_buy_car(
        r['Mileage_K'],
        r['Age_Years'],
        r['Horsepower'],
        r['Accident'],
        r['Auction_Price'],
        model_v1,
    ),
    axis=1,
)

# Calculate profit for every car bought: Later_Sale_Price - Auction_Price - 1500
auction['Actual_Profit'] = (
    auction['Later_Sale_Price'] - auction['Auction_Price'] - 1500
)

print('\n=== AUCTION SIMULATION (40 LOTS) ===')
for name, col in [
    ('Baseline', 'Baseline_Decision'),
    ('Model v1', 'v1_Decision'),
]:
  bought = auction[auction[col] == 'buy']
  num_bought = len(bought)
  num_losers = (bought['Actual_Profit'] < 0).sum()
  total_profit = bought['Actual_Profit'].sum()
  print(
      f'{name}: Bought {num_bought}/40 | Money-losing cars: {num_losers} |'
      f' Total Profit: ${total_profit:,.0f}'
  )

# 6. Generate Chart for Slide 2 (Actual vs. Predicted on Test Cars)
plt.figure(figsize=(7, 5), dpi=300)
plt.scatter(
    y_test,
    y_pred_v1,
    alpha=0.75,
    color='#1f77b4',
    edgecolors='k',
    label='Model v1 Predictions',
)
plt.axhline(
    baseline_pred_value,
    color='#d62728',
    linestyle=':',
    linewidth=2,
    label=f'Baseline Guess (${baseline_pred_value:,.0f})',
)

# Perfect prediction reference line
min_val, max_val = y_test.min() - 1000, y_test.max() + 1000
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color='black',
    linestyle='--',
    linewidth=1.5,
    label='Perfect Prediction',
)

plt.title(
    'Predicted vs. Actual Sale Price (72 Unseen Test Cars)',
    fontsize=12,
    fontweight='bold',
)
plt.xlabel('Actual Lot Sale Price ($)', fontsize=10)
plt.ylabel('Predicted Sale Price ($)', fontsize=10)
plt.legend(frameon=True)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('test_predictions_chart.png')
print('\nSaved chart to test_predictions_chart.png')