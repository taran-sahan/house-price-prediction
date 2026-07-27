# 🏠 House Price Prediction

A beginner machine learning project that predicts median house prices for California districts using Linear Regression.

## 📌 Objective
Predict house prices based on features like median income, location, room counts, and ocean proximity.

## 📊 Dataset
[California Housing Prices](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv) — 20,640 rows, 10 columns.

> Note: This project uses the California Housing dataset instead of the older "Boston Housing" dataset, which was removed from scikit-learn due to ethical concerns with some of its features.

**Features:** longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity (categorical)
**Target:** median_house_value

## 🔧 Workflow
1. **Load & Explore** — inspected shape, distributions, missing values, and correlations
2. **Preprocess** — filled 207 missing `total_bedrooms` values with the median, one-hot encoded `ocean_proximity`
3. **Feature Engineering** — added `rooms_per_household`, `bedrooms_per_room`, `population_per_household`
4. **Train/Test Split** — 80/20 split (random_state=42)
5. **Scaling** — StandardScaler, fit on training data only
6. **Model** — Linear Regression
7. **Evaluation** — MSE, RMSE, MAE, R²

## 📈 Results
| Metric | Value |
|---|---|
| RMSE | ~$72,669 |
| MAE | ~$50,889 |
| R² Score | 0.597 |

**Key finding:** `median_income` is by far the strongest predictor of house price (0.69 correlation) — much stronger than housing age or room counts.

## 🖼️ Visualizations
- Price distribution (revealed a data-capping issue at $500,000)
- Feature correlation heatmap
- Price vs. median income scatter plot
- Predicted vs. actual price plot

## 🛠️ How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python house_price_prediction.py
```
Make sure `housing.csv` is in the same folder as the script.

## 🚀 Next Steps
- Try RandomForestRegressor / GradientBoostingRegressor for better accuracy
- Log-transform skewed features
- Cross-validation instead of a single train/test split
- Handle the $500k price cap explicitly

## 🧰 Tech Stack
Python · pandas · NumPy · scikit-learn · matplotlib · seaborn

---
*Built as a hands-on project to practice the full ML regression workflow — from raw data to evaluated model.*
