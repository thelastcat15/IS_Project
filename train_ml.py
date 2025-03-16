import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# โหลดข้อมูล (เปลี่ยนเป็นไฟล์ของคุณ)
df = pd.read_csv("Walmart.csv")

# ดูข้อมูลเบื้องต้น
print(df.head())

# ตรวจสอบค่าว่าง
print(df.isnull().sum())

# แปลงวันที่ให้เป็นตัวเลข
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", dayfirst=True, errors="coerce")
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year
df["Weekday"] = df["Date"].dt.weekday

# เลือก Features และ Target
features = ["Day", "Month", "Year", "Weekday", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
target = "Weekly_Sales"

# ลบค่าว่าง
df = df.dropna()

# แยกข้อมูล Train / Test
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# โมเดลที่ 1: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# โมเดลที่ 2: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# ฟังก์ชันช่วยคำนวณค่า MAE และ RMSE
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{model_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print("-" * 30)

evaluate_model(y_test, rf_predictions, "Random Forest")
evaluate_model(y_test, lr_predictions, "Linear Regression")

plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:50], label="Actual Sales", marker="o")
plt.plot(rf_predictions[:50], label="Random Forest Predictions", marker="x")
plt.plot(lr_predictions[:50], label="Linear Regression Predictions", marker="s")
plt.legend()
plt.title("Actual vs Predicted Sales")
plt.xlabel("Data Points")
plt.ylabel("Weekly Sales")
plt.show()



joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(lr_model, "linear_regression_model.pkl")