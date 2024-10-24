import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the data
# Assuming the DataFrame has the following columns: 'ecg_value', 'hip_angle', 'knee_angle'
df = pd.read_csv('your_data.csv')

# Step 1: Data processing and averaging of ECG values
window_size = 5  # Define window size for averaging
ecg_avg = df['ecg_value'].rolling(window=window_size).mean().dropna()  # Calculate rolling average

# Adjust the dataset for equal length after averaging
df = df.iloc[window_size-1:]
df['ecg_avg'] = ecg_avg.values

# Step 2: Handle missing data (optional imputation)
imputer = SimpleImputer(strategy='mean')  # Fill missing values with the mean
df[['hip_angle', 'knee_angle']] = imputer.fit_transform(df[['hip_angle', 'knee_angle']])

# Step 3: Standardize ECG values to handle scaling differences
scaler = StandardScaler()
df['ecg_avg_scaled'] = scaler.fit_transform(df[['ecg_avg']])

# Step 4: Prepare features and targets
X = df[['ecg_avg_scaled']]
y = df[['hip_angle', 'knee_angle']]

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the RandomForestRegressor model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
model = MultiOutputRegressor(rf)
model.fit(X_train, y_train)

# Step 7: Predict on test data
y_pred = model.predict(X_test)

# Step 8: Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R2 Score: {r2}')

# Step 9: Improved fit score calculation
def calculate_fit_score(y_true, y_pred, hip_angle_range, knee_angle_range):
    hip_diff = np.abs(y_true[:, 0] - y_pred[:, 0])
    knee_diff = np.abs(y_true[:, 1] - y_pred[:, 1])
    
    hip_fit_score = np.sum((y_pred[:, 0] >= hip_angle_range[0]) & (y_pred[:, 0] <= hip_angle_range[1]))
    knee_fit_score = np.sum((y_pred[:, 1] >= knee_angle_range[0]) & (y_pred[:, 1] <= knee_angle_range[1]))
    
    # Additional weighted penalty for predictions outside the range
    hip_penalty = np.sum((y_pred[:, 0] < hip_angle_range[0]) | (y_pred[:, 0] > hip_angle_range[1])) * 0.1
    knee_penalty = np.sum((y_pred[:, 1] < knee_angle_range[0]) | (y_pred[:, 1] > knee_angle_range[1])) * 0.1
    
    total_fit_score = (hip_fit_score - hip_penalty) + (knee_fit_score - knee_penalty)
    max_score = len(y_true) * 2  # Max score is the total number of samples * 2 (hip and knee)
    
    fit_percentage = (total_fit_score / max_score) * 100
    return fit_percentage

# Define acceptable angle ranges (based on your domain knowledge)
hip_angle_range = [30, 150]  # Example range, adjust as needed
knee_angle_range = [0, 140]  # Example range, adjust as needed

fit_score = calculate_fit_score(y_test.values, y_pred, hip_angle_range, knee_angle_range)
print(f'Fit Score Percentage: {fit_score:.2f}%')
