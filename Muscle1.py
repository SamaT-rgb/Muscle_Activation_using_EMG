import pandas as pd 
import numpy as np 
import math
df1=pd.read_csv("test.csv")
x=len(df1)
x
ecg_df1=pd.read_csv("person 1.csv")
y=len(ecg_df1)
y
count=np.round(y/x)
count
ecg_df1.drop(columns=['Identifier'], inplace=True)
ecg_df1
ecg_value=[]
for i in range(0,y,3):
    x=ecg_df1[i:i+3]
    avg=np.mean(x)
    ecg_value.append(avg)
ecg_value
    
df = pd.DataFrame(ecg_value, columns=['ecg_values_avg'])

# Convert to DataFrame

# Add the new column to the existing DataFrame
df1['ecg_values_avg'] = df['ecg_values_avg']

# Save the modified DataFrame back to the existing CSV file
df1.to_csv('test.csv', index=False)
df1
# Drop rows with NaN values

#df1.dropna(inplace=True)

df1.to_csv('test.csv', index=False)
# NOW lets work on side of person 1 
df2=pd.read_csv("test1.csv")
df2
x=len(df2)
count=np.round(y/x)
count=int(count)

ecg_value=[]
for i in range(0,y,count):
    x=ecg_df1[i:i+count]
    avg=np.mean(x)
    ecg_value.append(avg)
ecg_value
df = pd.DataFrame(ecg_value, columns=['ecg_values_avg'])

# Convert to DataFrame

# Add the new column to the existing DataFrame
df2['ecg_values_avg'] = df['ecg_values_avg']
df2.dropna(inplace=True)
# Save the modified DataFrame back to the existing CSV file
df2.to_csv('test1.csv', index=False)
df2
# now lets do for 2nd person 
df3=pd.read_csv("test2.csv")
ecg_df2=pd.read_csv("person 2.csv")
ecg_df2.drop(columns=['Identifier'], inplace=True)
ecg_df2
x=len(df3)
y=len(ecg_df2)
count=np.round(y/x)
count=int(count)
count

ecg_value=[]
for i in range(0,y,count):
    x=ecg_df2[i:i+count]
    avg=np.mean(x)
    ecg_value.append(avg)
ecg_value
df = pd.DataFrame(ecg_value, columns=['ecg_values_avg'])

# Convert to DataFrame

# Add the new column to the existing DataFrame
df3['ecg_values_avg'] = df['ecg_values_avg']
df3.dropna(inplace=True)
# Save the modified DataFrame back to the existing CSV file
df3.to_csv('test2.csv', index=False)
df3
# now lets do for side 2 person
df4=pd.read_csv("test3.csv")
df4
x=len(df4)
y=len(ecg_df2)
count=np.round(y/x)
count=int(count)
count
ecg_value=[]
for i in range(0,y,count):
    x=ecg_df2[i:i+count]
    avg=np.mean(x)
    ecg_value.append(avg)
ecg_value
df = pd.DataFrame(ecg_value, columns=['emg_values_avg'])

# Convert to DataFrame

# Add the new column to the existing DataFrame
df4['ecg_values_avg'] = df['emg_values_avg']
df4.dropna(inplace=True)
# Save the modified DataFrame back to the existing CSV file
df4.to_csv('test3.csv', index=False)
df4.drop(columns=["emg_values_avg"],inplace=True)
df4
df4.to_csv('test3.csv', index=False)
# we have now taken the emg values as  the average of the window 

import csv 
import pandas as pd 
df1=pd.read_csv("test.csv")
df2=pd.read_csv("test1.csv")
df3=pd.read_csv("test2.csv")
df4=pd.read_csv("test3.csv")

csv_filename = 'output.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Side_direction', 'hip_angle', 'knee_angle']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
combined_data = pd.concat([df1,df2], axis=0, ignore_index=True)
print(len(combined_data))
combined_data.to_csv("output.csv", index=False)
data =pd.read_csv("output.csv")
data.head
data =pd.read_csv("output.csv")
combined_data = pd.concat([data,df3], axis=0, ignore_index=True)
print(len(combined_data))
combined_data.to_csv("output.csv", index=False)
data =pd.read_csv("output.csv")
combined_data = pd.concat([data,df4], axis=0, ignore_index=True)
print(len(combined_data))
combined_data.to_csv("output.csv", index=False)
data =pd.read_csv("output.csv")
data.head
df=pd.read_csv("output.csv")
df.drop(columns=["Side_direction"],inplace=True)
df
hip_angle=df.iloc[:,0]
hip_angle
## now we can use ML for checking it 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Given dataset
ecg_values = df.iloc[:,-1]
hip_angles = df.iloc[:,0]
knee_angles = df.iloc[:,1]

ecg_values = ecg_values.values
hip_angles = hip_angles.values
knee_angles = knee_angles.values
# Reshape the ECG values to match sklearn's requirements
X = ecg_values.reshape(-1, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_hip_train, y_hip_test, y_knee_train, y_knee_test = train_test_split(X, hip_angles, knee_angles, test_size=0.2, random_state=42)

# Create and fit the Random Forest regression models
hip_model = RandomForestRegressor(n_estimators=100, random_state=42)
hip_model.fit(X_train, y_hip_train)

knee_model = RandomForestRegressor(n_estimators=100, random_state=42)
knee_model.fit(X_train, y_knee_train)

# Predict hip and knee angles for new ECG values
def predict_angles(ecg_value):
    ecg_value = np.array(ecg_value).reshape(-1, 1)
    predicted_hip_angle = hip_model.predict(ecg_value)
    predicted_knee_angle = knee_model.predict(ecg_value)
    return predicted_hip_angle[0], predicted_knee_angle[0]


# Example usage:
new_ecg_value = 2.0745475105806808
predicted_hip, predicted_knee = predict_angles(new_ecg_value)
print("Predicted Hip Angle:", predicted_hip)
print("Predicted Knee Angle:", predicted_knee)
# lets see using multioutput regressor 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Given dataset
ecg_values = df.iloc[:,-1]
y=df.iloc[:,:-1]

# Reshape the ECG values to match sklearn's requirements
X = ecg_values.values.reshape(-1, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Random Forest regression model with MultiOutputRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
multi_output_rf = MultiOutputRegressor(rf)
multi_output_rf.fit(X_train, y_train)

# Predict both hip and knee angles for new ECG values
def predict_angles(ecg_value):
    ecg_value = np.array(ecg_value).reshape(1, -1)
    predicted_angles = multi_output_rf.predict(ecg_value)
    return predicted_angles[0]


# Example usage:
new_ecg_value = 2.0745475105806808
predicted_angles = predict_angles(new_ecg_value)
print("Predicted Hip Angle:", predicted_angles[0])
print("Predicted Knee Angle:", predicted_angles[1])



# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

# Given dataset
df = pd.read_csv("output.csv")
df.drop(columns=["Side_direction"],inplace=True)

# Good/Perfect Range for Hip and Knee Angle
hip_angle_range = [70, 90]  # Example range for hip angle during squats
knee_angle_range = [80, 110]  # Example range for knee angle during squats

# Taking the Average of the ECG Values
avg_ecg_value = df['ecg_values_avg'].mean()
x=df["hip_angle"]
y1=df["knee_angle"]

# Using the Predict Function to Predict Hip and Knee Angle

ecg_values = df.iloc[:,-1]
y=df.iloc[:,:-1]

# Reshape the ECG values to match sklearn's requirements
X = ecg_values.values.reshape(-1, 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Random Forest regression model with MultiOutputRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
multi_output_rf = MultiOutputRegressor(rf)
multi_output_rf.fit(X_train, y_train)

# Predict both hip and knee angles for new ECG values
def predict_angles(ecg_value):
    ecg_value = np.array(ecg_value).reshape(1, -1)
    predicted_angles = multi_output_rf.predict(ecg_value)
    return predicted_angles[0]

predicted_angles = predict_angles(avg_ecg_value)

#Calculate how well the predicted angles fit the desired range
def calculate_fit_score(angle,ideal_range):
    min_range=ideal_range[0]
    max_range=ideal_range[1]
    count = ((angle >= min_range) & (angle <= max_range)).sum()
    return count
# Calculating fit scores
hip_fit_score = calculate_fit_score(x, hip_angle_range)
knee_fit_score = calculate_fit_score(y1, knee_angle_range)
# Calculating the Percentage
overall_fit_percentage = (hip_fit_score + knee_fit_score) / len(df) * 100 # Scaling to percentage scale

# Printing the results
print()
print("Average ECG Value:", avg_ecg_value)
print("Predicted Hip Angle:", predicted_angles[0])
print("Predicted Knee Angle:", predicted_angles[1])
print("Hip Fit Score:", hip_fit_score)
print("Knee Fit Score:", knee_fit_score)
print("Overall Fit Percentage:", overall_fit_percentage)
