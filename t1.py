#print
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
data=pd.read_csv(r'D:\BE\LP3 practicals\ML\Practical-1\uber.csv')
data.dropna(inplace=True)
print(data.head())
 
#print
columns_names=data.columns
print(columns_names)

#print
data.info()

#print
data.drop(columns=['Unnamed: 0','key'],inplace=True)
data.isnull().sum()

#print
data.dropna(inplace=True)

#print
data['pickup_datetime']=pd.to_datetime(data['pickup_datetime'])
data['hour']=data['pickup_datetime'].dt.hour
data['day']=data['pickup_datetime'].dt.day
data['month']=data['pickup_datetime'].dt.month
data.drop(['pickup_datetime'],axis=1,inplace=True)

#print
scaler=StandardScaler()
numerical_features=['fare_amount','pickup_longitude','pickup_latitude',
                  'dropoff_longitude','dropoff_latitude','passenger_count']
data[numerical_features]=scaler.fit_transform(data[numerical_features])

x=data.drop('fare_amount',axis=1)
y=data['fare_amount']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


#2.Identify Outliers

import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(x=data['fare_amount'])
plt.title("boxplot of fare amount")
plt.show()


#3.Check Correlation

#print
corr_matrix=data.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix,annot=True,cmap='coolwarm')
plt.title('Correlation matrix')
plt.show()


#4.Implement Linear Regression and Random Forest Regression Models

#print
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

linear_model=LinearRegression()
linear_model.fit(x_train,y_train)

rf_model=RandomForestRegressor(n_estimators=100,random_state=42)
rf_model.fit(x_train,y_train)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
y_pred_linear = linear_model.predict(x_test)
r2_linear = r2_score(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
mae_linear = mean_absolute_error(y_test, y_pred_linear)

# Random Forest Predictions
y_pred_rf = rf_model.predict(x_test)


#5.Evaluate the Models and Compare Their Respective Scores (R2, RMSE, MAE)

r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)

# Display the scores
print("Linear Regression: R2 =", r2_linear, "RMSE =", rmse_linear, "MAE =", mae_linear)
print("Random Forest: R2 =", r2_rf, "RMSE =", rmse_rf, "MAE =", mae_rf)

