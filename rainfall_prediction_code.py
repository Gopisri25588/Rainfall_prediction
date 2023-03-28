import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import plotly.express as px

data = pd.read_csv("D:\\deepai_assignment\\rainfall_predict_DSAI\\rainfall_data_1901_2015_champawat_uk.csv")
st.dataframe(data)
data.head()
data.info()
data.isnull().sum()
data.duplicated().sum()
data.mean()
data = data.fillna(data.mean())
data.isnull().any()
data.YEAR.unique()
data.describe()

fig = px.line(data, x='YEAR', y='ANNUAL', title='Rainfall over Years', color_discrete_sequence=['red'],
              markers=dict(size=10, color='red'))
fig.update_xaxes(title='Years')
fig.update_yaxes(title='Annual Rainfall in mm')
fig.update_layout(xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                  yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
st.plotly_chart(fig)

fig = px.line(data, x='YEAR', y=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'],
              title='Year vs Rainfall in Each Month',
              color_discrete_sequence=['red', 'blue', 'green', 'yellow', 'orange', 'violet', 'black', 'pink',
                                       'forestgreen', 'brown', 'cyan', 'grey'], markers=dict(size=10, color='red'))
fig.update_xaxes(title='Years')
fig.update_yaxes(title='Rainfall in mm')
fig.update_layout(xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                  yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
st.plotly_chart(fig)

fig = px.line(data, x='YEAR', y=['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec'], title='Year vs Rainfall in Combine Month',
              color_discrete_sequence=['red', 'blue', 'green', 'yellow'], markers=dict(size=10, color='red'))
fig.update_xaxes(title='Years')
fig.update_yaxes(title='Rainfall in mm')
fig.update_layout(xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                  yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'))
st.plotly_chart(fig)

UK = data.loc[(data['SUBDIVISION'] == 'Champawat')]
UK.head(4)
plt.figure(figsize=(10, 6))
UK[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].mean().plot(kind="bar",
                                                                                                     width=0.5,
                                                                                                     linewidth=2)
plt.title("Champawat Rainfall v/s Months", size=20)
plt.xlabel("Months", size=14)
plt.ylabel("Rainfall in MM", size=14)
plt.grid(axis="both", linestyle="-.")
Month_rainfall = plt.show()
st.pyplot(Month_rainfall)

UK.groupby("YEAR").sum()['ANNUAL'].plot(ylim=(50, 1500), color='r', marker='o', linestyle='-', linewidth=2,
                                        figsize=(12, 8));
plt.xlabel('Year', size=14)
plt.ylabel('Rainfall in MM', size=14)
plt.title('Champawat Annual Rainfall from Year 1901 to 2021', size=20)
plt.grid()
Annual_rainfall = plt.show()
st.pyplot(Annual_rainfall)

plt.figure(figsize=(15, 6))
sns.heatmap(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'ANNUAL']].corr(),
            annot=True)
heat_map = plt.show()
st.pyplot(heat_map)

data["SUBDIVISION"].nunique()
group = data.groupby('SUBDIVISION')[
    'YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
data = group.get_group('Champawat')
data.head()
df = data.melt(['YEAR']).reset_index()
df.head()
df = df[['YEAR', 'variable', 'value']].reset_index().sort_values(by=['YEAR', 'index'])
df.head()
df.YEAR.unique()
df.columns = ['Index', 'Year', 'Month', 'Avg_Rainfall']
df.head()
Month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9,
             'OCT': 10, 'NOV': 11, 'DEC': 12}
df['Month'] = df['Month'].map(Month_map)
df.head(12)
df.drop(columns="Index", inplace=True)
df.head(2)
df.groupby("Year").sum().plot()
month_avg = plt.show()
st.pyplot(month_avg)
X = np.asanyarray(df[['Year', 'Month']]).astype('int')
y = np.asanyarray(df['Avg_Rainfall']).astype('int')
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# 1.Linear_Regression - Model
st.write('1.Linear_Regression - Model')
LR = LinearRegression()
LR.fit(X_train, y_train)
y_train_predict = LR.predict(X_train)
y_test_predict = LR.predict(X_test)
st.write("-------Test Data--------")
st.write('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
st.write('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))
st.write("-------Train Data--------")
st.write('MAE:', metrics.mean_absolute_error(y_train, y_train_predict))
st.write('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))
st.write("-----Training Accuracy-------")
st.write(round(LR.score(X_train, y_train), 3) * 100)
st.write("-----Testing Accuracy--------")
st.write(round(LR.score(X_test, y_test), 3) * 100)
predicted_LR1 = LR.predict([[2016, 11]])
predicted_LR2 = LR.predict([[2022, 8]])
st.write(predicted_LR1, predicted_LR2)

# 2.Lasso_Regression - Model
st.write('2.Lasso_Regression - Model')
lasso = Lasso(max_iter=100000)
parameter = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]}
lasso_regressor = GridSearchCV(lasso, parameter, scoring='neg_mean_squared_error', cv=5)
lasso_regressor.fit(X_train, y_train)
st.write("Best Parameter for Lasso:", lasso_regressor.best_estimator_)
lasso = Lasso(alpha=100.0, max_iter=100000)
lasso.fit(X_train, y_train)
y_train_predict = lasso.predict(X_train)
y_test_predict = lasso.predict(X_test)
st.write("-------Test Data--------")
st.write('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
st.write('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))
st.write("-------Train Data--------")
st.write('MAE:', metrics.mean_absolute_error(y_train, y_train_predict))
st.write('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))
st.write("-----Training Accuracy-------")
st.write(round(lasso.score(X_train, y_train), 3) * 100)
st.write("-----Testing Accuracy--------")
st.write(round(lasso.score(X_test, y_test), 3) * 100)
predicted_lasso1 = lasso.predict([[2016, 11]])
predicted_lasso2 = lasso.predict([[2022, 8]])
st.write(predicted_lasso1, predicted_lasso2)

# 3.Ridge_Regression - Model
st.write('3.Ridge_Regression - Model')
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X_train, y_train)
st.write(ridge_regressor.best_params_)
st.write(ridge_regressor.best_score_)
st.write("Best Parameter for Ridge:", ridge_regressor.best_estimator_)
ridge = Ridge(alpha=100.0)
ridge.fit(X_train, y_train)
y_train_predict = ridge.predict(X_train)
y_test_predict = ridge.predict(X_test)
st.write("-------Test Data--------")
st.write('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
st.write('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))
st.write("-------Train Data--------")
st.write('MAE:', metrics.mean_absolute_error(y_train, y_train_predict))
st.write('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))
st.write("-----Training Accuracy-------")
st.write(round(ridge.score(X_train, y_train), 3) * 100)
st.write("-----Testing Accuracy--------")
st.write(round(ridge.score(X_test, y_test), 3) * 100)
predicted_ridge1 = ridge_regressor.predict([[2016, 11]])
predicted_ridge2 = ridge_regressor.predict([[2022, 8]])
st.write(predicted_ridge1, predicted_ridge2)

# 4.Random_Forest - Model
st.write('4.Random_Forest - Model')
random_forest_model = RandomForestRegressor(max_depth=100, max_features='sqrt', min_samples_leaf=4,
                                            min_samples_split=10, n_estimators=800)
random_forest_model.fit(X_train, y_train)
y_train_predict = random_forest_model.predict(X_train)
y_test_predict = random_forest_model.predict(X_test)
st.write("-------Test Data--------")
st.write('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
st.write('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))
st.write("-------Train Data--------")
st.write('MAE:', metrics.mean_absolute_error(y_train, y_train_predict))
st.write('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))
st.write("-----------Training Accuracy------------")
st.write(round(random_forest_model.score(X_train, y_train), 3) * 100)
st.write("-----------Testing Accuracy------------")
st.write(round(random_forest_model.score(X_test, y_test), 3) * 100)
predicted_rf1 = random_forest_model.predict([[2016, 11]])
predicted_rf2 = random_forest_model.predict([[2022, 8]])
st.write(predicted_rf1, predicted_rf2)

# # 5.SVM_Regression - Model
# svm_regr = svm.SVC(kernel='rbf')
# svm_regr.fit(X_train, y_train)
# y_test_predict = svm_regr.predict(X_test)
# y_train_predict = svm_regr.predict(X_train)
# print("-------Test Data--------")
# print('MAE:', metrics.mean_absolute_error(y_test, y_test_predict))
# print('MSE:', metrics.mean_squared_error(y_test, y_test_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_test_predict)))
# print("//n-------Train Data--------")
# print('MAE:', metrics.mean_absolute_error(y_train, y_train_predict))
# print('MSE:', metrics.mean_squared_error(y_train, y_train_predict))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_train, y_train_predict)))
# print("//n-----Training Accuracy-------")
# print(round(svm_regr.score(X_train, y_train), 3) * 100)
# print("-----Testing Accuracy--------")
# print(round(svm_regr.score(X_test, y_test), 3) * 100)
# predicted_svm1 = svm_regr.predict([[2016, 11]])
# predicted_svm2 = svm_regr.predict([[2022, 8]])
# print(predicted_svm1, predicted_svm2)

# data.shape
# data[["SUBDIVISION", "ANNUAL"]].groupby("SUBDIVISION").sum().sort_values(by='ANNUAL', ascending=False).plot(kind='barh',
#                                                                                                             stacked=True,
#                                                                                                             figsize=(
#                                                                                                                15, 10))
# plt.xlabel("Rainfall in MM", size=12)
# plt.ylabel("Sub-Division", size=12)
# plt.title("Annual Rainfall v/s SubDivisions")
# plt.grid(axis="x", linestyle="-.")
# Rainfall_subdivision = plt.show()
# st.pyplot(Rainfall_subdivision)

# plt.figure(figsize=(15, 8))
# data.groupby("YEAR").sum()['ANNUAL'].plot(kind="line", color="r", marker=".")
# plt.xlabel("YEARS", size=12)
# plt.ylabel("RAINFALL IN MM", size=12)
# plt.grid(axis="both", linestyle="-.")
# plt.title("Rainfall over Years")
# Year_rainfall = plt.show()
# st.pyplot(Year_rainfall)

# data[['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
#       'OCT', 'NOV', 'DEC']].groupby("YEAR").sum().plot(kind="line", figsize=(18, 8))
# plt.xlabel("Year", size=13)
# plt.ylabel("Rainfall in MM", size=13)
# plt.title("Year v/s Rainfall in each month", size=20)
# Year_rainfall_month = plt.show()
# st.pyplot(Year_rainfall_month)

# data[['YEAR', 'Jan-Feb', 'Mar-May',
#       'Jun-Sep', 'Oct-Dec']].groupby("YEAR").sum().plot(figsize=(10, 7))
# plt.xlabel("Year", size=13)
# plt.ylabel("Rainfall in MM", size=13)
# year_rainfall_quart = plt.show()
# st.pyplot(year_rainfall_quart)

# data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
#       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("SUBDIVISION").sum().plot(kind="barh", stacked=True, figsize=(13, 8))
# plt.title("Sub-Division v/s Rainfall in each month")
# plt.xlabel("Rainfall in MM", size=12)
# plt.ylabel("Sub-Division", size=12)
# plt.grid(axis="x", linestyle="-.")
# sub_div_rainfall = plt.show()
# st.pyplot(sub_div_rainfall)
#
# data[['SUBDIVISION', 'Jan-Feb', 'Mar-May',
#       'Jun-Sep', 'Oct-Dec']].groupby("SUBDIVISION").sum().plot(kind="barh", stacked=True, figsize=(16, 8))
# plt.xlabel("Rainfall in MM", size=12)
# plt.ylabel("Sub-Division", size=12)
# plt.grid(axis="x", linestyle="-.")
# sub_div_rainfall_two_month = plt.show()
# st.pyplot(sub_div_rainfall_two_month)
