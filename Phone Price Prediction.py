import pandas as pd
import streamlit as st
import os
import numpy as np

dataset = pd.read_csv("C:\\Users\\tejak\\OneDrive\\Desktop\\New folder\\Example.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=5, random_state=0)
regressor.fit(X_train, y_train)
y_pred=regressor.predict(X_test)

# from sklearn.metrics import r2_score as r
# metrics=r(y_test,y_pred)
# print(metrics)
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # Make predictions on the test data
# y_pred1 = model.predict(X_test)
# print(r(y_test,y_pred1))
# from sklearn.linear_model import Lasso
# model = Lasso(alpha=0.1)  # set the regularization parameter
# model.fit(X_train, y_train)
#
# # Make predictions on the test data
# y_pred = model.predict(X_test)
#
# # Evaluate the performance of the model
# r2 = r(y_test, y_pred)
# print("R^2 score:", r2)

st.header("Phone Price Prediction")

# Define dictionary to map processor options to processor models
processor_models = {
    "1": ['778', '680', '888','8.1','675','678','870','732','855'],
    "2": ['88', '95','80','900','700','8100','920','810','9000', '1200', '1100'],
    "3": ['9611','1280','820','850'],

}

st.write("For your convenience we have changed the mobile brand names such that if you want to select Samsung, select 1 in the first button. Similarly, go through the below table for other brands.")
df = pd.DataFrame({
    'Brand': ["Samsung", "Mi", "Oppo", "OnePlus", "Vivo", "iQOO"],
    'Number': [1, 2, 3, 4, 5, 6],
})
df.index = df.index + 1
st.table(df)

st.write("For your convenience we have changed the mobile processor names such that if you want to select Snapdragon, select 1 in the second button. Similarly, go through the below table for other processors.")
df1 = pd.DataFrame({
    'Processor': ["Snapdragon", "Dimensity", "Exynos"],
    'Number': [1, 2, 3],

})
df1.index = df1.index + 1
st.table(df1)

df2 = pd.DataFrame({
    'Screen': ["AMOLED", "LCD", "TFT"],
    'Number': [1, 2, 3],

})
st.write("For your convenience we have changed the mobile screen names such that if you want to select AMOLED, select 1 in the second button. Similarly, go through the below table for other screens.")
df2.index = df2.index + 1
st.table(df2)

ques = st.radio(
    "Select Brand",
    ('1', '2', '3', '4', '5', '6'))

ques1 = st.radio(
    'Select Processor',
    ('1','2','3'))
print(ques1)

# Retrieve the list of processor models for the selected processor option
radio_options1 = processor_models[ques1]
print(radio_options1)

ques2 = st.selectbox(
    "Select Processor Model",
    radio_options1)
print(ques2)

ques3 = st.radio(
    "Select RAM",
    ('4', '6', '8'))

ques4 = st.radio(
    "Select Display",
    ('1', '2', '3'))



ques5 = st.slider("Select Battery",3000, 6000, 4000)
ques6 = st.radio(
    "Select Genaration",
    ('4','5',))
ques7 = st.radio(
    "Select Camera",
      ('16','20','32'))


if(st.button('predict')):
   a=regressor.predict([[ques,ques1,ques2,ques3,ques4,ques5,ques6,ques7]])
   n=a.item()
   n=int(n)
   #st.subheader()
   st.header("₹"+""+str(n))
#import streamlit as st
#import pandas as pd


