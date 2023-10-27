# Lab-12:classification des fleur iris
#Realise par Ahmed Tbibi

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import pandas as pd

#Step1: DataSet
iris = datasets.load_iris()
print(iris.data)
print(iris.feature_names)
print(iris.target)
print(iris.target_names)
print(iris.data.shape)

#step2:Model
model = RandomForestClassifier()

#step3:Train
model.fit(iris.data,iris.target)

#step4:Test
prediction = model.predict([[5.,3.9,6.1,9]])
print(prediction)
print(iris.target_names[prediction])

#Model deployement with streamlit :
# streamlit run Lab12.py

st.header('iris classification model')
st.image('Lab1/Images/iris.jpeg')
st.write(iris.data)
st.write(iris.feature_names)

def user_input():
    sepal_length = st.sidebar.slider('sepal length',4.3, 7.9,6.)
    sepal_width = st.sidebar.slider('sepal width',2.0, 4.4, 3.)
    petal_length = st.sidebar.slider('petal length', 1., 9.2, 2.)
    petal_width = st.sidebar.slider('petal width', 0.1, 2.5, 1.)
    data={
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    flower_features = pd.DataFrame(data,index=[0])
    return flower_features
st.sidebar.header('Iris features')
df = user_input()
st.write(df)
st.subheader('Prediction')
prediction = model.predict(df)
st.write(iris.target_names[prediction])
st.image("Lab1/Images/"+iris.target_names[prediction][0]+".png")