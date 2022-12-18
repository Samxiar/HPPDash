import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

header =  st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


def get_data(filename):
    df=pd.read_csv(filename)
    return df


with header:
    st.title("Dashboarding with Sam 👽")
    st.text('''This is a Data Science Project on House Price Prediction 
using the Support Vector Machine Algorithm.''')


with dataset:
    st.title("Mumbai Housing Price prediction.")
    st.text('The Dataset is of mumbai house Prices withdrawn from MagicBricks.com.')
    df=get_data('mumbai.csv')
    price_dist = pd.DataFrame(df['price'].head(150))
    st.subheader('Houseprice Trends')
    st.bar_chart(price_dist)
    st.write(df.head(5))
    

with features:
    st.title('Features')
    st.text('Area, Price, No of Bedrooms, No of Bathrooms are some of the features.')
    st.markdown('* **Area**')
    st.markdown('* **Bedrooms**')
    st.markdown('* **Bathrooms**')
    st.markdown('* **Balconies**')
    st.markdown('* **Locality**')


with model_training:
    st.title('Model')
    st.text('Model Trained With Support Vector Machine.')
    in_put,out_put = st.columns(2)
    areaa=in_put.slider('Area',min_value=0,max_value=6000,value=500,step=10)
    lr = LinearRegression()
    lr = LinearRegression().fit(df[['area']],df['price'])
    prediction=lr.predict([[areaa]])

    out_put.subheader('The Price that the Model Predicted was')
    out_put.write(prediction)
