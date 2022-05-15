from turtle import onclick
import streamlit as st
from PIL import Image
from classification import classify
from os import path
import random
import time



st.backgroundcolor="#919e8b"

header = st.header('LUNG XRAY EVALUATOR')
st.write("""
Description:
This is an application that allows doctors to be able to check the results of an XRAY and let the doctor know if a patient possibly has a form of pneumonia.
This helps a doctors better diagnose their patients and reduce chances of a misdiagnosis. 
This is built with machine learning and has been provided with many XRAY scans of both normal and infected lungs to be able to accurately recongnise the two apart.





Disclaimer = Non X-ray/edited X-rays images may return inacurate results""")


#upload file of type PNG, JPG, JPEG, WEBP
uploaded_file = st.file_uploader("Choose a file",type=(['png', 'jpg', 'jpeg', 'webp']))
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(uploaded_file)
    
    st.sidebar.image(image)
    st.sidebar.subheader('Uploaded image rendering')
    st.sidebar.write('Trained with Kaggle dataset with ~5000 files, https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia, reaching around ~95%' +' accuracy on non edited images')
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 30vw;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 30vw;
            margin-left: -500px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
    st.header("Prediction:")
    
    if classify(uploaded_file) == 'PNEUMONIA':
        st.subheader("- This patient's xray shows signs of PNEUMONIA")
    else:
        st.subheader("- This patient's xray looks NORMAL")

    