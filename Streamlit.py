# To run streamlit - go to cmd administrator
# type 'cd C:\Users\FKW-HP\Desktop\AFoong\Aaron F\FYP'
# 'streamlit run Streamlit.py'

import streamlit as st
import pandas as pd
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Reshape
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras import backend as K
import os
import time
import io
from PIL import Image
import plotly.express as px
import cv2


def render_header():
    st.write("""
        <p align="center"> 
            <H1> Skin Disease Classification 
        </p>
    """, unsafe_allow_html=True)


@st.cache
def load_mekd():
    img = Image.open('test photo.jpg')
    return img


def data_gen_upload(x):
    width = 64
    height = 64
    img = Image.open(x)
    st.image(img, caption=None, width=width, use_column_width=True)
    img_array = np.array(img)
    # cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)

    return rgb_tensor


def load_model():
    model = tf.keras.models.load_model('dermnet.hdf5', custom_objects={"tf": tf}, compile=False)
    return model


@st.cache
def display_prediction(X_class):
    """Display image and preditions from model"""
    
    result = pd.DataFrame({'Probability': X_class}, index=np.arange(23))
    result = result.reset_index()
    result.columns = ['Classes', 'Probability']
    lesion_type_dict = {'Nail Fungus and other Nail Disease':0,'Tinea Ringworm Candidiasis and other Fungal Infections':1,
               'Eczema':2,'Psoriasis pictures Lichen Planus':3, 
               'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions':4, 
               'Warts Molluscum and other Viral Infections':5, 
               'Seborrheic Keratoses and other Benign Tumors': 6,
               'Acne and Rosacea':7,
               'Light Diseases and Disorders of Pigmentation':8,
               'Bullous Disease':9,
               'Melanoma Skin Cancer Nevi and Moles':10,
               'Exanthems and Drug Eruptions':11,
               'Vasculitis':12,
               'Scabies Lyme Disease and other Infestations and Bites':13,
               'Atopic Dermatitis':14,
               'Vascular Tumors':15,
               'Lupus and other Connective Tissue diseases':16,
               'Cellulitis Impetigo and other Bacterial Infections':17,
               'Systemic Disease':18,
               'Hair Loss  Alopecia and other Hair Diseases':19,
               'Herpes HPV and other STDs':20,
               'Poison Ivy  and other Contact Dermatitis':21,
               'Urticaria Hives':22,}
    result["Classes"] = result["Classes"].map(lesion_type_dict)
    return result

def main():

    st.sidebar.header('Skin Disease Classification')
    st.sidebar.subheader('Choose a page to proceed:')
    page = st.sidebar.selectbox("", ["Sample Data", "Upload Your Image"])

    if page == "Sample Data":
        st.header("Sample Data Skin Disease Classification")
        st.markdown("""
        **Now, this is probably why you came here. Let's get you some Predictions**
        You need to choose Sample Data
        """)

        mov_base = ['Sample Data I']
        movies_chosen = st.multiselect('Choose Sample Data', mov_base)

        if len(movies_chosen) > 1:
            st.error('Please select Sample Data')
        if len(movies_chosen) == 1:
            st.success("You have selected Sample Data")
        else:
            st.info('Please select Sample Data')

        if len(movies_chosen) == 1:
            if st.checkbox('Show Sample Data'):
                st.info("Showing Sample data---->>>")
                image = load_mekd()
                st.image(image, caption='Sample Data', use_column_width=True)
                if st.checkbox('Analyse'):
                    x_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=7)
                    st.image(x_sobel, caption='Contrast', use_column_width=True)
                    canny = cv2.Canny(image, 100, 250)
                    st.image(canny, caption='Outline', use_column_width=True)
                st.subheader("Choose Training Algorithm!")
                if st.checkbox('Keras'):
                    model = load_model()
                    st.success("Hooray !! Keras Model Loaded!")
                    if st.checkbox('Show Prediction Probablity on Sample Data'):
                        x_test = data_gen_upload('test photo.jpg')
                        pred_classes = model.predict(x_test)
                        classes_x=np.argmax(pred_classes,axis=1)
                        result = display_prediction(classes_x)
                        st.write(result)
                        if st.checkbox('Display Probability Graph'):
                            fig = px.bar(result, x="Classes",
                                         y="Probability", color='Classes')
                            st.plotly_chart(fig, use_container_width=True)

    if page == "Upload Your Image":

        st.header("Upload Your Image")

        file_path = st.file_uploader('Upload an image', type=['png', 'jpg'])

        if file_path is not None:
            x_test = data_gen_upload(file_path)
            upload_image = Image.open(file_path)

            st.success('File Upload Success!!')
        else:
            st.info('Please upload Image file')

        if st.checkbox('Show Uploaded Image'):
            st.info("Showing Uploaded Image ---->>>")
            st.image(upload_image, caption='Uploaded Image',
                     use_column_width=True)
            st.subheader("Show Disease Characteristics with Image Processing")
            if st.checkbox('Analyse'):
                x_sobel = cv2.Sobel(upload_image, cv2.CV_64F, 1, 0, ksize=7)
                st.image(x_sobel, caption='Contrast', use_column_width=True)
                canny = cv2.Canny(upload_image, 100, 250)
                st.image(canny, caption='Outline', use_column_width=True)
            st.subheader("Choose Training Algorithm!")
            if st.checkbox('Keras'):
                model = load_model()
                st.success("Hooray !! Keras Model Loaded!")
                if st.checkbox('Show Prediction Probablity for Uploaded Image'):
                    Y_pred_classes = model.predict(x_test)
                    classes_x=np.argmax(pred_classes,axis=1)
                    result = display_prediction(classes_x)
                    st.write(result)
                    if st.checkbox('Display Probability Graph'):
                        fig = px.bar(result, x="Classes",
                                     y="Probability", color='Classes')
                        st.plotly_chart(fig, use_container_width=True)


main()