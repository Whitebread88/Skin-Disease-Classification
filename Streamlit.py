# To run streamlit - go to cmd administrator
# type 'cd C:\Users\FKW-HP\Desktop\AFoong\Aaron F\FYP'
# 'streamlit run Streamlit.py'

import streamlit as st
import pandas as pd
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Reshape
from PIL import Image
import plotly.express as px
import cv2
# import tensorflow.keras.backend.tensorflow_backend as tb
# tb._SYMBOLIC_SCOPE.value = True



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
    width = 128
    height = 128
    img = Image.open(x)
    img = np.array(img)
    inp = cv2.resize(img, (width , height ))
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
   
    # cv2.imwrite('out.jpg', cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)

    return rgb_tensor



def display_prediction(pred_prob):
    """Load the 23 types/classes of skin diseases"""
    result = pd.DataFrame({'Probability': pred_prob}, index=np.arange(23))
    result = result.reset_index()
    result.columns = ['Classes', 'Probability']
    lesion_type_dict = {22:'Nail Fungus and other Nail Disease',21:'Tinea Ringworm Candidiasis and other Fungal Infections',
                       20:'Eczema',19:'Psoriasis pictures Lichen Planus', 
                       18:'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 
                       17:'Warts Molluscum and other Viral Infections', 
                       16:'Seborrheic Keratoses and other Benign Tumors',
                       15:'Acne and Rosacea',
                       14:'Light Diseases and Disorders of Pigmentation',
                       13:'Bullous Disease',
                       12:'Melanoma Skin Cancer Nevi and Moles',
                       11:'Exanthems and Drug Eruptions',
                       10:'Vasculitis',
                       9:'Scabies Lyme Disease and other Infestations and Bites',
                       8:'Atopic Dermatitis',
                       7:'Vascular Tumors',
                       6:'Lupus and other Connective Tissue diseases',
                       5:'Cellulitis Impetigo and other Bacterial Infections',
                       4:'Systemic Disease',
                       3:'Hair Loss  Alopecia and other Hair Diseases',
                       2:'Herpes HPV and other STDs',
                       1:'Poison Ivy  and other Contact Dermatitis',
                        0:'Urticaria Hives'}
    result["Classes"] = result["Classes"].map(lesion_type_dict)
    return result



def predict(x_test, model):
    Y_pred = model.predict(x_test)
    Y_pred_classes = Y_pred.argmax(axis=1) #Convert to single digit class
    Y_prob = round(100 * (np.max(Y_pred)), 2)
    
    return Y_prob, Y_pred_classes


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
                    num_image = np.array(image)
                    x_sobel = cv2.Sobel(num_image, cv2.CV_64F, 1, 0, ksize=7)
                    st.image(x_sobel, caption='X-Sobel / Contrast', use_column_width=True, clamp=True, channels='BGR')
                    canny = cv2.Canny(num_image, 100, 250)
                    st.image(canny, caption='Outline', use_column_width=True, clamp=True)
                    edged = cv2.Canny(num_image, 30, 200)
                    contour, hier = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    st.info("Count of Contours  = " + str(len(contour)))
                    cont = cv2.drawContours(num_image, contour, -30, (0,255,0), 1)
                    st.image(cont, caption='Contours', use_column_width=True, clamp=True)
                    blob_image = cv2.cvtColor(num_image, cv2.COLOR_BGR2GRAY)
                    # using the SIRF algorithm to detect keypoints in the image
                    detector = cv2.SimpleBlobDetector_create()
                    features = cv2.SIFT_create()
                    keypoints = features.detect(blob_image, None)
                    # drawKeypoints function is used to draw keypoints
                    output_image = cv2.drawKeypoints(blob_image, keypoints, 0, (255, 0, 0),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    st.image(output_image, caption='Detect Blobs on image', use_column_width=True, clamp=True)
                st.subheader("Choose Training Algorithm!")
                if st.checkbox('Keras'):
                    cnn_model = tf.keras.models.load_model('dermnet.h5')
                    st.success("Hooray !! Keras Model Loaded!")
                    if st.checkbox('Show Prediction Probablity on Sample Data'):
                        x_test = data_gen_upload('test photo.jpg')
                        pred_prob, pred_class = predict(x_test, cnn_model)
                        result = display_prediction(pred_prob)
                        # normalized_prediction = predictions.argmax(axis=1)
                        # st.write(normalized_prediction)
#                         predictions = np.round(predictions, 2)
#                         predictions = predictions*100
#                         new_predictions = predictions[0].tolist()
#                         Y_pred_classes = np.argmax(Y_pred, axis=1)
                        # confidence = round(100 * (np.max(predictions)), 2)
                        # result = display_prediction(confidence)
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
                num_image = np.array(upload_image)
                x_sobel = cv2.Sobel(num_image, cv2.CV_64F, 1, 0, ksize=7)
                st.image(x_sobel, caption='X-Sobel / Contrast', use_column_width=True, clamp=True, channels='BGR')
                canny = cv2.Canny(num_image, 100, 250)
                st.image(canny, caption='Outline', use_column_width=True, clamp=True)
                edged = cv2.Canny(num_image, 30, 200)
                contour, hier = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                st.info("Count of Contours  = " + str(len(contour)))
                cont = cv2.drawContours(num_image, contour, -30, (0,255,0), 1)
                st.image(cont, caption='Contours', use_column_width=True, clamp=True)
                blob_image = cv2.cvtColor(num_image, cv2.COLOR_BGR2GRAY)
                # using the SIRF algorithm to detect keypoints in the image
                detector = cv2.SimpleBlobDetector_create()
                features = cv2.SIFT_create()
                keypoints = features.detect(blob_image, None)
                # drawKeypoints function is used to draw keypoints
                output_image = cv2.drawKeypoints(blob_image, keypoints, 0, (255, 0, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                st.image(output_image, caption='Detect Blobs on image', use_column_width=True, clamp=True)
            st.subheader("Choose Training Algorithm!")
            if st.checkbox('Keras'):
                cnn_model = tf.keras.models.load_model('dermnet.h5')
                st.success("Hooray !! Keras Model Loaded!")
                if st.checkbox('Show Prediction Probablity for Uploaded Image'):
                    Y_pred_classes = cnn_model.predict(x_test)
                    classes_x=np.argmax(pred_classes,axis=1)
                    result = display_prediction(classes_x)
                    st.write(result)
                    if st.checkbox('Display Probability Graph'):
                        fig = px.bar(result, x="Classes",
                                     y="Probability", color='Classes')
                        st.plotly_chart(fig, use_container_width=True)


main()