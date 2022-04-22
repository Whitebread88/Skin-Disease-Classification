# To run streamlit - go to cmd administrator
# type 'cd C:\Users\FKW-HP\Desktop\AFoong\Aaron F\FYP'
# 'streamlit run Streamlit.py'

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from keras import backend as K
import plotly.express as px
import cv2


def render_header():
    st.write("""
        <p align="center"> 
            <H1> Skin Disease Classification 
        </p>
    """, unsafe_allow_html=True)

#Load sample image
@st.cache
def load_mekd():
    img = Image.open('test photo.jpg')
    return img

#resize images and convert to tensor
def data_gen_upload(x):
    width = 128
    height = 128
    img = Image.open(x)
    img = np.array(img)
    inp = cv2.resize(img, (width , height ))
    rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.float32)
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    return rgb_tensor


#Display probability of each class
def display_prediction(pred_prob):
    """Load the 23 types/classes of skin diseases"""
    result = pd.DataFrame({'Probability': pred_prob}, index=np.arange(23))
    result = result.reset_index()
    result.columns = ['Classes', 'Probability']
    lesion_type_dict = {22:'Nail Fungus and other Nail Disease',
                        21:'Tinea Ringworm Candidiasis and other Fungal Infections',
                        20:'Eczema',
                        19:'Psoriasis pictures Lichen Planus', 
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
    display_prediction.variable = result["Classes"]
    return result


#Predict images
def predict(x_test, model):
    Y_pred = model.predict(x_test)
    K.clear_session()
    Y_pred_classes = Y_pred.argmax(axis=1) #Convert to single digit class
    Y_prob = np.round(Y_pred, 2)
    Y_prob = Y_prob*100  
    Y_prob = Y_prob[0]
    confidence = round(100 * (np.max(Y_pred)), 2)
    K.clear_session()
    return Y_prob, Y_pred_classes, confidence



def main():

    st.sidebar.header('Skin Disease Classification')
    st.sidebar.subheader('Choose a page to proceed:')
    page = st.sidebar.selectbox("", ["Sample Data", "Upload Your Image", "Information"])

    if page == "Sample Data":
        st.header("Sample Data Skin Disease Classification")
        st.markdown("""
        **Now, this is probably why you came here. Let's get you some Predictions**
        You need to choose Sample Data
        """)

        samp_data = ['Sample Data I']
        img_chosen = st.multiselect('Choose Sample Data', samp_data)

        if len(img_chosen) > 1:
            st.error('Please select Sample Data')
        if len(img_chosen) == 1:
            st.success("You have selected Sample Data")
        else:
            st.info('Please select Sample Data')

        if len(img_chosen) == 1:
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
                st.subheader("Load Classifier Model")
                if st.checkbox('Keras'):
                    cnn_model = tf.keras.models.load_model('dermnet2')
                    st.success("Keras Model Loaded!")
                    if st.checkbox('Show Prediction Probablity on Sample Data'):
                        x_test = data_gen_upload('test photo.jpg')
                        pred_prob, pred_class, confidence = predict(x_test, cnn_model)
                        result = display_prediction(pred_prob)
                        predicted_class = display_prediction.variable[pred_class]
                        st.write("The predicted Skin Disease is: ",predicted_class)
                        st.metric("Confidence is: ", confidence)
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
            st.success('File Upload Success!')
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
            st.subheader("Load Classifier Model")
            if st.checkbox('Keras'):
                cnn_model = tf.keras.models.load_model('dermnet')
                st.success("Keras Model Loaded!")
                if st.checkbox('Show Prediction Probablity on Sample Data'):
                    xx_test = data_gen_upload('test photo.jpg')
                    pred_prob, pred_class, confidence = predict(xx_test, cnn_model)
                    result = display_prediction(pred_prob)
                    predicted_class = display_prediction.variable[pred_class]
                    st.write("The predicted Skin Disease is: ",predicted_class)
                    st.metric("Confidence is: ", confidence)
                    st.write(result)
                    if st.checkbox('Display Probability Graph'):
                        fig = px.bar(result, x="Classes",
                                     y="Probability", color='Classes')
                        st.plotly_chart(fig, use_container_width=True)
                        
    
    if page == "Information":
        st.header("Information")
        st.markdown("""
        The classifier model is trained on 23 different types of skin disease listed below.
        """)
        st.markdown("""
        1. Acne and Rosacea
        2. Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions
        3. Atopic Dermatitis                
        4. Bullous Disease
        5. Cellulitis Impetigo and other Bacterial Infections
        6. Eczema
        7. Exanthems and Drug Eruptions
        8. Hair Loss Alopecia and other Hair Diseases
        9. Herpes HPV and other STDs
        10. Light Diseases and Disorders of Pigmentation
        11. Lupus and other Connective Tissue diseases
        12. Melanoma Skin Cancer Nevi and Moles
        13. Nail Fungus and other Nail Disease
        14. Poison Ivy and other Contact Dermatitis
        15. Psoriasis pictures Lichen Planus and related diseases
        16. Scabies Lyme Disease and other Infestations and Bites
        17. Seborrheic Keratoses and other Benign Tumors
        18. Systemic Disease
        19. Tinea Ringworm Candidiasis and other Fungal Infections
        20. Urticaria Hives
        21. Vascular Tumors
        22. Vasculitis
        23. Warts Molluscum and other Viral Infections
        """)
        st.header("Limitations")
        st.markdown("""
        1.  Web application and model is not capable of identifying images of skin diseases apart from other objects. 
            The model will attempt to predict any image uploaded by the user. Image processing techniques will also be applied on any image uploaded by the user.
        
        2.  Model will attempt to predict images of skin diseases that are not part of the 23 types of diseases that the model is trained on.
        
        3.  Due to the possibility of false positive or false negative, users are advised to seek proper diagnosis from a medical professional. 
            Model classifier is only for pre-diagnosis and not a definitive diagnosis.
        """)
    
    
    
    
main()