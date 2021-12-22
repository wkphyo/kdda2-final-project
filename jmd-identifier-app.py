import tensorflow as tf
from tensorflow import keras
import streamlit as st
import joblib
from PIL import Image
from skimage.transform import resize
import numpy as np
import time

# Import model
dollar_model = keras.models.load_model('dollar-model')

st.header("JMD Bank Note Identification App 🇯🇲 💵 💴 💶 💷")
st.write('\n')
st.markdown(
    "__*Built by*__: _Denecian Dennis, Elombe Calvert, Simon Lee, Clifton Lee, Win Phyo_")
st.markdown("__*Powered by*__: _TensorFlow, Streamlit_")

st.sidebar.title("Upload Image")
st.sidebar.write("Please upload an image of a $500 or $1000 JMD bank note.")

# Disable warning
st.set_option('deprecation.showfileUploaderEncoding', False)
# User image upload
uploaded_file = st.sidebar.file_uploader("", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:

    u_img = Image.open(uploaded_file)
    show = st.image(u_img, use_column_width=True)
    show.image(u_img, 'Uploaded Image', use_column_width=True)
    # Image Preprocessing
    image = u_img.resize((180, 180))

    img_array = keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

st.sidebar.write('\n')

if st.sidebar.button("Click Here to Identify the Value of the Bank Note"):

    if uploaded_file is None:

        st.sidebar.write("Please upload an image")

    else:

        with st.spinner('Classifying ...'):

            predictions = dollar_model.predict(img_array)
            score = predictions[0]
            time.sleep(2)
            st.success('Done!')

        st.markdown("__Algorithm Predicts:__")
        st.write(
            "This image is %.2f percent this and %.2f percent that."
            % (100 * (1 - score), 100 * score)
        )
