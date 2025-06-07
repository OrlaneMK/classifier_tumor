import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model  
from PIL import Image,ImageOps
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array

st.title("Image Classification with Tensorflow")
st.write("Upload an image to classify.")

uploaded_file = st.sidebar.file_uploader("Choose an image...",
                                         type=["jpg","jpeg","png"])
generated_pred = st.sidebar.button("Predict")
model = tf.keras.models.load_model('model.keras')

classes_prediction = {'glioma_tumor': 0, 'meningioma_tumor': 1, 'no_tumor': 2, 'pituitary_tumor': 3}

if uploaded_file is not None:
    st.image(uploaded_file, caption='Image Telechargee.', use_container_width=True)
    test_image = image.load_img(uploaded_file, target_size=(64, 64))  # Load image with target size
    img_array = img_to_array(test_image)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

if generated_pred:
    predictions = model.predict(img_array)
    predicted_class=np.argmax(predictions[0])
    for key,value in classes_prediction.items():
        if value == predicted_class:
            st.title(f"Predicted class: {key}")