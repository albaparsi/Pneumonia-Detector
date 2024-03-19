import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import math

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Model code
model = load_model('keras_model.h5')
# load the labels
class_names = open("labels.txt", "r").readlines()


# Title
st.title("Pneumonia De-Tech")

st.image('Streptococcus_pneumoniae.jpg', caption='Streptococcus pneumoniae. Source: Center for Disease Control and Prevention\'s Public Health Image Library')

# Sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("Upload lung X-Ray Here")
    if uploaded_file is not None:
        st.write("File uploaded!")
        with open("uploaded.jpg", 'wb') as f: 
            f.write(uploaded_file.getbuffer())
    else:
        st.write("File not uploaded.")

# Information about pneumonia
st.divider()
st.subheader("What is pneumonia?")

st.markdown(
    '''Pneumonia is an infection in one or both lungs that causes the alveoli to fill up with fluid or pus. According to the CDC, the most common causes of viral Pneumonia in the United States are influenza viruses, SARS-CoV-2, and RSV.
    For more information, visit [CDC.gov](https://www.cdc.gov/pneumonia/index.html)
    '''
)

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
if (uploaded_file):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open("uploaded.jpg").convert("RGB")

    # Put the image in:
    st.image("uploaded.jpg", caption="Uploaded image")
    # Confirm the image
    if (st.button("Confirm Image")):
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score\
        if (class_name):
            st.write("Class:", class_name[2:], end="")
            st.write("Confidence Score:", (math.floor((confidence_score*100*1000)))/1000, "%")
            st.progress(float(confidence_score))
        else:
            st.write("Waiting...")
    

