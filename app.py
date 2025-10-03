import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="OCT Eye Disease Classifier",
    page_icon="🩺",
    layout="centered",
)

st.title("OCT Eye Disease Classifier")
st.write("Upload an OCT image and the model will classify it into one of the categories 👇")

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("DenseNet121_best_weights.keras")
    return model

model = load_model()

# ------------------ Class Labels ------------------
class_labels = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL'}

# ------------------ File Upload ------------------
uploaded_file = st.file_uploader("Upload an OCT image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100

    # Result
    st.subheader(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Probability chart
    fig, ax = plt.subplots()
    ax.bar(class_labels.values(), prediction[0])
    ax.set_ylabel("Probability")
    ax.set_title("Class Probabilities")
    plt.xticks(rotation=30)
    st.pyplot(fig)

else:
    st.info("Please upload an OCT image to classify")

