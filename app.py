import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Flowers Recognition",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('flower-recog-cnn.h5')
    return model

IMAGE_SIZE = 224
CHANNELS = 3

class_indices = {'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4}
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

st.header("Flowers Recognition")



 


def try_model():
    with st.spinner("Loading Model"):
        model = load_model()
    uploaded_file = st.file_uploader("Choose an Image", type=['png', 'jpg', 'JPEG'])
    if st.button("Submit"):
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            with st.spinner("Predicting"):
                pred = predict(image, model)
                predicted_class = class_names[np.argmax(pred)]
            st.success(f"I think the flower is **{predicted_class}**")

def predict(image, model):
    data = np.ndarray(shape=(1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32)
    size = (IMAGE_SIZE, IMAGE_SIZE)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0)
    data[0] = normalized_image_array
    prediction = model.predict(data)
    return prediction


def launch():
    
    selected = option_menu(
    menu_title=None,
    options = ["Detect Flowers", "About"],
    icons= ["play", "info"],
    menu_icon="list",
    default_index=0,
    orientation="vertical"
    )

    
    if selected == "Try Model":
        try_model()
    if selected == "About":
        about()
    


def about():
    st.write("### Ibrahim M. Nasser")
    st.write("Freelance Machine Learning Engineer")
    st.write("[Website](https://ibrahim-nasser.com/)",  
             "[LinkedIn](https://www.linkedin.com/in/ibrahimnasser96/)",
             "[GitHub](https://github.com/96ibman)",
             "[Youtube](https://www.youtube.com/channel/UC7N-dy3UbSBHnwwv-vulBAA)",
             "[Twitter](https://twitter.com/mleng_ibrahim)"
            )
    st.image("my_picture.jpeg", width=350)

if __name__ == "__main__":
    launch()
 
