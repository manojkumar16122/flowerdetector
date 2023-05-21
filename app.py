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


def homepage():
    st.subheader("Dataset")
    with st.expander("Link"):
        st.write("[Kaggle](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)")

    with st.expander("Examples"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("datasamples/daisy.jpg", width=200, caption="daisy")
        with col2:
            st.image("datasamples/dandelion.jpg", width=200, caption="dandelion")
        with col3:
            st.image("datasamples/rose.jpg", width=250, caption="rose")
        col4, col5, col6 = st.columns(3)
        with col4:
            st.image("datasamples/sunflower.jpg", width=200, caption="sunflower")
        with col5:
            st.image("datasamples/tulip.jpg", width=200, caption="tulip")
    
    with st.expander("Splits"):
        st.write("**Overall Samples:** 4318")
        st.write("**Training:** 70%")
        st.write("**Validaiton:** 20%")
        st.write("**Testing:** 10%")
    
    with st.expander("Data Augmentation"):
        code = """
        ImageDataGenerator(
            rescale = 1./255,
            rotation_range = 25,
            shear_range = 0.5,
            zoom_range = 0.5,
            width_shift_range = 0.2,
            height_shift_range=0.2,
            horizontal_flip=True
            )
        
        """
        st.code(code, language='python')
    
    with st.expander("Image Shape"):
        st.write((IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    
    st.subheader("Model Architecture")

    with st.expander("Feature Extraction"):
        code = """
mobile_net = MobileNet(
    weights = 'imagenet', 
    include_top = False, 
    input_shape = IMG_SHAPE)
        
for layer in mobile_net.layers:
            layer.trainable = False
        """
        st.code(code, language='python')

    with st.expander("Final Model"):

        code = """
        model = Sequential([
        mobile_net,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dense(5, activation='softmax')
        ])
        """
        st.code(code, language='python')

    st.write("**Optimizer:**", "ADAM")
    st.write("**Loss:**", "Categorical Crossentropy")

    st.subheader("Training and Validaiton Loss")
    st.image("loss.png")

    st.subheader("Model Testing")
    code = """
testing = ImageDataGenerator(rescale = 1./255)
test_batches = testing.flow_from_directory(
                    TEST_PATH, 
                    target_size=(IMAGE_SIZE,IMAGE_SIZE), 
                    batch_size=BATCH_SIZE, 
                    class_mode="categorical",
                    seed=42)

results = model.evaluate(test_batches)
    """
    st.code(code, language='python')
    st.write("**Testing Loss:**","0.32296934723854065")
    st.write("**Testing Accuracy:**","0.8812785148620605")



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
    options = ["Code", "Try Model", "About"],
    icons= ["code-slash","play", "info"],
    menu_icon="list",
    default_index=0,
    orientation="horizontal"
    )

    if selected == "Code":
        homepage()
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
