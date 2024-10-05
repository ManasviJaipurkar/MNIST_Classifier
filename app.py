
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import cv2
import numpy as np

# Load your trained model
model_new = keras.models.load_model('models/mnist_cnn_model.pth')

# Set the title of the app
st.title("MNIST Digit Recognizer")

# Size of the canvas
SIZE = 192

# Create a canvas for user input
canvas_result = st_canvas(
    fill_color="#ffffff",
    stroke_width=10,
    stroke_color='#ffffff',
    background_color="#000000",
    height=150,
    width=150,
    drawing_mode='freedraw',
    key="canvas",
)

# If the user has drawn something on the canvas
if canvas_result.image_data is not None:
    # Resize the image to 28x28 for the model input
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    
    # Show the input image
    st.write('Input Image')
    st.image(img_rescaling)

    # Prediction on button click
    if st.button('Predict'):
        # Convert the image to grayscale
        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Make prediction
        pred = model_new.predict(test_x.reshape(1, 28, 28, 1))
        
        # Display the prediction result
        st.write(f'Result: {np.argmax(pred[0])}')
        st.bar_chart(pred[0])  # Display a bar chart of prediction probabilities
