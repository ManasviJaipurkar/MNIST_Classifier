
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Load the model
model = load_model()  # Ensure this function is defined

# Define image preprocessing function
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return preprocess(image).unsqueeze(0)

# Streamlit app layout
st.title("MNIST Digit Recognition")
st.write("Upload an image of a digit (0-9):")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    # Preprocess the image and make predictions
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1, keepdim=True)
        
    st.write(f'Predicted Digit: {prediction.item()}')
