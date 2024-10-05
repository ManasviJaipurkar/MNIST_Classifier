
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Define the CNN model class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=3)
        self.conv4 = nn.Conv2d(40, 80, kernel_size=1)
        self.drop1 = nn.Dropout2d()
        self.fc1 = nn.Linear(80, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.drop1(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 80)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('MNIST_CNN_model.pth_1', map_location=torch.device('cpu')))
model.eval()

# Function to preprocess the image and make predictions
def predict_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28 (MNIST size)
    img = np.array(img)
    img = (img / 255.0 - 0.1307) / 0.3081  # Normalize with MNIST values
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions

    with torch.no_grad():
        output = model(img)
        prediction = output.argmax(dim=1, keepdim=True).item()
    return prediction

# Streamlit app layout
st.title("MNIST Digit Classifier")
st.write("Upload an image of a digit (0-9) and the model will predict what digit it is.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    prediction = predict_image(image)
    st.write(f'Predicted Digit: {prediction}')
