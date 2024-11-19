import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# Define EnhancedModel
class EnhancedModel(nn.Module):
    def __init__(self, num_classes=4):
        super(EnhancedModel, self).__init__()
        self.resnet = models.resnet50(weights=None)
        for param in list(self.resnet.parameters())[:-20]:  # Unfreeze only the last 20 layers
            param.requires_grad = False
        self.resnet.fc = nn.Sequential(
            nn.Linear(self.resnet.fc.in_features, 512),  # resnet.fc.0
            nn.BatchNorm1d(512),  # resnet.fc.1
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # resnet.fc.4
            nn.BatchNorm1d(256),  # resnet.fc.5
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # resnet.fc.8
        )

    def forward(self, x):
        return self.resnet(x)


# Load model
@st.cache_resource
def load_model():
    model = EnhancedModel(num_classes=4)
    model.load_state_dict(torch.load('enhanced_model_saved.pth', map_location=torch.device('cpu')))
    model.eval()
    return model


# Preprocessing for prediction
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Streamlit App
st.title("Soil Classification App")
st.write("Upload an image to classify its soil type.")

# Upload and process image
uploaded_file = st.file_uploader("Choose a soil image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    input_tensor = val_transform(image).unsqueeze(0)

    # Load model and predict
    with st.spinner("Classifying..."):
        model = load_model()
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        classes = ["Black Soil", "Cinder Soil", "Laterite Soil", "Yellow Soil"]  # Replace with actual class names
        st.write(f"Predicted Class: **{classes[predicted]}**")
