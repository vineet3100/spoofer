import streamlit as st
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
from torchvision import models

# Function to apply adversarial transformations
def process_single_image(input_path, output_path):
    # Load image
    image = Image.open(input_path).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    # Load pretrained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()

    # Apply FGSM attack
    epsilon = 0.1
    image_tensor.requires_grad = True
    output = model(image_tensor)
    target = torch.tensor([0])
    loss = torch.nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = image_tensor.grad.data
    perturbed_image = image_tensor + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Save adversarial image
    perturbed_image = transforms.ToPILImage()(perturbed_image.squeeze(0))
    perturbed_image.save(output_path)

# Streamlit app layout
st.title("Adversarial Image Generator")
st.write("Upload an image, and this app will generate an adversarial image.")

uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Save uploaded file
    input_path = os.path.join("uploads", uploaded_file.name)
    output_path = os.path.join("outputs", f"spoofed_{uploaded_file.name}")
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the image
    st.write("Processing your image...")
    process_single_image(input_path, output_path)

    # Display the adversarial image
    st.image(output_path, caption="Adversarial Image")
    st.success("Your adversarial image has been generated!")

    # Download the adversarial image
    with open(output_path, "rb") as file:
        st.download_button(
            label="Download Adversarial Image",
            data=file,
            file_name=f"spoofed_{uploaded_file.name}",
            mime="image/png"
        )
