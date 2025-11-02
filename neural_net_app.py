import os
import io
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

st.set_page_config(page_title="CIFAR-10 Classifier", layout="centered")

st.title("🔎 CIFAR-10 Image Classifier (PyTorch + Streamlit)")

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)   

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) 
        return x

@st.cache_resource(show_spinner=False)
def load_model(weights_path: str = "cifar_net.pth") -> nn.Module:
    model = Net()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()
uploaded = st.file_uploader("Выбери изображение", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    try:
        image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except Exception as e:
        st.error(f"Не удалось прочитать файл: {e}")
        st.stop()

    st.image(image, caption="Загруженное изображение", use_container_width=True)
    x = transform(image).unsqueeze(0)  
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0) 

    top_prob, top_idx = probs.max(dim=0)
    pred_class = CLASSES[top_idx.item()]
    st.success(f"Предсказанный класс: **{pred_class}** "
               f"(вероятность {top_prob.item():.2%})")


    with st.expander("Показать все вероятности по классам"):
        for cls, p in zip(CLASSES, probs.tolist()):
            st.write(f"{cls:>6}: {p:.2%}")
else:
    st.info("🖼 Загрузите изображение, чтобы получить предсказание.")
