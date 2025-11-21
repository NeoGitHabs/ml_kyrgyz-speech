# main.py
import torch
import torchaudio
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
import streamlit as st
from pathlib import Path

# --- Модель ---
class VGG16Gender(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1,64,3,padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(64,64,3,padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64,128,3,padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(128,128,3,padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128,256,3,padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(256,256,3,padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(256,256,3,padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(256,512,3,padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(512,512,3,padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(512,512,3,padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(512,512,3,padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(512,512,3,padding=1), torch.nn.ReLU(),
            torch.nn.Conv2d(512,512,3,padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512*4*12, 4096), torch.nn.ReLU(), torch.nn.Dropout(0.528),
            torch.nn.Linear(4096, 4096), torch.nn.ReLU(), torch.nn.Dropout(0.528),
            torch.nn.Linear(4096, 2)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# Загрузка модели
MODEL_PATH = "kyrgyz_vgg16.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16Gender().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128).to(device)


# --- Streamlit ---
st.title("Распознавание пола по голосу (киргизский)")
uploaded = st.file_uploader("Загрузи аудио (wav/mp3)", type=["wav", "mp3"])

if uploaded:
        audio_bytes = uploaded.read()
        st.audio(audio_bytes, format="audio/wav")

        if st.button("Определить пол"):
            with st.spinner("Анализ..."):
                # Повторяем предобработку
                audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=22050)
                audio = torch.from_numpy(audio).to(device)

                with torch.no_grad():
                    mel = mel_transform(audio.unsqueeze(0))[:, :, :400]
                    if mel.shape[2] < 400:
                        mel = torch.nn.functional.pad(mel, (0, 400 - mel.shape[2]))
                    logits = model(mel)
                    prob = torch.softmax(logits, dim=1)[0]
                    pred = torch.argmax(prob).item()
                    conf = prob[pred].item()

                gender = "Мужской 👨" if pred == 0 else "Женский 👩"
                st.success(f"{gender} — уверенность {conf:.1%}")

# --- FastAPI ---
# app = FastAPI(title="Голос → Пол (Кыргизский)")
#
# class Prediction(BaseModel):
#     gender: str
#     confidence: float
#
# @app.post("/predict", response_model=Prediction)
# async def predict(file: UploadFile = File(...)):
#     audio_bytes = await file.read()
#     audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=22050)
#     audio = torch.from_numpy(audio).to(device)
#
#     with torch.no_grad():
#         mel = mel_transform(audio.unsqueeze(0))[:, :, :400]
#         if mel.shape[2] < 400:
#             mel = torch.nn.functional.pad(mel, (0, 400 - mel.shape[2]))
#         logits = model(mel)
#         prob = torch.softmax(logits, dim=1)[0]
#         pred = torch.argmax(prob).item()
#         confidence = prob[pred].item()
#
#     gender = "Мужской" if pred == 0 else "Женский"
#     return Prediction(gender=gender, confidence=round(confidence, 3))

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)