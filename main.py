# main.py
import io
import torch
import librosa
import uvicorn
import numpy as np
import torchaudio
import streamlit as st
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile

# --- Модель (остаётся без изменений) ---
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16Gender().to(device)
model.load_state_dict(torch.load("kyrgyz_vgg16.pth", map_location=device))
model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=22050, n_fft=1024, hop_length=512, n_mels=128).to(device)

# --- Streamlit интерфейс ---
st.title("Распознавание пола по голосу (киргизский)")

st.write("### Вариант 1: Загрузить файл")
uploaded_file = st.file_uploader("Выбери аудио (wav, mp3)", type=["wav", "mp3"])

st.write("### Вариант 2: Запись с микрофона")
recorded_audio = st.audio_input("Нажми и говори")

# Выбираем, какое аудио использовать
audio_bytes = None
if recorded_audio:
    audio_bytes = recorded_audio.read()
    st.audio(audio_bytes, format="audio/wav")
elif uploaded_file:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format="audio/wav")

if audio_bytes and st.button("🔊 Определить пол"):
    with st.spinner("Анализирую голос..."):
        # Предобработка
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        audio_tensor = torch.from_numpy(audio).to(device)

        with torch.no_grad():
            mel = mel_transform(audio_tensor.unsqueeze(0))[:, :, :400]
            if mel.shape[2] < 400:
                mel = torch.nn.functional.pad(mel, (0, 400 - mel.shape[2]))
            logits = model(mel)
            prob = torch.softmax(logits, dim=1)[0]
            pred = torch.argmax(prob).item()
            conf = prob[pred].item()

        gender = "Мужской 👨" if pred == 0 else "Женский 👩"
        st.success(f"**{gender}** — уверенность {conf:.1%}")

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