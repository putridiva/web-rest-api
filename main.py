from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np
import tensorflow as tf

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9000"],  # Sesuaikan dengan frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = tf.keras.models.load_model("my_model.keras")
class_names = ['organik', 'non-organic', 'berbahaya']

# Tempat menyimpan riwayat prediksi
history_data = []

# Struktur data untuk riwayat prediksi
class PredictRecord(BaseModel):
    filename: str
    predicted_class: str
    confidence: float

# Endpoint prediksi
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 224, 224, 3)

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        record = PredictRecord(
            filename=file.filename,
            predicted_class=predicted_class,
            confidence=round(confidence, 3)
        )
        history_data.append(record)

        return JSONResponse(record.dict())

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.get("/history")
async def get_history():
    return [record.dict() for record in history_data]

@app.delete("/history/{index}")
async def delete_history(index: int):
    if 0 <= index < len(history_data):
        deleted = history_data.pop(index)
        return {"message": "Data berhasil dihapus", "deleted": deleted}
    else:
        raise HTTPException(status_code=404, detail="Index tidak ditemukan")

@app.put("/history/{index}")
async def update_history(index: int, update: PredictRecord):
    if 0 <= index < len(history_data):
        history_data[index] = update
        return {"message": "Data berhasil diperbarui", "updated": update}
    else:
        raise HTTPException(status_code=404, detail="Index tidak ditemukan")
    
@app.get("/kategori")
def get_kategori():
    return [
        {"kategori": "organik", "harga_per_kg": 1000},
        {"kategori": "non-organic", "harga_per_kg": 1500},
        {"kategori": "berbahaya", "harga_per_kg": 5000},
    ]

