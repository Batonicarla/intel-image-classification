import os
import shutil
import json
import time
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.prediction import predict_image, get_model
from src.model import retrain_existing_model

app = FastAPI(title="Intel Image Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/intel_cnn_model.h5"
CLASS_INDICES_PATH = "models/class_indices.json"
UPLOAD_DIR = "data/retrain_uploads"
START_TIME = time.time()

os.makedirs(UPLOAD_DIR, exist_ok=True)

with open(CLASS_INDICES_PATH) as f:
    CLASS_INDICES = json.load(f)


@app.on_event("startup")
def load_model_on_startup():
    """Warms the model cache so the first prediction request isn't slow."""
    get_model(MODEL_PATH)
    print("Model loaded and ready.")


@app.get("/")
def root():
    return {"message": "Intel Image Classification API is running."}


@app.get("/health")
def health_check():
    """Used by the UI to display model up-time."""
    uptime_seconds = time.time() - START_TIME
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime_seconds, 2),
        "uptime_readable": str(datetime.utcfromtimestamp(uptime_seconds).strftime('%H:%M:%S')),
        "model_loaded": os.path.exists(MODEL_PATH)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts a single image and returns the predicted class + confidence."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict_image(temp_path, MODEL_PATH, CLASS_INDICES)
    finally:
        os.remove(temp_path)

    return JSONResponse(content=result)


@app.post("/upload")
async def upload_data(class_name: str, files: list[UploadFile] = File(...)):
    """
    Accepts bulk images for a given class, saved for future retraining.
    class_name must match one of the existing class folders.
    """
    valid_classes = list(CLASS_INDICES.keys())
    if class_name not in valid_classes:
        raise HTTPException(status_code=400, detail=f"class_name must be one of {valid_classes}")

    class_dir = os.path.join(UPLOAD_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    saved_files = []
    for file in files:
        dest = os.path.join(class_dir, file.filename)
        with open(dest, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)

    return {"message": f"{len(saved_files)} files saved to {class_name}.", "files": saved_files}


@app.post("/retrain")
async def trigger_retrain(epochs: int = 5, min_new_images: int = 20):
    try:
        history, retrained = retrain_existing_model(
            model_path=MODEL_PATH,
            new_data_dir=UPLOAD_DIR,
            epochs=epochs,
            min_new_images=min_new_images
        )

        if not retrained:
            return {"retrained": False, "message": "Not enough new data to trigger retraining."}

        from src import prediction
        prediction._cached_model = None

        final_accuracy = history.history.get('accuracy', [None])[-1]
        final_val_accuracy = history.history.get('val_accuracy', [None])[-1]

        return {
            "retrained": True,
            "epochs_run": len(history.history['loss']),
            "final_train_accuracy": final_accuracy,
            "final_val_accuracy": final_val_accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

    if not retrained:
        return {"retrained": False, "message": "Not enough new data to trigger retraining."}

    get_model.__wrapped__ if hasattr(get_model, "__wrapped__") else None
    # force cache refresh so /predict uses the newly retrained model
    from src import prediction
    prediction._cached_model = None

    final_accuracy = history.history.get('accuracy', [None])[-1]
    final_val_accuracy = history.history.get('val_accuracy', [None])[-1]

    return {
        "retrained": True,
        "epochs_run": len(history.history['loss']),
        "final_train_accuracy": final_accuracy,
        "final_val_accuracy": final_val_accuracy
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)