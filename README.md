# Intel Image Scene Classification — ML Pipeline

## Project Description
This project implements an end-to-end machine learning pipeline for classifying
natural scene images into 6 categories: buildings, forest, glacier, mountain,
sea, and street. It uses transfer learning (MobileNetV2) trained on the
[Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification),
served via a FastAPI backend, with a Streamlit UI for predictions, data
visualization, and triggering model retraining on newly uploaded data.

## Video Demo
🎥 [Watch the demo on YouTube](PASTE_YOUR_YOUTUBE_LINK_HERE)

## Live URL
- API: [PASTE_URL_IF_DEPLOYED or "Not publicly deployed — run locally, see setup below"]
- UI: [PASTE_URL_IF_DEPLOYED]

## Project Structure
intel-image-classification/
│
├── README.md
├── notebook/
│   └── intel_image_classification.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
├── data/
│   ├── train/
│   └── test/
├── models/
│   └── intel_cnn_model.h5
├── api/
│   └── main.py
├── ui/
│   └── app.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── locust/
│   └── locustfile.py
└── requirements.txt
## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Batonicarla/intel-image-classification.git
cd intel-image-classification
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
pip install -r requirements.txt
```

### 3. Run the API
```bash
uvicorn api.main:app --reload
```
API docs available at `http://localhost:8000/docs`

### 4. Run the UI (in a separate terminal)
```bash
streamlit run ui/app.py
```
Opens at `http://localhost:8501`

### 5. Run with Docker (containerized API)
```bash
docker build -t intel-classifier-api -f docker/Dockerfile .
docker run -d -p 8000:8000 --name api1 intel-classifier-api
```

### 6. Load testing with Locust
```bash
locust -f locust/locustfile.py --host http://localhost:8000
```
Open `http://localhost:8089`, set users/spawn rate, and start swarming.

## Model Details
- **Architecture**: MobileNetV2 (transfer learning, frozen base) + custom
  classification head (GlobalAveragePooling → Dense(128, relu) → Dropout(0.4)
  → Dense(6, softmax))
- **Optimization techniques**: Pretrained ImageNet weights, Dropout
  regularization, Early Stopping, ReduceLROnPlateau
- **Evaluation metrics**: Accuracy, Loss, Precision, Recall, F1-score
  (see notebook for full classification report and confusion matrix)

| Metric | Score |
|---|---|
| Test Accuracy | [FILL IN] |
| Test Loss | [FILL IN] |
| Precision (weighted) | [FILL IN] |
| Recall (weighted) | [FILL IN] |
| F1-score (weighted) | [FILL IN] |

## Data Insights
1. **Class distribution**: Classes are fairly balanced (~2,000–2,500 images
   each in training), so accuracy is a reliable metric without needing class
   weighting.
2. **Visual similarity**: Classes like glacier/mountain and sea/buildings
   (coastal cityscapes) are visually similar and the most common source of
   misclassification — visible in the confusion matrix.
3. **Brightness patterns**: Sea and glacier images have higher average pixel
   brightness (reflective water/ice/sky), while forest images are darkest —
   showing that color/luminance alone carries meaningful class signal.

## Results from Flood Request Simulation (Locust)

Load tested with 20 concurrent users, spawn rate of 5/sec, against 3 separately
running Docker containers (each an independent instance of the API):

| Containers | Avg Response Time (ms) | Median (ms) | RPS | Failures |
|---|---|---|---|---|
| 1 (port 8000) | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| 1 (port 8001) | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |
| 1 (port 8002) | [FILL IN] | [FILL IN] | [FILL IN] | [FILL IN] |

Screenshots: see `locust_results/1_container/`, `locust_results/2_container/`,
`locust_results/3_container/`

**Observations**: [Write 2-3 sentences — e.g. did response time stay stable
across containers? Any failures? What would change with a load balancer
distributing traffic across multiple containers simultaneously?]

## Retraining Pipeline
Users can upload new labeled images via the UI's "Upload & Retrain" tab. The
system:
1. Saves uploaded images into class-labeled folders
2. Checks against a minimum image threshold to trigger retraining
3. Loads the **existing trained model** as a pretrained base (not from scratch)
4. Fine-tunes on the new data with a lower learning rate
5. Saves the updated model, replacing the previous version

## Author
[Your name]
African Leadership University — BSE Machine Learning Pipeline Summative