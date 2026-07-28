# Intel Image Scene Classification — ML Pipeline

## Project Description
This project implements an end-to-end machine learning pipeline for classifying
natural scene images into 6 categories: buildings, forest, glacier, mountain,
sea, and street. It uses transfer learning (MobileNetV2) trained on the
[Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification),
served via a FastAPI backend, with a Streamlit UI for predictions, data
visualization, and triggering model retraining on newly uploaded data.

## Video Demo


## Live URL
- API: https://intel-image-classification-atlz.onrender.com
- UI: https://intel-image-classification-v7qfqm4rdqjypkhqm4pg5e.streamlit.app/
  

## Note: 

The API is hosted on Render's free tier, which spins down after 
> 15 minutes of inactivity. The first request may take 30-60 seconds while 
> it wakes up — this also explains some of the response-time variance seen 
> in the load test results below.

intel-image-classification/
│

├── README.md
├── requirements.txt
│

├── notebook/
│   └── intel_image_classification.ipynb
│

├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
│

├── api/
│   └── main.py
│

├── ui/
│   ├── app.py
│   └── requirements.txt
│

├── docker/
│   ├── Dockerfile
│   └── requirements-api.txt
│

├── locust/
│   └── locustfile.py
│

├── locust_results/
│   ├── locust_results_1_container.png
│   ├── locust_results_2_container.png
│   └── locust_results_3_container.png
│

├── data/
│   ├── train/
│   │   ├── buildings/
│   │   ├── forest/
│   │   ├── glacier/
│   │   ├── mountain/
│   │   ├── sea/
│   │   └── street/
│   ├── test/
│   │   ├── buildings/
│   │   ├── forest/
│   │   ├── glacier/
│   │   ├── mountain/
│   │   ├── sea/
│   │   └── street/
│   └── retrain_uploads/
│

└── models/
    ├── intel_cnn_model.h5
    └── class_indices.json

    
## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Batonicarla/intel-image-classification.git
cd intel-image-classification
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
venv\Scripts\activate       
source venv/bin/activate    
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
| Test Accuracy | 0.9020 |
| Test Loss | 0.2635 |
| Precision (weighted) | 0.9034 |
| Recall (weighted) |  0.9020 |
| F1-score (weighted) | 0.9018 |

## Data Insights
1. **Class distribution**: Classes are fairly balanced (~2,000–2,500 images
   each in training), so accuracy is a reliable metric without needing class
   weighting.
2. **Visual similarity**: Classes like glacier/mountain and sea/buildings
   (coastal cityscapes) are visually similar and the most common source of
   misclassification  visible in the confusion matrix.
3. **Brightness patterns**: Sea and glacier images have higher average pixel
   brightness (reflective water/ice/sky), while forest images are darkest 
   showing that color/luminance alone carries meaningful class signal.

## Results from Flood Request Simulation (Locust)

Load tested with 20 concurrent users, spawn rate of 5/sec, against 3 separately
running Docker containers (each an independent instance of the API):

| Containers | Avg Response Time (ms) | 95th Percentile (ms) |Peak  RPS | Failures |
|---|---|---|---|---|
| container 1 (port 8000) | 13,000  | 20,000–58,000 |  3.0  | Some, during load spikes |
| container 2 (port 8001) | 25,000–30,000  | 40,000–90,000 | 1.8 | Some, during load spikes |
| container 3 (port 8002) | 15,000–30,000 | 30,000–70,000 | 3.0 | Some, during load spikes|

### 1 Container
![1 container results](locust_results/locust_results_1_container.png)

### 2 Containers
![2 container results](locust_results/locust_results_2_container.png)

### 3 Containers
![3 container results](locust_results/locust_results_3_container.png)

**Observations**: 

Average response times ranged from roughly 13,000-30,000ms 
across the three containers, with 95th percentile response times spiking as 
high as 58,000-90,000ms under peak load. These relatively high latencies are 
expected given that each container runs TensorFlow CNN inference on CPU only 
(no GPU acceleration), which is significantly slower under concurrent load 
than typical lightweight APIs  every `/predict` request requires a full 
forward pass through the MobileNetV2-based model. Some request failures 
occurred during peak load periods across all three containers, likely due to 
timeouts under sustained high concurrency combined with CPU-bound inference. 
In a production setup, adding a load balancer to distribute the 20 concurrent 
users across all three containers simultaneously, along with GPU acceleration 
or a lighter-weight model architecture, would significantly reduce both 
average and tail latency.

## Retraining Pipeline
Users can upload new labeled images via the UI's "Upload & Retrain" tab. The
system:
1. Saves uploaded images into class-labeled folders
2. Checks against a minimum image threshold to trigger retraining
3. Loads the **existing trained model** as a pretrained base (not from scratch)
4. Fine-tunes on the new data with a lower learning rate
5. Saves the updated model, replacing the previous version

## Author
Carla BATONI
African Leadership University — BSE Machine Learning Pipeline Summative
