from locust import HttpUser, task, between
import os
import random

SAMPLE_IMAGES_DIR = "data/test"

def get_sample_images():
    images = []
    if os.path.isdir(SAMPLE_IMAGES_DIR):
        for cls in os.listdir(SAMPLE_IMAGES_DIR):
            cls_dir = os.path.join(SAMPLE_IMAGES_DIR, cls)
            if os.path.isdir(cls_dir):
                for f in os.listdir(cls_dir):
                    images.append(os.path.join(cls_dir, f))
    return images

SAMPLE_IMAGES = get_sample_images()


class PredictUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict(self):
        if not SAMPLE_IMAGES:
            return
        img_path = random.choice(SAMPLE_IMAGES)
        with open(img_path, "rb") as f:
            self.client.post(
                "/predict",
                files={"file": (os.path.basename(img_path), f, "image/jpeg")}
            )

    @task(1)
    def health(self):
        self.client.get("/health")