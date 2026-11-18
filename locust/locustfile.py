"""Flood test: locust -f locust/locustfile.py --host http://localhost:8080"""
import random
from pathlib import Path

from locust import HttpUser, between, task

IMAGES = [p.read_bytes() for p in list((Path(__file__).resolve().parents[1] / "data" / "test").rglob("*.jp*g"))[:20]]


class XrayUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        img = random.choice(IMAGES)
        self.client.post("/predict", files={"file": ("xray.jpeg", img, "image/jpeg")})

    @task(1)
    def health(self):
        self.client.get("/health")
