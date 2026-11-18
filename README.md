# Pneumonia Detection from Chest X-rays: End-to-End ML Pipeline

Binary image classifier (NORMAL vs PNEUMONIA) with an API, UI, bulk-upload retraining, Docker scaling and a Locust flood test.

- **Video demo (YouTube):** _add link_
- **Live URL:** _add link_
- **GitHub:** _add link_

## Project description
Chest X-rays are classified with a MobileNetV2 transfer-learning model. The system covers data acquisition, preprocessing, training, testing, retraining on user-uploaded data (manual button + automatic threshold trigger), a FastAPI service, a Streamlit dashboard (prediction, data insights, upload/retrain, uptime monitoring) and cloud deployment behind an nginx load balancer.

> Educational project, not a medical device.

## Features
- **Single and batch prediction** (`/predict`, `/predict/batch`)
- **Data insights**: sample X-rays per class and 3+ feature charts with interpretations
- **Bulk upload** of a zip (folders `NORMAL/` and `PNEUMONIA/`) or loose images, validated and saved to SQLite + disk
- **Train from scratch** and **retrain** from the UI or API; retraining needs at least 20 new images and starts from the saved model
- **Automatic retrain trigger** at 50 pending images
- **Monitoring**: uptime, request count, latency, test metrics, training history
- **Docker + nginx** so the API can run as several containers; **Locust** flood test; optional **ONNX** export

## Repo structure
```
notebook/pneumonia_pipeline.ipynb   preprocessing, training, evaluation
src/
  preprocessing.py   dataset pipeline (small single-purpose functions)
  model.py           build / train / evaluate
  prediction.py      PneumoniaClassifier class (single + batch, auto-reloads new model)
  retrain.py         train_from_scratch() and retrain() with promotion rule
  database.py        SQLite (uploads, training history)
  insights.py        dataset statistics for the UI
  onnx_export.py     optional ONNX conversion + speed benchmark
  utils/             config, data_acquisition, validation, image_io, archive (zip ingest)
api/main.py          FastAPI service
ui/app.py            Streamlit UI (one function per page)
locust/              locustfile.py + summarize.py
deploy/terraform/    AWS infrastructure as code
DEPLOYMENT.md        cloud setup steps
data/train, data/test    class folders NORMAL / PNEUMONIA
models/pneumonia_model.h5   trained model
```

## API
| Endpoint | Purpose |
|---|---|
| `POST /predict` | one image |
| `POST /predict/batch` | up to 32 images |
| `POST /upload/zip` | zip of labelled folders, for retraining |
| `POST /upload` | loose images with one label |
| `POST /train` | train from scratch on `data/train` |
| `POST /retrain` | fine-tune the saved model on uploaded data |
| `GET /retrain/status`, `/health`, `/metrics`, `/insights`, `/samples` | status, uptime, metrics, data views |

## Setup
```bash
git clone <repo-url> && cd pneumonia-mlops
pip install -r requirements.txt
# 1. get data + train (notebook does both)
jupyter notebook notebook/pneumonia_pipeline.ipynb
# 2a. run locally
uvicorn api.main:app --port 8000
API_URL=http://localhost:8000 streamlit run ui/app.py
# 2b. or with Docker (API behind nginx on :8080, UI on :8501)
docker compose up --build --scale api=1
```
API docs: `http://localhost:8080/docs`

## Retraining
1. **Upload** a zip of labelled folders (UI → Upload, train & retrain). Every file is validated, saved to `data/uploads/<label>/` and registered in SQLite. At least 20 new images are required.
2. **Preprocess**: same resize/scale/augmentation pipeline as training.
3. **Retrain**: `src/retrain.py` loads `models/pneumonia_model.h5` (the custom pre-trained model), fine-tunes on uploads + original data, evaluates on the test set and promotes the new model only if recall does not drop.
Triggers: the **Trigger retraining** button, or automatically once `AUTO_RETRAIN_THRESHOLD` (default 50) new images are pending.

## Flood-request simulation (Locust)
```bash
docker compose down
docker compose up --build -d --scale api=N          # N = 1, 2, 4
locust -f locust/locustfile.py --host http://localhost:8080 --headless -u 100 -r 10 -t 2m --csv results_N
python locust/summarize.py results_1 results_2 results_4   # prints the table below
```
For the "1000 requests at once" case also run `-u 1000 -r 100 -t 2m`. If failures appear, report where the system starts to break.

| API containers | Users | Requests/s | Median latency (ms) | 95th pct (ms) | Failures |
|---|---|---|---|---|---|
| 1 | 100 | _fill_ | _fill_ | _fill_ | _fill_ |
| 2 | 100 | _fill_ | _fill_ | _fill_ | _fill_ |
| 4 | 100 | _fill_ | _fill_ | _fill_ | _fill_ |
| 4 | 1000 | _fill_ | _fill_ | _fill_ | _fill_ |

## Cloud deployment
See [DEPLOYMENT.md](DEPLOYMENT.md) (VM + docker compose, Terraform for AWS).

## Optional: ONNX
`pip install -r requirements-onnx.txt && python -m src.onnx_export` converts the model and prints Keras vs ONNX Runtime latency. Run the API with `USE_ONNX=1` to use it.

## Model results
Evaluated on the held-out test set (624 images: 234 Normal, 390 Pneumonia):

| Metric | Score | Note |
|---|---|---|
| **Accuracy** | 80.45% | Overall correct classifications (502 / 624) |
| **Loss** | 0.6493 | Binary crossentropy loss on test set |
| **Recall (Sensitivity)** | **99.49%** | 388 / 390 pneumonia cases detected (2 missed) |
| **Precision** | 76.38% | 388 true positives vs 120 false positives |
| **F1-Score** | 0.8641 | Harmonic mean of precision and recall for Pneumonia |
| **AUC-ROC** | **0.9624** | Area under the ROC curve |

```text
Confusion Matrix:
               Predicted Normal    Predicted Pneumonia
Actual Normal        114                  120
Actual Pneumonia       2                  388
```
