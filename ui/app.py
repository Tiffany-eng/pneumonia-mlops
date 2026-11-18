"""Streamlit UI. One small function per page."""
import base64
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Pneumonia X-ray Classifier", page_icon="🫁", layout="wide")


def call(method, path, **kw):
    """Small wrapper so every page handles an unreachable API the same way."""
    try:
        return requests.request(method, f"{API}{path}", timeout=kw.pop("timeout", 300), **kw)
    except requests.RequestException as e:
        st.error(f"API unreachable: {e}")
        return None


def show_error(r):
    try:
        st.error(r.json().get("detail", r.text))
    except ValueError:
        st.error(r.text)


# ------------------------------------------------------------------ Predict
def page_predict():
    st.header("Predict from chest X-rays")
    single, batch = st.tabs(["Single image", "Batch"])

    with single:
        f = st.file_uploader("Upload one X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"], key="single")
        if f:
            c1, c2 = st.columns(2)
            c1.image(f, caption=f.name, use_container_width=True)
            if c2.button("Run prediction", type="primary"):
                r = call("post", "/predict", files={"file": (f.name, f.getvalue(), f.type)})
                if r is not None and r.ok:
                    out = r.json()
                    color = "red" if out["label"] == "PNEUMONIA" else "green"
                    c2.markdown(f"### Prediction: :{color}[{out['label']}]")
                    c2.metric("Confidence", f"{out['confidence'] * 100:.1f}%")
                    c2.progress(out["probability_pneumonia"], text="P(pneumonia)")
                elif r is not None:
                    show_error(r)

    with batch:
        files = st.file_uploader("Upload several X-rays (max 32)", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True, key="batch")
        if files and st.button("Predict batch", type="primary"):
            r = call("post", "/predict/batch", files=[("files", (f.name, f.getvalue(), f.type)) for f in files])
            if r is not None and r.ok:
                df = pd.DataFrame(r.json()["results"])
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df["label"].value_counts())
            elif r is not None:
                show_error(r)


# ------------------------------------------------------------------ Insights
def show_samples():
    st.subheader("Sample images")
    r = call("get", "/samples")
    if r is not None and r.ok:
        for cls, images in r.json().items():
            st.markdown(f"**{cls}**")
            cols = st.columns(max(len(images), 1))
            for col, b64 in zip(cols, images):
                col.image(base64.b64decode(b64), use_container_width=True)


def show_feature(df_by_class, column, title, story):
    st.subheader(title)
    fig, ax = plt.subplots(figsize=(6, 3))
    for cls, df in df_by_class.items():
        ax.hist(df[column], bins=25, alpha=0.6, label=cls)
    ax.set_xlabel(column)
    ax.legend()
    st.pyplot(fig)
    means = {c: round(df[column].mean(), 1) for c, df in df_by_class.items()}
    st.caption(f"Mean {column}: {means}. {story}")


def page_insights():
    st.header("What does the training data look like?")
    show_samples()
    r = call("get", "/insights")
    if r is None or not r.ok:
        return
    d = r.json()
    counts = d["class_counts"]
    feats = {c: pd.DataFrame(rows) for c, rows in d["features"].items()}

    st.subheader("Class balance")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(counts.keys(), counts.values(), color=["#4c9be8", "#e8704c"])
    ax.set_ylabel("Images")
    st.pyplot(fig)
    total = max(sum(counts.values()), 1)
    st.info(f"Pneumonia makes up {counts['PNEUMONIA'] / total:.0%} of the training set. A model that always "
            "says 'pneumonia' would score well on accuracy alone, so we use class weights and judge the "
            "model on recall, precision, F1 and AUC.")
    show_feature(feats, "brightness", "Brightness (mean pixel intensity)",
                 "Fluid and infiltrates make pneumonia lungs look whiter and denser.")
    show_feature(feats, "contrast", "Contrast (pixel standard deviation)",
                 "Contrast reflects how clearly lung tissue separates from ribs and background.")
    show_feature(feats, "width", "Image width (pixels)",
                 "Scan sizes vary a lot, which is why every image is resized to 224x224.")


# ------------------------------------------------------------------ Upload, train, retrain
def upload_section(min_needed):
    st.subheader("1. Upload new data")
    st.caption(f"Retraining needs at least {min_needed} new images. Use a good number of samples from both classes.")
    zip_tab, files_tab = st.tabs(["Zip file (recommended)", "Loose images"])
    with zip_tab:
        st.caption("A .zip with two folders: NORMAL/ and PNEUMONIA/")
        z = st.file_uploader("Select a zip", type=["zip"], key="zip")
        if z and st.button("Upload zip to database"):
            r = call("post", "/upload/zip", files={"file": (z.name, z.getvalue(), "application/zip")})
            if r is not None and r.ok:
                res = r.json()
                st.success(f"Saved {res['total_saved']} images {res['saved']}; skipped {res['skipped']}. "
                           f"Pending for training: {res['pending_for_training']}")
                if res["auto_retrain_started"]:
                    st.info("Auto-retrain threshold reached, retraining started.")
            elif r is not None:
                show_error(r)
    with files_tab:
        label = st.selectbox("Class of these images", ["NORMAL", "PNEUMONIA"])
        files = st.file_uploader("Select multiple X-rays", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True, key="loose")
        if files and st.button("Upload images to database"):
            r = call("post", "/upload", data={"label": label},
                     files=[("files", (f.name, f.getvalue(), f.type)) for f in files])
            if r is not None and r.ok:
                res = r.json()
                st.success(f"Saved {res['saved']} images. Pending for training: {res['pending_for_training']}")
                if res["rejected"]:
                    st.warning(f"Rejected invalid files: {res['rejected']}")
            elif r is not None:
                show_error(r)


def training_section(info):
    st.subheader("2. Train or retrain")
    st.write(f"**Pending uploads:** {info['pending_uploads']} (need {info['min_to_retrain']}) | "
             f"**State:** {info['state']} | {info['message']}")
    c1, c2 = st.columns(2)
    if c1.button("Retrain on uploaded data", type="primary"):
        r = call("post", "/retrain")
        if r is not None and r.ok:
            st.success("Retraining started")
        elif r is not None:
            show_error(r)
    if c2.button("Train from scratch (data/train)"):
        r = call("post", "/train")
        if r is not None and r.ok:
            st.success("Training started")
        elif r is not None:
            show_error(r)
    if info["history"]:
        st.subheader("Training history")
        st.dataframe(pd.DataFrame(info["history"]), use_container_width=True)
    if info["state"] == "running":
        time.sleep(5)
        st.rerun()


def page_upload_retrain():
    st.header("Upload new data, train and retrain")
    s = call("get", "/retrain/status")
    if s is None or not s.ok:
        return
    info = s.json()
    upload_section(info["min_to_retrain"])
    st.divider()
    training_section(info)


# ------------------------------------------------------------------ Monitoring
def page_monitoring():
    st.header("Model & service monitoring")
    h, m = call("get", "/health"), call("get", "/metrics")
    if h is not None and h.ok:
        hd = h.json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Status", hd["status"].upper())
        c2.metric("Uptime", f"{hd['uptime_seconds'] / 60:.1f} min")
        c3.metric("Training running", "Yes" if hd["retraining"] else "No")
    if m is not None and m.ok:
        md = m.json()
        sv = md["service"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Requests", sv["requests"])
        c2.metric("Predictions", sv["predictions"])
        c3.metric("Avg latency", f"{sv['avg_latency_ms']} ms")
        if md["model"]:
            mm = md["model"]
            st.subheader("Model performance on the test set")
            for col, k in zip(st.columns(5), ["accuracy", "precision", "recall", "f1", "auc"]):
                col.metric(k.upper(), f"{mm[k]:.3f}")
            st.dataframe(pd.DataFrame(mm["confusion_matrix"], index=["True NORMAL", "True PNEUMONIA"],
                                      columns=["Pred NORMAL", "Pred PNEUMONIA"]))
        else:
            st.warning("No model metrics yet. Train the model first.")
    st.button("Refresh")


PAGES = {"Predict": page_predict, "Data insights": page_insights,
         "Upload, train & retrain": page_upload_retrain, "Monitoring": page_monitoring}
st.sidebar.title("🫁 Pneumonia Detector")
PAGES[st.sidebar.radio("Go to", list(PAGES))]()
st.sidebar.caption("Educational demo. Not a medical device.")
