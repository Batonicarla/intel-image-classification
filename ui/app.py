import streamlit as st
import requests
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

API_URL = "https://intel-image-classification-atlz.onrender.com"

st.set_page_config(page_title="Intel Scene Classifier", layout="wide")

st.title(" Intel Image Scene Classification")
st.caption("End-to-end ML pipeline — predict, visualize, retrain")

tab1, tab2, tab3, tab4 = st.tabs([" Status", " Data Insights", " Predict", " Upload & Retrain"])

# ---------------------------------------------------------------
# TAB 1: Model Up-time / Status
# ---------------------------------------------------------------
with tab1:
    st.header("Model Health & Uptime")

    if st.button("Refresh Status"):
        st.rerun()

    try:
        resp = requests.get(f"{API_URL}/health", timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            col1, col2, col3 = st.columns(3)
            col1.metric("Status", data["status"].upper())
            col2.metric("Uptime", data["uptime_readable"])
            col3.metric("Model Loaded", "✅" if data["model_loaded"] else "❌")
        else:
            st.error(f"API returned status code {resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure it's running at " + API_URL)
    except requests.exceptions.ReadTimeout:
        st.warning("The API is waking up (Render free tier cold start). Please wait a moment and click Refresh Status.")

# ---------------------------------------------------------------
# TAB 2: Data Visualizations
# ---------------------------------------------------------------
with tab2:
    st.header("Dataset Insights")

    train_dir = st.text_input("Path to training data (class subfolders)", "data/train")

    if os.path.isdir(train_dir):
        classes = sorted([c for c in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, c))])

        if classes:
            counts = {c: len(os.listdir(os.path.join(train_dir, c))) for c in classes}

            st.subheader("1. Class Distribution")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="viridis", ax=ax1)
            plt.xticks(rotation=30)
            st.pyplot(fig1)
            st.info("Classes are fairly balanced, so accuracy is a trustworthy metric here — no heavy class weighting needed.")

            st.subheader("2. Sample Images per Class")
            cols = st.columns(len(classes))
            for col, cls in zip(cols, classes):
                folder = os.path.join(train_dir, cls)
                imgs = os.listdir(folder)
                if imgs:
                    img = Image.open(os.path.join(folder, imgs[0]))
                    col.image(img, caption=cls, use_column_width=True)
            st.info("Visually similar classes (e.g. glacier vs mountain, sea vs buildings) are the most likely sources of confusion.")

            st.subheader("3. Class Sample Counts Table")
            df_counts = pd.DataFrame(list(counts.items()), columns=["Class", "Image Count"])
            st.dataframe(df_counts, use_container_width=True)
        else:
            st.warning("No class subfolders found in that path.")
    else:
        st.warning("That path doesn't exist. Point it to your local training data folder.")

# ---------------------------------------------------------------
# TAB 3: Prediction
# ---------------------------------------------------------------
with tab3:
    st.header("Predict a Single Image")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)

        if st.button("Predict"):
            with st.spinner("Sending to model..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                try:
                    resp = requests.post(f"{API_URL}/predict", files=files, timeout=90)
                    if resp.status_code == 200:
                        result = resp.json()
                        st.success(f"Predicted class: **{result['predicted_class']}**")
                        st.metric("Confidence", f"{result['confidence']*100:.2f}%")

                        st.subheader("Full Probability Breakdown")
                        probs_df = pd.DataFrame(
                            list(result["all_probabilities"].items()),
                            columns=["Class", "Probability"]
                        ).sort_values("Probability", ascending=False)
                        st.bar_chart(probs_df.set_index("Class"))
                    else:
                        st.error(f"Prediction failed: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the API.")
                except requests.exceptions.ReadTimeout:
                    st.warning("The API is waking up (Render free tier cold start). Please wait a moment and try again.")

# ---------------------------------------------------------------
# TAB 4: Bulk Upload + Retrain Trigger
# ---------------------------------------------------------------
with tab4:
    st.header("Upload New Data & Retrain")

    class_name = st.selectbox(
        "Select the class these images belong to",
        ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    )

    bulk_files = st.file_uploader(
        "Upload multiple images for retraining",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if bulk_files and st.button("Upload Files"):
        with st.spinner(f"Uploading {len(bulk_files)} files..."):
            files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in bulk_files]
            try:
                resp = requests.post(
                    f"{API_URL}/upload?class_name={class_name}",
                    files=files_payload,
                    timeout=120
                )
                if resp.status_code == 200:
                    st.success(resp.json()["message"])
                else:
                    st.error(f"Upload failed: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API.")
            except requests.exceptions.ReadTimeout:
                st.warning("The API is waking up (Render free tier cold start). Please wait a moment and try again.")

    st.divider()

    st.subheader("Trigger Retraining")
    epochs = st.slider("Epochs", min_value=1, max_value=15, value=5)
    min_images = st.number_input("Minimum new images required to trigger", min_value=1, value=20)

    if st.button("🔁 Start Retraining"):
        with st.spinner("Retraining in progress — this may take a while..."):
            try:
                resp = requests.post(
                    f"{API_URL}/retrain?epochs={epochs}&min_new_images={min_images}",
                    timeout=600
                )
                if resp.status_code == 200:
                    result = resp.json()
                    if result["retrained"]:
                        st.success("Retraining complete!")
                        st.metric("Epochs Run", result["epochs_run"])
                        st.metric("Final Train Accuracy", f"{result['final_train_accuracy']*100:.2f}%")
                        st.metric("Final Val Accuracy", f"{result['final_val_accuracy']*100:.2f}%")
                    else:
                        st.warning(result["message"])
                else:
                    st.error(f"Retraining failed: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API.")
            except requests.exceptions.ReadTimeout:
                st.warning("The API is waking up (Render free tier cold start). Please wait a moment and try again.")