import requests
import streamlit as st
import pandas as pd
from models import SUPPORTED_MODELS, bytes_to_array, prepare_image
import torch
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI API client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

st.set_page_config(layout="wide")
st.title(":camera: Computer vision app")

# Let user upload pictures
with st.sidebar:
    st.title("Upload pictures")

    upload_type = st.radio(
        label="How to upload the pictures",
        options=["From file", "From URL", "From webcam"],
    )

    image_bytes_list = []

    if upload_type == "From file":
        files = st.file_uploader("Upload image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if files:
            image_bytes_list = [file.getvalue() for file in files]

    if upload_type == "From URL":
        urls = st.text_area("Paste URLs (one per line)")
        if urls:
            urls_list = urls.split("\n")
            image_bytes_list = [requests.get(url.strip()).content for url in urls_list if url.strip()]

    if upload_type == "From webcam":
        st.warning("Multiple image upload from webcam is not supported.")
        camera = st.camera_input("Take a picture!")
        if camera:
            image_bytes_list = [camera.getvalue()]

if not image_bytes_list:
    st.warning("👈 Please upload images first...")
    st.stop()

st.write("## Model predictions")

# Add a progress bar
progress_bar = st.progress(0)
total_steps = len(image_bytes_list)
current_step = 0

def generate_poem(prompt):
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a creative and poetic assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=150)
    return response.choices[0].message.content.strip()

for image_index, image_bytes in enumerate(image_bytes_list):
    st.write(f"### Processing Image {image_index + 1}")
    columns = st.columns([1, 2, 3])
    with columns[0]:
        st.image(image_bytes, width=200)

    all_predictions = []

    for model_name in SUPPORTED_MODELS.keys():
        load_model, input_size, preprocess_input, decode_predictions = SUPPORTED_MODELS[model_name].values()

        # Load the model
        model = load_model()
        image_array = bytes_to_array(image_bytes)
        image_array = prepare_image(image_array, input_size, preprocess_input)
        prediction = model.predict(image_array)
        prediction_df = pd.DataFrame(decode_predictions(prediction, 5)[0])
        prediction_df.columns = ["label_id", "label", "probability"]
        prediction_df["model"] = model_name

        all_predictions.append(prediction_df)

        # Clear memory
        del model
        #torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Combine predictions for the current image
    if all_predictions:
        image_predictions_df = pd.concat(all_predictions, ignore_index=True)
        image_predictions_df = image_predictions_df.sort_values(by="probability", ascending=False)
        with columns[1]:
            st.dataframe(image_predictions_df, height=400)
    else:
        with columns[1]:
            st.write("No valid predictions were generated for this image.")

    # Check if at least 3 models have a probability greater than 0.5
    if not image_predictions_df.empty:
        high_prob_predictions = image_predictions_df[image_predictions_df["probability"] > 0.5]
        if len(high_prob_predictions) >= 3:
            # Use the highest probability label to generate a poem
            top_label = high_prob_predictions.iloc[0]["label"]

            input_text = f"Write a haiku about {top_label}, it should have a newline after each comma"
            poem = generate_poem(input_text)

            with columns[2]:
                st.write("### Poem")
                st.write(poem)
        else:
            with columns[2]:
                st.write("Not enough high-probability predictions to generate a poem.")
    else:
        with columns[2]:
            st.write("No predictions to analyze for poem generation.")

    # Update progress bar
    current_step += 1
    progress_bar.progress(current_step / total_steps)
