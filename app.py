import streamlit as st
import os
from PIL import Image
import tempfile

from preprocess_image import do_preprocess
from predict import make_predictions

st.title("Image to LaTeX Converter")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as input_temp:
        input_temp.write(uploaded_file.read())
        input_path = input_temp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as output_temp:
        output_path = output_temp.name
    do_preprocess(input_path, output_path)
    st.image(output_path, caption='Preprocessed Image', use_column_width=True)
    latex_code = make_predictions(output_path)
    st.subheader("Predicted LaTeX Code:")
    st.code(latex_code, language="latex")

