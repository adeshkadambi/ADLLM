"""
Build UI in streamlit where you can upload a results file, take a 
random stratified sample of 25 videos per class (25*7 = 175 videos). 
For each video, show the grid and the reasoning side by side. 
Assign a rating and any notes. Save the results.
"""

import json

import streamlit as st


def load_data(f):
    return json.load(f)


file = st.file_uploader("Upload results file", type=["json"])

if file:
    data = load_data(file)
    st.toast("File loaded successfully", icon=":material/verified:")

    st.write(data)