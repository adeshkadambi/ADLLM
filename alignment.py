"""
Build UI in streamlit where you can upload a results file, take a 
random stratified sample of 25 videos per class (25*7 = 175 videos). 
For each video, show the grid and the reasoning side by side. 
Assign a rating and any notes. Save the results.
"""

import json
import os
from typing import Any

import polars as pl
import streamlit as st

from evaluation import get_true_label


def random_stratified_sample(df: pl.DataFrame, n: int = 25):
    """Takes a random stratified sample of n videos per class."""
    return df.filter(pl.int_range(pl.len()).shuffle().over("true_label") < n)


def load_data(f, batch_path: str):
    """Extracts the data from the uploaded file."""
    results_dict = json.load(f)
    data = []

    for file_path, result in results_dict.items():
        grid_path = os.path.join(batch_path, result["grid_path"])
        true_label = get_true_label(file_path)
        data.append(
            {
                "file_path": file_path,
                "true_label": true_label,
                "pred_label": result["prediction"],
                "grid_path": grid_path,
                "tags": result["tags"],
                "reasoning": result["reasoning"],
            }
        )

    return pl.DataFrame(data)


def initialize_state():
    """Initialize session state variables."""
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "sampled_data" not in st.session_state:
        st.session_state.sampled_data = None
    if "ratings" not in st.session_state:
        st.session_state.ratings = {}


def save_rating(file_path: str, data: dict[str, Any], rating: int):
    """Save rating for current sample."""
    st.session_state.ratings[file_path] = {
        "true_label": data["true_label"].item(),
        "pred_label": data["pred_label"].item(),
        "tags": data["tags"].item().to_list(),
        "reasoning": data["reasoning"].item().to_list(),
        "rating": rating + 1,  # Convert 0-4 to 1-5
    }


def save_results(batch_path: str):
    """Save all ratings to file."""

    # if there are polars series in the ratings, use .item() to convert them to python objects
    for file_paths, rating_data in st.session_state.ratings.items():
        if isinstance(rating_data["tags"], pl.Series):
            rating_data["tags"] = rating_data["tags"].item().to_list()
        if isinstance(rating_data["reasoning"], pl.Series):
            rating_data["reasoning"] = rating_data["reasoning"].item().to_list()
        if isinstance(rating_data["true_label"], pl.Series):
            rating_data["true_label"] = rating_data["true_label"].item()
        if isinstance(rating_data["pred_label"], pl.Series):
            rating_data["pred_label"] = rating_data["pred_label"].item()

        st.session_state.ratings[file_paths] = rating_data

    output_path = os.path.join(batch_path, "alignment.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(st.session_state.ratings, f)
    return output_path


def display_ratings_summary():
    """Display summary of all ratings in an expander."""
    with st.expander("View All Ratings", expanded=False):
        if not st.session_state.ratings:
            st.info("No ratings yet")
            return

        for file_path, rating_data in st.session_state.ratings.items():
            col1, col2, col3 = st.columns([0.4, 0.4, 0.2])
            with col1:
                st.write(f"True: `{rating_data['true_label']}`")
                st.write(f"Pred: `{rating_data['pred_label']}`")
            with col2:
                st.write("Tags:", rating_data["tags"])
            with col3:
                st.write(f"Rating: {rating_data['rating']}/5")
            st.divider()


def next_sample(batch_path):
    if st.session_state.current_index < len(st.session_state.sampled_data) - 1:
        st.session_state.current_index += 1
    else:
        save_path = save_results(batch_path)
        st.success(f"All ratings saved to {save_path}")
        st.balloons()


def run():
    st.set_page_config(
        page_title="Manual Evaluater",
        page_icon=":memo:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    initialize_state()

    st.title("Reasoning Alignment Evaluator")

    file = st.file_uploader("Upload results file", type=["json"])
    batch_path = st.text_input(
        "Enter the path to the batch folder",
        value="/media/adeshkadambi/WD_BLACK/PhD/adl_recognition/results/batch_20241118_011946",
    )

    if file and st.session_state.sampled_data is None:
        data = load_data(file, batch_path)
        st.session_state.sampled_data = random_stratified_sample(data)
        st.toast("File loaded and sampled successfully", icon="✅")

    if st.session_state.sampled_data is not None:
        current_sample = st.session_state.sampled_data[st.session_state.current_index]

        st.progress(st.session_state.current_index / len(st.session_state.sampled_data))

        col1, col2 = st.columns([0.6, 0.4])

        with col1:
            st.image(current_sample["grid_path"].item())
            feedback = st.feedback(
                "stars", key=f"feedback_{st.session_state.current_index}"
            )

            if feedback is not None:
                save_rating(
                    current_sample["file_path"].item(), current_sample, feedback
                )

            with st.expander("Feedback Guidelines", expanded=True):
                st.markdown(
                    """
                    - 5 is perfect, including classification.
                    - 4 is good reasoning AND tags, incorrect classification.
                    - 3 is good reasoning OR tags, incorrect classification.
                    - 2 is some relevant tags OR reasoning.
                    - 1 is completely irrelevant.
                """
                )

            st.button("Next", on_click=next_sample, args=(batch_path,))

        with col2:
            st.write(f"### True Label: `{current_sample['true_label'].item()}`")
            st.write(f"### Pred Label: `{current_sample['pred_label'].item()}`")
            st.write("Tags")
            st.write(current_sample["tags"].item().to_list())
            st.write("Reasoning")
            st.write(current_sample["reasoning"].item().to_list())

        # Display ratings summary at bottom
        display_ratings_summary()

        with st.expander("Debug", expanded=False):
            st.write(st.session_state.ratings)


if __name__ == "__main__":
    run()
