import json
import os
from collections import defaultdict
from textwrap import wrap

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils import compute_class_weight
from wordcloud import WordCloud


def get_true_label(file_path: str) -> str:
    """
    Extract the true label from a file path.

    Parameters:
    file_path (str): Path to a file containing the label

    Returns:
    str: The true label extracted from the file path
    """
    return (
        file_path.split("/")[0]
        .upper()
        .replace("-", " ")
        .replace("GROOMING HEALTH MANAGEMENT", "GROOMING AND HEALTH MANAGEMENT")
        .replace("MEAL PREPARATION CLEANUP", "MEAL PREPARATION AND CLEANUP")
        .replace("LEISURE OTHER ACTIVITIES", "LEISURE")
        .replace("SELF FEEDING", "FEEDING")
    )


def calculate_alignment_rate(alignment_path: str, threshold=3.5):
    """
    Calculate clinical reasoning alignment rate from 5-point scale scores.

    Parameters:
    alignment_path (str): Path to the alignment scores file
    threshold (float): Minimum score to be considered "aligned" (default 3.5)

    Returns:
    dict: Contains alignment rate and detailed statistics
    """

    with open(alignment_path, encoding="utf-8") as f:
        alignment_scores = json.load(f)

    ratings = []

    for path, result in alignment_scores.items():
        ratings.append(result["rating"])

    scores = np.array(ratings)

    # Calculate alignment rate (% of scores >= threshold)
    aligned = scores >= threshold
    alignment_rate = (aligned.sum() / len(scores)) * 100

    # Convert scores to polars DataFrame for distribution analysis
    score_df = pl.DataFrame({"score": scores})

    # Calculate score distribution
    distribution = score_df.group_by("score").len().sort("score")

    # Calculate statistics
    stats = {
        "alignment_rate": alignment_rate,
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "score_distribution": dict(
            zip(distribution["score"].to_list(), distribution["len"].to_list())
        ),
        "n_samples": len(scores),
    }

    return stats


def plot_confusion_matrices(y_true, y_pred, labels):
    """
    Plot three types of confusion matrices:
    1. Raw counts
    2. Normalized by row (recall per class)
    3. Normalized by column (precision per class)

    Parameters:
    y_true (array-like): Ground truth labels
    y_pred (array-like): Predicted labels
    labels (list): List of class labels

    Returns:
    Figure: Matplotlib figure containing the plots
    """
    # Calculate confusion matrix (raw counts)
    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)

    # Normalize by row (true labels) - shows recall
    cm_norm_recall = (cm_raw / cm_raw.sum(axis=1)[:, np.newaxis]).round(3)

    # Create figure with three subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))

    # Plot normalized by row (recall)
    sns.heatmap(
        cm_norm_recall,
        annot=True,
        fmt=".1%",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax1,
    )
    # ax1.set_title("Confusion Matrix\n(Normalized by Row - Recall)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout()
    return fig


def evaluate_adl_classifications(results_dict, adjusted=False):
    """
    Evaluate ADL classification results with proper sample weighting for balanced metrics.

    Parameters:
    results_dict (dict): Dictionary containing classification results
    adjusted (bool): Whether to adjust balanced accuracy for chance

    Returns:
    dict: Evaluation results including metrics and visualizations
    """
    # Extract true and predicted labels
    data = []
    for file_path, result in results_dict.items():
        true_label = get_true_label(file_path)
        pred_label = result["prediction"]
        data.append({"true_label": true_label, "pred_label": pred_label})

    # Create Polars DataFrame
    df = pl.DataFrame(data)

    # Convert to numpy arrays for sklearn metrics
    y_true = np.array(df["true_label"].to_list())
    y_pred = np.array(df["pred_label"].to_list())

    # Calculate sample weights
    sample_weights = calculate_sample_weights(y_true)

    # Get unique labels
    labels = sorted(list(set(y_true) | set(y_pred)))

    # Calculate class distribution
    class_dist = (
        df.group_by("true_label")
        .agg(pl.count().alias("Count"))
        .sort("Count", descending=True)
    )

    # Generate classification report with sample weights
    report = classification_report(
        y_true, y_pred, sample_weight=sample_weights, output_dict=True, zero_division=0
    )

    # Convert to Polars DataFrame
    metrics_df = pl.DataFrame(
        {
            k: (
                v
                if isinstance(v, dict)
                else {"precision": v, "recall": v, "f1-score": v, "support": v}
            )
            for k, v in report.items()
        }
    ).transpose()

    # Calculate balanced accuracy with sample weights
    bal_acc = balanced_accuracy_score(
        y_true, y_pred, sample_weight=sample_weights, adjusted=adjusted
    )

    # Create confusion matrix visualizations
    confusion_matrices = plot_confusion_matrices(y_true, y_pred, labels)

    # Calculate additional weighted metrics
    additional_metrics = {
        "Balanced Accuracy": bal_acc,
        "Macro Avg F1": report["macro avg"]["f1-score"],
        "Weighted Avg F1": report["weighted avg"]["f1-score"],
    }

    # Add class weights to the results
    class_weights = dict(
        zip(
            labels,
            compute_class_weight(
                class_weight="balanced", classes=np.array(labels), y=y_true
            ),
        )
    )

    return {
        "class_distribution": class_dist,
        "metrics": metrics_df,
        "additional_metrics": additional_metrics,
        "confusion_matrices": confusion_matrices,
        "class_weights": class_weights,
        "sample_weights": sample_weights,
    }


def calculate_sample_weights(y_true):
    """
    Calculate sample weights based on class distribution.

    Parameters:
    y_true (array-like): Ground truth labels

    Returns:
    array: Sample weights for each instance
    """
    # Get unique classes and compute class weights
    classes = np.unique(y_true)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_true
    )

    # Create a dictionary mapping classes to weights
    weight_dict = dict(zip(classes, class_weights))

    # Map weights to samples
    sample_weights = np.array([weight_dict[y] for y in y_true])

    return sample_weights


def analyze_tags(results_dict, use_ground_truth=True):
    """
    Analyze tags associated with either ground truth or predicted labels.

    Parameters:
    results_dict (dict): Dictionary containing classification results
    use_ground_truth (bool): If True, use ground truth labels; if False, use predictions

    Returns:
    dict: Contains tag frequencies DataFrame and dictionary of word cloud figures
    """
    # Create a dictionary to store tags for each label
    tags_by_label = defaultdict(list)

    # Collect tags based on chosen label type
    for file_path, result in results_dict.items():
        if use_ground_truth:
            label = get_true_label(file_path)
        else:
            label = result["prediction"]

        tags = [tag.lower() for tag in result["tags"]]
        tags_by_label[label].extend(tags)

    # Convert to DataFrame for frequency analysis
    tag_data = []
    for label, tags in tags_by_label.items():
        for tag in tags:
            tag_data.append({"label": label, "tag": tag})

    # Create Polars DataFrame and calculate frequencies
    df = pl.DataFrame(tag_data)
    tag_frequencies = (
        df.group_by(["label", "tag"])
        .agg(pl.count().alias("frequency"))
        .sort(["label", "frequency"], descending=[False, True])
    )

    # Create individual word clouds
    wordcloud_figures = {}

    for label in sorted(tags_by_label.keys()):
        # Get tags and their frequencies for this label
        label_tags = tag_frequencies.filter(pl.col("label") == label)

        # Create frequency dictionary
        freq_dict = dict(
            zip(label_tags["tag"].to_list(), label_tags["frequency"].to_list())
        )

        # Generate word cloud
        wordcloud = WordCloud(
            width=1600,
            height=800,
            background_color="white",
            colormap="viridis",
            max_words=50,
            prefer_horizontal=0.7,
        ).generate_from_frequencies(freq_dict)

        # Create figure for this label
        fig = plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        label_type = "Ground Truth" if use_ground_truth else "Predicted"
        plt.title(f"Tags for {label} ({label_type})", pad=20, size=16)

        # Store the figure
        wordcloud_figures[label] = fig

    return {"tag_frequencies": tag_frequencies, "wordcloud_plots": wordcloud_figures}


def get_top_tags_by_prediction(results_dict, top_n=10):
    """
    Get the top N most frequent tags for each prediction.

    Parameters:
    results_dict (dict): Dictionary containing classification results
    top_n (int): Number of top tags to return for each prediction

    Returns:
    pl.DataFrame: DataFrame with top tags and their frequencies by prediction
    """
    # Create a dictionary to store tags for each prediction
    tags_by_prediction = defaultdict(list)

    # Collect tags for each prediction
    for result in results_dict.values():
        prediction = result["prediction"]
        tags = result["tags"]
        tags_by_prediction[prediction].extend(tags)

    # Convert to DataFrame and calculate frequencies
    tag_data = []
    for prediction, tags in tags_by_prediction.items():
        for tag in tags:
            tag_data.append({"prediction": prediction, "tag": tag})

    # Create Polars DataFrame and calculate top tags
    df = pl.DataFrame(tag_data)
    top_tags = (
        df.group_by(["prediction", "tag"])
        .agg(pl.count().alias("frequency"))
        .sort(["prediction", "frequency"], descending=[False, True])
        .group_by("prediction")
        .head(top_n)
    )

    return top_tags


def display_prediction_analysis(
    batch_path,
    results_dict,
    key=None,
    correct_only=False,
    incorrect_only=False,
    max_display=None,
):
    """
    Display prediction analysis including grid images, labels, tags, and reasoning.
    Can display either a single prediction or multiple predictions.

    Parameters:
    results_dict (dict): Dictionary containing classification results
    key (str): Specific key to display. If None, displays multiple predictions
    correct_only (bool): If True, show only correct predictions
    incorrect_only (bool): If True, show only incorrect predictions
    max_display (int): Maximum number of predictions to display, None for all

    Returns:
    None (displays figures)
    """

    # Handle single key display
    if key is not None:
        if key not in results_dict:
            print(f"Key '{key}' not found in results dictionary.")
            return

        # Create single-item list with the specified key
        display_items = [
            {
                "file_path": key,
                "grid_path": os.path.join(
                    batch_path,
                    results_dict[key]["grid_path"],
                ),
                "true_label": get_true_label(key),
                "pred_label": results_dict[key]["prediction"],
                "tags": results_dict[key]["tags"],
                "reasoning": results_dict[key]["reasoning"],
                "intermediates": results_dict[key].get("intermediates", {}),
            }
        ]
    else:
        # Process all results and filter based on correctness if requested
        display_items = []
        for file_path, result in results_dict.items():
            true_label = get_true_label(file_path)
            pred_label = result["prediction"]
            is_correct = true_label == pred_label

            if correct_only and not is_correct:
                continue
            if incorrect_only and is_correct:
                continue

            display_items.append(
                {
                    "file_path": file_path,
                    "grid_path": os.path.join(batch_path, result["grid_path"]),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "is_correct": is_correct,
                    "tags": result["tags"],
                    "reasoning": result["reasoning"],
                    "intermediates": result.get("intermediates", {}),
                }
            )

        # Sort by correctness (incorrect first if showing both)
        display_items.sort(key=lambda x: x["is_correct"])

        # Apply max_display limit if specified
        if max_display is not None:
            display_items = display_items[:max_display]

    # Display each prediction
    for idx, item in enumerate(display_items, 1):
        # Create figure with custom layout
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1])

        # Left side: Image
        ax_img = fig.add_subplot(gs[0, 0])
        try:
            img = mpimg.imread(item["grid_path"])
            ax_img.imshow(img)
        except Exception as e:
            ax_img.text(
                0.5, 0.5, f"Error loading image: {str(e)}", ha="center", va="center"
            )
        ax_img.axis("off")

        # Right side: Text content
        ax_text = fig.add_subplot(gs[0, 1])
        ax_text.axis("off")

        # Prepare text content
        is_correct = item["true_label"] == item["pred_label"]
        correct_str = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        color = "green" if is_correct else "red"

        title = (
            f"File: {item['file_path']}\n"
            f"True: {item['true_label']} → Predicted: {item['pred_label']}\n"
            f"{correct_str}"
        )
        ax_img.set_title(title, fontsize=14, color=color)

        # Combine all text content
        text_parts = [
            "Tags:",
            "• " + "\n• ".join(item["tags"]) + "\n",
            "Reasoning:",
            item["reasoning"] + "\n",
        ]

        if item["intermediates"]:
            text_parts.append("Intermediate Steps:")
            for key, value in item["intermediates"].items():
                text_parts.extend([f"\n{key}:", value])

        # Wrap text and join with newlines
        wrapped_text = []
        for line in text_parts:
            if line.endswith(":"):
                wrapped_text.append(line)
            else:
                wrapped_text.extend(wrap(line, width=60))

        text_content = "\n".join(wrapped_text)

        # Add text to plot
        ax_text.text(
            0.02,
            0.98,
            text_content,
            va="top",
            ha="left",
            fontsize=11,
            linespacing=1.2,
            transform=ax_text.transAxes,
        )

        plt.tight_layout(pad=1.0)
        plt.show()


def analyze_predictions(results_dict):
    """
    Generate a summary of prediction patterns.

    Parameters:
    results_dict (dict): Dictionary containing classification results

    Returns:
    dict: Summary statistics and patterns
    """
    summary = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "misclassified_as": defaultdict(int),
        }
    )

    # Analyze each prediction
    for file_path, result in results_dict.items():
        true_label = get_true_label(file_path)
        pred_label = result["prediction"]

        summary[true_label]["total"] += 1
        if true_label == pred_label:
            summary[true_label]["correct"] += 1
        else:
            summary[true_label]["incorrect"] += 1
            summary[true_label]["misclassified_as"][pred_label] += 1

    # Convert to regular dict and calculate percentages
    formatted_summary = {}
    for label, stats in summary.items():
        formatted_summary[label] = {
            "total_samples": stats["total"],
            "correct": stats["correct"],
            "accuracy": round(stats["correct"] / stats["total"] * 100, 2),
            "misclassifications": dict(
                sorted(
                    stats["misclassified_as"].items(), key=lambda x: x[1], reverse=True
                )
            ),
        }

    return formatted_summary
