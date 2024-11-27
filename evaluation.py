import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn import metrics
from sklearn.utils import compute_class_weight
from wordcloud import WordCloud


def get_true_label(file_path: str) -> str:
    """Extract the true label from a file path."""
    return (
        file_path.split("/")[0]
        .upper()
        .replace("-", " ")
        .replace("GROOMING HEALTH MANAGEMENT", "GROOMING AND HEALTH MANAGEMENT")
        .replace("MEAL PREPARATION CLEANUP", "MEAL PREPARATION AND CLEANUP")
        .replace("LEISURE OTHER ACTIVITIES", "LEISURE")
        .replace("SELF FEEDING", "FEEDING")
    )


def calculate_alignment_rate(alignment_path: str, threshold: int = 4) -> dict:
    """Calculate clinical reasoning alignment rate from 5-point scale scores."""

    with open(alignment_path, encoding="utf-8") as f:
        alignment_scores = json.load(f)

    ratings = []

    for path, result in alignment_scores.items():
        assert "rating" in result, f"Rating field not found in alignment for {path}"
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


def evaluate_adl_classifications(results_dict: dict) -> dict:
    # Extract true and predicted labels
    data = []
    for file_path, result in results_dict.items():
        true_label = get_true_label(file_path)
        pred_label = result["prediction"]
        data.append({"true_label": true_label, "pred_label": pred_label})

    df = pl.DataFrame(data)
    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()

    # Calculate sample weights
    sample_weights = calculate_sample_weights(y_true)

    # Get unique labels
    labels = sorted(list(set(y_true) | set(y_pred)))

    # Calculate class distribution
    class_dist = (
        df.group_by("true_label").agg(pl.count().alias("Count")).sort("Count", descending=True)
    )

    # Generate classification report with sample weights
    report = metrics.classification_report(
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
    bal_acc = metrics.balanced_accuracy_score(y_true, y_pred, sample_weight=sample_weights)

    # Create confusion matrix visualizations
    confusion_matrices = metrics.ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        normalize="true",
        xticks_rotation=90,
        cmap=plt.cm.Blues,
    )

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
            compute_class_weight(class_weight="balanced", classes=np.array(labels), y=y_true),
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


def calculate_sample_weights(y_true: np.ndarray) -> np.ndarray:
    """Calculate sample weights based on class distribution."""

    classes = np.unique(y_true)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_true)

    # Create a dictionary mapping classes to weights
    weight_dict = dict(zip(classes, class_weights))

    # Map weights to samples
    return np.array([weight_dict[y] for y in y_true])


def analyze_tags(
    results_dict: dict[str, dict[str, str | list[str] | list[dict[str, str]]]],
    use_ground_truth: bool = True,
) -> dict[str, pl.DataFrame | dict[str, plt.Figure]]:
    """Analyzes tag frequencies and generates word clouds for each label."""

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
        freq_dict = dict(zip(label_tags["tag"].to_list(), label_tags["frequency"].to_list()))

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


def get_top_tags_by_prediction(
    results_dict: dict[str, dict[str, str | list[str] | list[dict[str, str]]]],
    top_n: int = 10,
) -> pl.DataFrame:
    """Returns the most frequent tags for each prediction category."""

    # Create a dictionary to store tags for each prediction
    tags_by_prediction = defaultdict(list)
    for result in results_dict.values():
        prediction = result["prediction"]
        tags = result["tags"]
        tags_by_prediction[prediction].extend(tags)

    # Convert to DataFrame and calculate frequencies
    tag_data = []
    for prediction, tags in tags_by_prediction.items():
        for tag in tags:
            tag_data.append({"prediction": prediction, "tag": tag})

    df = pl.DataFrame(tag_data)
    top_tags = (
        df.group_by(["prediction", "tag"])
        .agg(pl.count().alias("frequency"))
        .sort(["prediction", "frequency"], descending=[False, True])
        .group_by("prediction")
        .head(top_n)
    )

    return top_tags


def analyze_predictions(
    results_dict: dict[str, dict[str, str | list[str] | list[dict[str, str]]]],
) -> dict[str, dict[str, int | float | dict[str, int]]]:
    """Generates summary statistics and error patterns for predictions."""

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
                sorted(stats["misclassified_as"].items(), key=lambda x: x[1], reverse=True)
            ),
        }

    return formatted_summary
