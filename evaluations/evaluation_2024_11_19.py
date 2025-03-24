# %%
import sys

sys.path.append("..")

# %%
import json
import os

import polars as pl

import evaluation as ev

pl.config.Config.set_tbl_rows(100)

# %%
# Set the batch path for evaluation.
batch_path = os.path.abspath(
    "/media/adeshkadambi/WD_BLACK/PhD/adl_recognition/results/batch_20241119_121641/"
)

# %%
# Make sure that all predictions are correct values.
results_path = os.path.join(batch_path, "results.json")

with open(results_path, "r") as f:
    results = json.load(f)

data = []
for file_path, result in results.items():
    true_label = ev.get_true_label(file_path)
    pred_label = result["prediction"]
    data.append({"true_label": true_label, "pred_label": pred_label})

df = pl.DataFrame(data)
df["pred_label"].value_counts()

# %%
# Evaluate the batch.
evaluation = ev.evaluate_adl_classifications(results_dict=results)

# %%
# Get balanced accuracy, macro f1, and weighted f1.
balanced_accuracy = evaluation["additional_metrics"]['Balanced Accuracy']
macro_f1 = evaluation["additional_metrics"]['Macro Avg F1']
weighted_f1 = evaluation["additional_metrics"]['Weighted Avg F1']

# %%
# Generate word clouds for ground truth.
analysis_results_gt = ev.analyze_tags(results)

# %%
# Generate word clouds for predictions.
analysis_results_pred = ev.analyze_tags(results, use_ground_truth=False)

# %%
# Generate and print summary.
summary = ev.analyze_predictions(results)

print("\nPrediction Analysis Summary:")
for label, stats in summary.items():
    print(f"\n{label}:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Accuracy: {stats['accuracy']}%")
    if stats["misclassifications"]:
        print("Misclassified as:")
        for wrong_label, count in stats["misclassifications"].items():
            print(f"  {wrong_label}: {count} times")

# %%
# Print the alignment rate.
results = ev.calculate_alignment_rate(os.path.join(batch_path, "alignment.json"))

print(f"Alignment Rate: {results['alignment_rate']:.1f}%")
print(f"Mean Score: {results['mean_score']:.2f}")
print(f"Median Score: {results['median_score']:.1f}")
print("\nScore Distribution:")
for score, count in results["score_distribution"].items():
    print(f"Score {score}: {count} samples")
print(f"\nTotal Samples: {results['n_samples']}")
