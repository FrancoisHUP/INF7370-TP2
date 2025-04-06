from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model
import random

# ---------------------------
# Configuration
# ---------------------------
BATCH_SIZE = 1
MODEL_PATH = "models/default_Model.keras"  # path to the saved model
TEST_PATH = "donnees/test"         # path to the test dataset


# ---------------------------
# Data Loading
# ---------------------------
def get_test_generator(target_size, color_mode):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=target_size,
        batch_size=BATCH_SIZE,
        class_mode="categorical",  # multi-class classification
        shuffle=False,
        color_mode=color_mode
    )
    return test_generator
# ---------------------------
# Model Evaluation
# ---------------------------
def evaluate_model(model, test_generator):
    eval_results = model.evaluate(test_generator, verbose=1)
    print("> Test Loss:", eval_results[0])
    print("> Test Accuracy:", eval_results[1])
    return eval_results

# ---------------------------
# Predictions and Confusion Matrix
# ---------------------------
def get_predictions(model, test_generator):
    predictions = model.predict(test_generator, verbose=1)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = test_generator.classes
    num_classes = len(test_generator.class_indices)
    class_names = list(test_generator.class_indices.keys())
    return true_labels, predicted_labels, num_classes, class_names

def compute_confusion_matrix(true_labels, predicted_labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels, predicted_labels):
        cm[t, p] += 1
    return cm

def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Confusion matrix plot saved to {output_path}")
    plt.close()

def plot_misclassified_grid(true_labels, predicted_labels, test_generator, class_names, output_path):
    """
    Creates a grid (num_classes x num_classes) where each cell (i, j)
    displays one random misclassified image for which the true label is i and predicted label is j.
    For diagonal cells (i == j), the text "Correct" is shown.
    If no misclassified sample is available for a cell, the text "None" is shown.
    Global axis labels "True label" (y-axis) and "Predicted label" (x-axis) are added.
    """
    num_classes = len(class_names)
    # Build a mapping from (true, predicted) to list of indices (for misclassified samples)
    misclassified = {}
    for idx, (t, p) in enumerate(zip(true_labels, predicted_labels)):
        if t != p:
            key = (t, p)
            misclassified.setdefault(key, []).append(idx)
    
    # Create the grid of subplots.
    fig, axes = plt.subplots(num_classes, num_classes, figsize=(num_classes * 2, num_classes * 2))
    
    # Loop over each cell in the grid.
    for i in range(num_classes):
        for j in range(num_classes):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            # For the leftmost column, add true label as y-axis label.
            if j == 0:
                ax.set_ylabel(class_names[i], fontsize=10)
            # For the bottom row, add predicted label as x-axis label.
            if i == num_classes - 1:
                ax.set_xlabel(class_names[j], fontsize=10)
            # Diagonal cell: show "Correct"
            if i == j:
                ax.text(0.5, 0.5, "Correct", fontsize=12, ha='center', va='center', transform=ax.transAxes)
                ax.set_facecolor("lightgreen")
            else:
                key = (i, j)
                if key in misclassified:
                    sample_idx = random.choice(misclassified[key])
                    filename = test_generator.filenames[sample_idx]
                    img_path = os.path.join(test_generator.directory, filename)
                    img = plt.imread(img_path)
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "None", fontsize=12, ha='center', va='center', transform=ax.transAxes)
                    ax.set_facecolor("lightgray")
    
    # Add global axis labels.
    fig.text(0.5, 0.04, "Predicted label", ha="center", fontsize=12)
    fig.text(0.04, 0.5, "True label", va="center", rotation="vertical", fontsize=12)
    
    plt.tight_layout(rect=[0.1, 0.1, 1, 1])
    plt.savefig(output_path)
    print(f"Misclassified grid plot saved to {output_path}")
    plt.close()

# ---------------------------
# Per-Class Metrics and Plotting
# ---------------------------
def compute_class_metrics(cm, class_names):
    metrics = {}
    for i, cls in enumerate(class_names):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(cm[i, :])
        metrics[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }
    return metrics

def plot_class_metrics(metrics, output_path):
    labels = list(metrics.keys())
    precision_vals = [metrics[label]["precision"] for label in labels]
    recall_vals = [metrics[label]["recall"] for label in labels]
    f1_vals = [metrics[label]["f1"] for label in labels]
    
    x = np.arange(len(labels))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision_vals, width, label='Precision')
    plt.bar(x, recall_vals, width, label='Recall')
    plt.bar(x + width, f1_vals, width, label='F1-score')
    plt.ylabel('Score')
    plt.xlabel('Class')
    plt.title('Per-Class Metrics')
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Per-class metrics plot saved to {output_path}")
    plt.close()

# ---------------------------
# Main Evaluation Function
# ---------------------------
def eval(model_path="output/10_deep_wide/best_model.keras"):
    # Load the trained model
    model = load_model(model_path)
    model.summary()

    # Extract model's expected input size and color mode based on its input shape.
    input_size = model.input_shape[1:3]
    channels = model.input_shape[-1]
    color_mode = "rgb" if channels == 3 else "grayscale"
    
    print(f"Model input size: {input_size}")
    print(f"Model expects {channels} channel(s) -> using color_mode: {color_mode}")
    
    # Get test generator with the appropriate target size and color mode
    test_generator = get_test_generator(target_size=input_size, color_mode=color_mode)
    output_dir = os.path.dirname(model_path)
    
    # Evaluate the model
    evaluate_model(model, test_generator)
    
    # Get predictions and true labels
    true_labels, predicted_labels, num_classes, class_names = get_predictions(model, test_generator)
    
    # Compute confusion matrix and plot it
    cm = compute_confusion_matrix(true_labels, predicted_labels, num_classes)
    print("Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, class_names, os.path.join(output_dir, "confusion_matrix.png"))
    
    # Compute per-class metrics and plot them
    metrics = compute_class_metrics(cm, class_names)
    for cls, vals in metrics.items():
        print(f"Class {cls}: Precision: {vals['precision']:.4f}, Recall: {vals['recall']:.4f}, F1: {vals['f1']:.4f}, Support: {vals['support']}")
    plot_class_metrics(metrics, os.path.join(output_dir, "class_metrics.png"))
    # Plot the misclassified images grid.
    misclassified_output = os.path.join(output_dir, "misclassified_grid.png")
    plot_misclassified_grid(true_labels, predicted_labels, test_generator, class_names, misclassified_output)


if __name__ == "__main__":
    eval()
