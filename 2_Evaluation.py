from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os
from keras.models import load_model

# ---------------------------
# Configuration
# ---------------------------
IMAGE_SCALE = 224
IMAGES_COLOR_MODE = "grayscale"  # or 'rgb' if used during training
BATCH_SIZE = 8

MODEL_PATH = "models/Model.keras"  # path to the saved model
TEST_PATH = "donnees/test"         # path to the test dataset
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Data Loading
# ---------------------------
def get_test_generator():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(IMAGE_SCALE, IMAGE_SCALE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",  # multi-class classification
        shuffle=False,
        color_mode=IMAGES_COLOR_MODE
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
def main():
    # Load test generator
    test_generator = get_test_generator()

    # Load the trained model
    model = load_model(MODEL_PATH)
    
    # Evaluate the model
    evaluate_model(model, test_generator)
    
    # Get predictions and true labels
    true_labels, predicted_labels, num_classes, class_names = get_predictions(model, test_generator)
    
    # Compute confusion matrix and plot it
    cm = compute_confusion_matrix(true_labels, predicted_labels, num_classes)
    print("Confusion Matrix:")
    print(cm)
    plot_confusion_matrix(cm, class_names, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    
    # Compute per-class metrics and plot them
    metrics = compute_class_metrics(cm, class_names)
    for cls, vals in metrics.items():
        print(f"Class {cls}: Precision: {vals['precision']:.4f}, Recall: {vals['recall']:.4f}, F1: {vals['f1']:.4f}, Support: {vals['support']}")
    plot_class_metrics(metrics, os.path.join(OUTPUT_DIR, "class_metrics.png"))

if __name__ == "__main__":
    main()
