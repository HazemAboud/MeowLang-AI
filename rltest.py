import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# --- Configuration ---
MODEL_PATH = r'E:\new_data\final_model_full.h5'
TEST_DIR = r'E:\new_data\test_img'
IMG_SIZE = 512
BATCH_SIZE = 24
EXPECTED_CLASSES = ['Angry', 'Food', 'Isolation', 'MotherCall', 'Resting']

def make_model(num_classes):
    """Reconstructs the model architecture to load weights into."""
    def block(filters, name):
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, 3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ], name=name)

    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        block(32,  "B1"), tf.keras.layers.MaxPooling2D(2),
        block(64,  "B2"), tf.keras.layers.MaxPooling2D(2),
        block(128, "B3"), tf.keras.layers.MaxPooling2D(2),
        block(256, "B4"), tf.keras.layers.MaxPooling2D(2),
        block(512, "B5"), tf.keras.layers.MaxPooling2D(2),
        block(512, "B6"),  
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes)
    ])

def preprocess_input(x):
    """Normalizes pixel values to [-1, 1] range."""
    return (x / 127.5) - 1.0

def main():
    # 1. Environment Setup
    tf.keras.backend.clear_session()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_DIR):
        print("Error: Model path or Test directory not found.")
        return

    # 2. Load Model
    num_classes = len(EXPECTED_CLASSES)
    model = make_model(num_classes)
    model.load_weights(MODEL_PATH)
    print("Model loaded successfully.")

    # 3. Data Loading
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        classes=EXPECTED_CLASSES
    )

    # 4. Predictions
    print("Running inference...")
    y_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = test_gen.classes

    overall_acc = np.sum(y_true == y_pred) / len(y_true)

    per_class_prec, per_class_rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(num_classes), zero_division=0
    )

    print(f"\n{'='*30}")
    print(f"OVERALL ACCURACY: {overall_acc:.4f}")
    print(f"{'='*30}")
    for i, name in enumerate(EXPECTED_CLASSES):
        print(f"{name:12} | Precision: {per_class_prec[i]:.4f} | Recall: {per_class_rec[i]:.4f}")

    x = np.arange(len(EXPECTED_CLASSES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))

    rects1 = ax.bar(x - width/2, per_class_prec, width, label='Precision', color='#3498db')
    rects2 = ax.bar(x + width/2, per_class_rec, width, label='Recall', color='#e74c3c')

    # Benchmark line for Overall Accuracy
    ax.axhline(y=overall_acc, color='black', linestyle='--', alpha=0.5, 
               label=f'Global Accuracy: {overall_acc:.4f}')

    ax.set_ylabel('Scores')
    ax.set_title('Classification Performance by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(EXPECTED_CLASSES)
    ax.set_ylim(0, 1.1)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.3f')
    ax.bar_label(rects2, padding=3, fmt='%.3f')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()