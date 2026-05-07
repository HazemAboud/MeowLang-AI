import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, time, pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

# 0. Environment Setup
tf.keras.backend.clear_session()
try:
    # Force float32 policy to ensure the model and TFLite conversion stay 32-bit
    tf.keras.mixed_precision.set_global_policy('float32')
except Exception:
    pass
tf.keras.utils.set_random_seed(42)

# ── Configuration ──────────────────────────────────────────
TRAIN_DIR   = r'E:\new_data\img'
TEST_DIR    = r'E:\new_data\test_img'
IMG_SIZE    = 512
BATCH_SIZE  = 24
EPOCHS      = 50
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))

# ── Helpers ───────────────────────────────────────────────
def preprocess(x):
    """Scaling pixels to [-1, 1] to match existing model preprocessing."""
    return (x / 127.5) - 1.0

def make_datagen(augment=True):
    """Returns a data generator matching original training hyperparameters."""
    if augment:
        return tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=preprocess,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )
    return tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess)

def make_model(num_classes):
    """Architecture matching the Sequential structure used in the context."""
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

def build_df(root, classes):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    rows = []
    for label in classes:
        target_path = os.path.join(root, label)
        if not os.path.exists(target_path): continue
        for dirpath, _, files in os.walk(target_path):
            rows += [{'filename': os.path.join(dirpath, f), 'class': label}
                     for f in files if f.lower().endswith(exts)]
    return pd.DataFrame(rows)

def main():
    # GPU Memory Growth
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    class_names = sorted(d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d)))
    num_classes = len(class_names)

    print("--- Preparing Data ---")
    # Ensure class_names are consistent across train and test
    train_df = build_df(TRAIN_DIR, class_names)
    test_df  = build_df(TEST_DIR,  class_names)
    print(f"Training on {len(train_df)} images across {num_classes} classes.")

    # Generators
    train_gen = make_datagen(augment=True).flow_from_dataframe(
        train_df, x_col='filename', y_col='class', target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True, classes=class_names
    )
    test_gen = make_datagen(augment=False).flow_from_dataframe(
        test_df, x_col='filename', y_col='class', target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False, classes=class_names
    )

    # Class Weighting
    weights = compute_class_weight('balanced', classes=np.unique(train_df['class']), y=train_df['class'])
    cw_dict = dict(enumerate(weights))

    # 1. Training
    model = make_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Define the path for saving the best model weights
    h5_path = os.path.join(SCRIPT_DIR, 'final_model_full.h5')

    # ModelCheckpoint callback to save weights at the epoch with the least loss
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=h5_path,
        monitor='loss',          # Monitor training loss
        save_best_only=True,     # Save only the best model
        save_weights_only=True,  # Save only the weights
        mode='min',              # 'min' because we want the least loss
        verbose=1
    )

    # Add a callback to dynamically reduce the learning rate if the loss stops improving
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
    )
    
    print(f"\n--- Training for {EPOCHS} Epochs ---")
    start_time = time.time()
    # Training on the FULL dataset without a validation set
    history = model.fit(
        train_gen, epochs=EPOCHS,
        class_weight=cw_dict, verbose=1, # Pass class_weight directly to model.fit
        callbacks=[checkpoint_callback, reduce_lr]
    )
    
    # Load the best weights saved by the callback
    model.load_weights(h5_path)

    # 2. Testing and Visualization
    print("\n--- Evaluating on Test Set ---")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.title(f'Training Accuracy (Final Test: {test_acc*100:.1f}%)'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.title('Training Loss'); plt.legend()
    plt.savefig(os.path.join(SCRIPT_DIR, 'training_curves_final.png'))

    # Confusion Matrix
    y_pred = np.argmax(model.predict(test_gen), axis=1)
    cm = confusion_matrix(test_gen.classes, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap='Blues', ax=ax)
    plt.title('Final Model Confusion Matrix')
    plt.savefig(os.path.join(SCRIPT_DIR, 'confusion_matrix_final.png'))

    # Calculate per-class accuracy (recall)
    # precision_recall_fscore_support returns (precision, recall, fscore, support)
    # We are interested in recall for per-class accuracy.
    _, per_class_recall, _, _ = precision_recall_fscore_support(
        test_gen.classes, y_pred, labels=range(num_classes), average=None, zero_division=0
    )

    # Plotting per-class accuracy
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, per_class_recall, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (Recall)')
    plt.title('Per-Class Accuracy (Recall) on Test Set')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'per_class_accuracy_final.png'))

    # 3. TFLite Conversion
    print("\n--- Converting to TFLite (Pure Float32) ---")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Force standard TFLite ops (no Flex/flexible ops)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    # To keep it pure 32-bit float, we omit optimization flags that trigger quantization
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(SCRIPT_DIR, 'final_model_float32.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✓ DONE. TFLite saved to: {tflite_path}")
    print(f"Total time: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()