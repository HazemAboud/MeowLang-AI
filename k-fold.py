import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, time, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

tf.keras.utils.set_random_seed(42)

# ── Config ────────────────────────────────────────────────
TRAIN_DIR   = r'E:\new_data\img'
TEST_DIR    = r'E:\new_data\test_img'
IMG_SIZE    = 512
BATCH_SIZE  = 24 
EPOCHS      = 25
K_FOLDS     = 5
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))

# ── Helpers ───────────────────────────────────────────────
def preprocess(x):
    return (x / 127.5) - 1.0  # Scale to [-1, 1]

def make_datagen():
    return tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

def make_model(num_classes):
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

def flow(datagen, df, shuffle=True, x_col='filename', y_col='class', classes=None):
    return datagen.flow_from_dataframe(
        df, x_col=x_col, y_col=y_col,
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=shuffle, classes=classes
    )

def macro_accuracy(model, generator, num_classes):
    y_pred = np.argmax(model.predict(generator), axis=1)
    cm = confusion_matrix(generator.classes, y_pred)
    per_class = [cm[i, i] / cm[i].sum() if cm[i].sum() else 0 for i in range(num_classes)]
    return np.mean(per_class), cm

def save_plot(title, xlabel, ylabel, path, **series):
    plt.figure(figsize=(10, 6))
    for label, values in series.items():
        plt.plot(range(1, len(values) + 1), values, label=label)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(path); plt.close()

def class_weights(df, col='class'):
    classes = np.unique(df[col])
    weights = compute_class_weight('balanced', classes=classes, y=df[col])
    return dict(enumerate(weights))

def build_df(root, classes, x='filename', y='class'):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    rows = []
    for label in classes:
        for dirpath, _, files in os.walk(os.path.join(root, label)):
            rows += [{x: os.path.join(dirpath, f), y: label}
                     for f in files if f.lower().endswith(exts)]
    return pd.DataFrame(rows)

# ── Main ──────────────────────────────────────────────────
def main():
    # GPU setup
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    
    class_names = sorted(d for d in os.listdir(TRAIN_DIR)
                         if os.path.isdir(os.path.join(TRAIN_DIR, d)))
    num_classes = len(class_names)

    train_df = build_df(TRAIN_DIR, class_names)
    test_df  = build_df(TEST_DIR,  class_names, x='f', y='c')
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    datagen   = make_datagen()
    test_gen  = flow(datagen, test_df, shuffle=False, x_col='f', y_col='c', classes=class_names)
    skf       = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_val_accs, fold_test_accs, all_histories = [], [], []
    start = time.time()

    # ── K-Fold training ───────────────────────────────────
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df['filename'], train_df['class'])):
        print(f"\n── Fold {fold+1} ──")
        tf.keras.backend.clear_session()

        tr_gen = flow(datagen, train_df.iloc[tr_idx])
        va_gen = flow(datagen, train_df.iloc[va_idx], shuffle=False)

        model = make_model(num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        ckpt_path = os.path.join(SCRIPT_DIR, f'fold_{fold+1}.weights.h5')
        history = model.fit(
            tr_gen, epochs=EPOCHS, validation_data=va_gen,
            class_weight=class_weights(train_df.iloc[tr_idx]),
            callbacks=[
                tf.keras.callbacks.EarlyStopping('val_loss', patience=12, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=4, min_lr=1e-6),
                tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy',
                                                   save_best_only=True, save_weights_only=True)
            ], verbose=1
        )

        all_histories.append(history.history)
        best_val = max(history.history['val_accuracy'])
        fold_val_accs.append(best_val)

        # Load best weights for this fold → evaluate on test set
        model.load_weights(ckpt_path)
        test_acc, _ = macro_accuracy(model, test_gen, num_classes)
        fold_test_accs.append(test_acc)
        print(f"Fold {fold+1} | Val: {best_val:.4f} | Test (macro): {test_acc:.4f}")

        # Individual Fold Plot
        save_plot(f'Fold {fold+1} Accuracy', 'Epoch', 'Accuracy',
                  os.path.join(SCRIPT_DIR, f'fold_{fold+1}_accuracy.png'),
                  Train=history.history['accuracy'],
                  Val=history.history['val_accuracy'])

    # ── Final Summary Visualizations ──────────────────────
    print("\n── Generating Final Summary Graphs ──")
    
    plt.figure(figsize=(14, 6))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    for i, h in enumerate(all_histories):
        plt.plot(h['loss'], label=f'Fold {i+1}')
    plt.title('Training Loss (All Folds)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    for i, h in enumerate(all_histories):
        plt.plot(h['val_accuracy'], label=f'Fold {i+1} (Test: {fold_test_accs[i]:.2f})')
    plt.title('Validation Accuracy (All Folds)'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, 'kfold_summary_results.png'))
    plt.show()

    print(f"\nDone in {(time.time()-start)/60:.1f} min")
    print(f"Average Macro Test Accuracy across {K_FOLDS} folds: {np.mean(fold_test_accs)*100:.2f}%")

if __name__ == "__main__":
    main()