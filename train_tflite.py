import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

tf.keras.utils.set_random_seed(42)

def create_model(num_classes, input_size=(512, 512, 3)):
    """Create CNN model matching PyTorch architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_size),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalAveragePooling2D(),
        
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes)
    ])
    return model

def main():
    print("\n" + "="*60)
    print("GPU CONFIGURATION")
    print("="*60)
    
    print(f"TensorFlow Version: {tf.__version__}")

    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✓ GPUs detected: {len(gpus)}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  - {gpu.name}: Memory growth enabled")

            print(f"\n✓ GPU is active and ready for training")
            print(f"✓ Using device: GPU")
            print(f"✓ GPU devices: {[gpu.name for gpu in gpus]}")
            
        except RuntimeError as e:
            print(f"\nGPU configuration error: {e}")
            print("Falling back to CPU...")
    else:
        print("\nNo GPU detected.")
    
    print("="*60 + "\n")
    

    
    # Configuration
    TRAIN_DIR = r'E:\new_data\img'
    TEST_DIR = r'E:\new_data\test_img'

    BATCH_SIZE = 16
    EPOCHS = 50
    IMG_SIZE = 512
    FINE_TUNE_EPOCHS = 25
    

    def preprocess_input(x):
        return (x / 127.5) - 1.0

    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print(f"Error: TRAIN_DIR or TEST_DIR not found.")
        return

    data = []
    class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    num_classes = len(class_names)

    for label in class_names:
        class_path = os.path.join(TRAIN_DIR, label)
        for img in os.listdir(class_path):
            data.append({'filename': os.path.join(class_path, img), 'class': label})

    df = pd.DataFrame(data)

    test_df_data = [{'f': os.path.join(TEST_DIR, c, i), 'c': c} 
                    for c in class_names for i in os.listdir(os.path.join(TEST_DIR, c))]
    test_df = pd.DataFrame(test_df_data)
    
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)
    

    final_test_generator = datagen.flow_from_dataframe(
        test_df, x_col='f', y_col='c', 
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, 
        class_mode='categorical', shuffle=False, classes=class_names
    )

    all_history = []
    
    fold_accuracies = []
    
    # store test accuracy for the best model of each fold
    fold_test_accuracies = []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['filename'], df['class'])):
        print(f"\nFOLD {fold + 1}")
        print("-" * 30)
        
        tf.keras.backend.clear_session()
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_generator = datagen.flow_from_dataframe(
            train_df, x_col='filename', y_col='class',
            target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, 
            class_mode='categorical', classes=class_names
        )
        val_generator = datagen.flow_from_dataframe(
            val_df, x_col='filename', y_col='class',
            target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='categorical',
            shuffle=False, classes=class_names
        )
        
        # Create and compile model for each fold
        model = create_model(num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        def lr_schedule(epoch, lr):
            if epoch > 0 and epoch % 5 == 0:
                return lr * 0.5
            return lr

        callbacks = [
            tf.keras.callbacks.LearningRateScheduler(lr_schedule),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        ]
        
        fold_model_path = os.path.join(script_dir, f'model_fold_{fold + 1}.h5')
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            fold_model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
        ))

        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )

        all_history.append(history.history)
        best_acc = max(history.history['val_accuracy'])
        fold_accuracies.append(best_acc)
        
        best_model_this_fold = tf.keras.models.load_model(fold_model_path)
        
        # Calculate accuracy averaged across all classes on TEST_DIR
        y_pred = np.argmax(best_model_this_fold.predict(final_test_generator), axis=1)
        y_true = final_test_generator.classes
        cm = confusion_matrix(y_true, y_pred)
        # class accuracy calculation
        class_accs = []
        for i in range(num_classes):
            if cm[i].sum() > 0:
                class_accs.append(cm[i, i] / cm[i].sum())
            else:
                class_accs.append(0.0)
        
        avg_class_acc = np.mean(class_accs)
        fold_test_accuracies.append(avg_class_acc)
        print(f"Fold {fold+1} Diagnostic Test Accuracy: {avg_class_acc*100:.2f}%")

        # Save fold accuracy image
        plt.figure(figsize=(8, 5))
        epochs_range = range(1, len(history.history['accuracy']) + 1)
        plt.plot(epochs_range, history.history['accuracy'], label='Train Accuracy')
        plt.plot(epochs_range, history.history['val_accuracy'], label='Val Accuracy')
        plt.title(f'Fold {fold+1} Accuracy (Best Val: {best_acc:.4f})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        fold_plot_path = os.path.join(script_dir, f'fold_{fold+1}_accuracy.png')
        plt.savefig(fold_plot_path)
        plt.close()
        
    # --- After K-Fold Cross-Validation ---
    
    # 1. Select the best model based *only* on validation accuracy from K-Fold
    best_fold_idx = np.argmax(fold_accuracies)
    best_val_accuracy = fold_accuracies[best_fold_idx]
    print(f"\n--- Model Selection Complete ---")
    print(f"Best performing fold based on validation accuracy: Fold {best_fold_idx + 1} (Validation Accuracy: {best_val_accuracy*100:.2f}%)")
    
    # 2. Load the best model from the K-Fold process
    best_kfold_model_path = os.path.join(script_dir, f'model_fold_{best_fold_idx + 1}.h5')
    best_kfold_model = tf.keras.models.load_model(best_kfold_model_path)
    
    # 3. Evaluate this best K-Fold model on the separate testing set (Pre-Retraining)
    print(f"\n--- Evaluating Best K-Fold Model (Fold {best_fold_idx + 1}) on Test Set (Pre-Retraining) ---")
    y_pred_best_kfold = np.argmax(best_kfold_model.predict(final_test_generator), axis=1)
    y_true_test = final_test_generator.classes
    
    cm_best_kfold = confusion_matrix(y_true_test, y_pred_best_kfold)
    overall_acc_best_kfold = accuracy_score(y_true_test, y_pred_best_kfold)
    print(f"Overall Test Accuracy for Best K-Fold Model: {overall_acc_best_kfold*100:.2f}%")
    
    class_accs_best_kfold = []
    for i in range(num_classes):
        if cm_best_kfold[i].sum() > 0:
            class_accs_best_kfold.append(cm_best_kfold[i, i] / cm_best_kfold[i].sum())
        else:
            class_accs_best_kfold.append(0.0)
    
    print("\nDetailed Per-Class Accuracy for Best K-Fold Model (Pre-Retraining):")
    for i, acc in enumerate(class_accs_best_kfold):
        print(f"  {class_names[i]}: {acc*100:.2f}%")
    
    # Plotting per-class accuracy for best K-Fold model
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, [a * 100 for a in class_accs_best_kfold])
    plt.title(f'Per-Class Test Accuracy for Best K-Fold Model (Fold {best_fold_idx + 1})')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'best_kfold_pre_retrain_per_class_accuracy.png'))
    plt.close()

    # fine tune the best model on the entire training dataset
    print("\n--- Fine-tuning Best K-Fold Model on Entire Training Dataset ---")
    full_train_df = df # All data used in K-Fold is now the full training set
    full_train_generator = datagen.flow_from_dataframe(
        full_train_df, x_col='filename', y_col='class',
        target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, 
        class_mode='categorical', classes=class_names
    )
    
    # Load the best model weights for fine-tuning instead of starting from zero
    final_retrained_model = tf.keras.models.load_model(best_kfold_model_path)
    
    final_retrained_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    final_retrained_model_path = os.path.join(script_dir, 'final_model_retrained.h5')
    final_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        final_retrained_model_path, monitor='accuracy', save_best_only=True, mode='max', verbose=1
    ) # Monitor training accuracy as no separate validation set here
    
    final_train_history = final_retrained_model.fit(
        full_train_generator,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=[final_checkpoint], 
        verbose=1
    )
    
    # Plotting epoch vs accuracy for final training
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(final_train_history.history['accuracy']) + 1), final_train_history.history['accuracy'], label='Train Accuracy')
    plt.title('Final Model Training Accuracy on Full Dataset')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(script_dir, 'final_model_full_train_accuracy.png'))
    plt.close()

    # 5. Evaluate the retrained final model on the separate testing set (Post-Retraining)
    print("\n--- Evaluating Retrained Final Model on Test Set (Post-Retraining) ---")
    loaded_retrained_model = tf.keras.models.load_model(final_retrained_model_path)
    
    y_pred_final = np.argmax(loaded_retrained_model.predict(final_test_generator), axis=1)
    
    cm_final = confusion_matrix(y_true_test, y_pred_final)
    overall_acc_final = accuracy_score(y_true_test, y_pred_final)
    print(f"Overall Test Accuracy for Retrained Final Model: {overall_acc_final*100:.2f}%")
    
    class_accs_final = []
    for i in range(num_classes):
        if cm_final[i].sum() > 0:
            class_accs_final.append(cm_final[i, i] / cm_final[i].sum())
        else:
            class_accs_final.append(0.0)
    
    print("\nDetailed Per-Class Accuracy for Retrained Final Model (Post-Retraining):")
    for i, acc in enumerate(class_accs_final):
        print(f"  {class_names[i]}: {acc*100:.2f}%")
        
    # Plotting per-class accuracy for retrained final model
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, [a * 100 for a in class_accs_final])
    plt.title('Per-Class Test Accuracy for Retrained Final Model')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'final_model_post_retrain_per_class_accuracy.png'))
    plt.close()

    # 6. Convert the retrained final model to TFLite
    print(f"\nConverting retrained final model to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(loaded_retrained_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # Enable default optimizations for mobile deployment
    tflite_model = converter.convert()
    tflite_path = os.path.join(script_dir, 'final_model_retrained.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Retrained model converted to TFLite and saved to: {tflite_path}")

    total_time = time.time() - start_time
    print(f"\nTotal Training Complete in {total_time/60:.2f} minutes.")
    print(f"Average Test Accuracy across {k_folds} folds: {np.mean(fold_test_accuracies)*100:.2f}%")

    # --- Consolidated Plotting ---
    # Visualization matching PyTorch training_results.png format
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for i, h in enumerate(all_history):
        plt.plot(h['loss'], label=f'Fold {i+1} (Final: {h["loss"][-1]:.4f})')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, h in enumerate(all_history):
        plt.plot(h['val_accuracy'], label=f'Fold {i+1} (Best: {fold_accuracies[i]:.2f})')
    plt.title('Summary: Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(script_dir, 'training_results.png')
    plt.savefig(plot_path)
    print(f"\nGraphs saved to {plot_path}")

    # Box Plot for Test Accuracy comparison
    plt.figure(figsize=(8, 6))
    plt.boxplot(fold_test_accuracies, labels=['Folds 1-5'])
    plt.title('Distribution of Best Model Test Accuracies Across Folds')
    plt.ylabel('Accuracy (Macro-Average)')
    plt.savefig(os.path.join(script_dir, 'test_accuracy_boxplot.png'))
    
    plt.show()

if __name__ == "__main__":
    main()
