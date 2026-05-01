import tensorflow as tf
import os


tf.keras.backend.clear_session()
tf.keras.mixed_precision.set_global_policy('float32')

def convert(h5_path, tflite_path, num_classes=5):
    print(f"--- Starting Conversion ---")
    
    print("Building clean float32 architecture...")
    model_f32 = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(512, 512, 3), dtype='float32'),
        
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
        tf.keras.layers.Dense(num_classes) # Final output layer
    ])


    print(f"Extracting weights from {h5_path}...")
    trained_model_f16 = tf.keras.models.load_model(h5_path, compile=False)
    
    model_f32.set_weights(trained_model_f16.get_weights())
    print("✓ Weights successfully sanitized to float32.")

    print("Converting to standard TFLite (No Flex Ops)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model_f32)
    
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"\n✓ DONE! File saved to: {tflite_path}")
        print("This model will now run on mobile without errors.")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    INPUT_H5 = r'E:\new_data\final_model_retrained.h5'
    OUTPUT_TFLITE = r'E:\new_data\model.tflite'
    CLASSES = 5 # Change this if your model has a different number of classes
    
    convert(INPUT_H5, OUTPUT_TFLITE, CLASSES)