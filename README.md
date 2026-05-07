# Introduction

This is a detailed walkthrough of the data cleaning process and model development for my project MeowLang

# Data Preprocessing

I used two datasets for this project

Cat Meow Classification - 440 audio samples across 3 categories Food, Isolation and Brushing

https://www.kaggle.com/datasets/andrewmvd/cat-meow-classification

Cat Sound Classification Dataset V2 - 5923 audio samples across 10 categories

https://zenodo.org/records/4724180 (this dataset is private)

First I used silhoutte scores and t-sne to assess data quality after converting the audio to spectrogram

<img width="1317" height="774" alt="Screenshot 2026-05-05 170604" src="https://github.com/user-attachments/assets/e12aa4d0-9d94-4a57-967c-c8710e3dadab" />

From the graph and silhoutte scores classes have low interclass separation, I removed low quality classes and sytehsized data for Food and Isolation classes by delaying or speeding the audio, increasing or reducing the pitch, I removed the class with the lowest silhoutte score and rechecked silhoutte scores after removing each class I rechecked class dissimilarity mainly using t-sne until I ended with 5 classes that have a relatively high interclass dissimilarity

<img width="1298" height="800" alt="image" src="https://github.com/user-attachments/assets/62809d8d-5b3f-49aa-82c2-bbf68c05298d" />

The decision to proceed with these 5 classes was also based on the classes I wanted the model to predict for the use cases the applicaiton was designed for in addition to the class quality as mentioned before

The final dataset had 2903 samples across 5 classes, namely Angry, Food, Isolation, MotherCall, Resting. I preserved the original class names.

# Model Development

Initially I used a CNN model with 4 blocks. This model trained using Pytorch scored 93%~ accuracy, but after switching to tensorflow and changing the spectrogram converison script from using librosa to a custom script in dart (this was done to remove the dependency on the Flask server and perform all processes locally on phone) the model had significantly lower acuracy of 84%~ therefore I retrained the model again using a 6 block network

# Training
First I split the data into training and testing sets 80:20 respectively

Then I performed Stratified K-fold cross validaiton using 5 folds to validate the model structure and hyper parameters

| Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| :---: | :---: | :---: | :---: | :---: |
| ![Fold 1](https://github.com/user-attachments/assets/4a5e3ee2-f520-4d66-9ea0-26ad76dbedf7) | ![Fold 2](https://github.com/user-attachments/assets/fb5f71a1-4a29-4fe3-bdb9-792183e7af3c) | ![Fold 3](https://github.com/user-attachments/assets/5eaf9d67-c097-4a04-8b15-ef56ede0bc36) | ![Fold 4](https://github.com/user-attachments/assets/b99a2612-2114-48ee-93ed-b97ed0efc02d) | ![Fold 5](https://github.com/user-attachments/assets/0774fb96-1420-40db-be6e-8e7888186a49) |

After choosing best model strucutre and hyper parameters based on the validation accuracy the final model was trained on the whole training set with the following structure and paramaters



```
IMG_SIZE    = 512
BATCH_SIZE  = 24
EPOCHS      = 50
```

```
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
```
<img width="1200" height="500" alt="training_curves_final" src="https://github.com/user-attachments/assets/5613b9b5-2a44-41ba-b4d6-d68baedace54" />

# Results
The best weights were chosen based on the training loss during the final model training, then the final model was tested on the testing set

<img width="1536" height="754" alt="Figure_1" src="https://github.com/user-attachments/assets/b328f7db-fde2-45b7-a49e-40fd06f857a8" />








