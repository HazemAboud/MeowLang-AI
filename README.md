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

The final dataset had 2987 samples across 5 classes, namely Angry, Food, Isolation, MotherCall, Resting. I preserved the original class names.

# Model Development

Initially I used a CNN model with 4 blocks
<img width="521" height="3000" alt="model_viz" src="https://github.com/user-attachments/assets/b64413e0-8ce1-4724-abdf-52d17967bf6b" />

This model trained using Pytorch scored 93%~ accuracy, but after switching to tensorflow and changing the spectrogram converison script from using librosa to a custom script in dart (this was done to remove the dependency on the Flask server and perform all processes locally on phone) the model had significantly lower acuracy of 84%~ therefore I retrained the model again using a 6 block network


# Training
First I split the data into training and testing sets 80:20 respectively

The model was then trained using stratified K-fold technique with k=5 to reduce overfitting.


