from flask import Flask, request, jsonify
import librosa
from collections import Counter
import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model
import os
app = Flask(__name__)
def get_CryDataSet(number):
    info = {
        1: 'Hungry',
        2: 'belly_pain',
        3: 'discomfort',
        4: 'No CryBaby'
    }
    return info[number]

def processing_audio(data, sr, option):
        func = random.choice(option)
        if func == 'Standard':
            processed = data
        else:
            processed, _ = func(data, sr)
        return processed
def readAudio(path):
    data ,sample_rate = librosa.load(path,duration=2.4 ,offset=0.6)
    return data , sample_rate

def add_noise(data, sr):
    noise = 0.035*np.random.uniform()*np.amax(data)
    data += noise * np.random.normal(size=data.shape[0])
    return data ,sr
def pitch(data, sr, factor=0.7):
    pitched = librosa.effects.pitch_shift(y=data, sr=sr, n_steps=factor)
    return pitched ,sr
def feature(data ,sr):
    mfcc=librosa.feature.mfcc(y=data, sr=sr)
    return mfcc


def get_Feature(path):
    data, sample_rate = readAudio(path)
    funcs = ['Standard', add_noise, pitch]
    random.choice(funcs)

    features = []

    f1 = processing_audio(data, sample_rate, funcs)
    f2 = processing_audio(f1, sample_rate, funcs)
    V_feature = feature(f1, sample_rate)
    if V_feature.shape == (20, 104):
        features.append(V_feature)
    f1 = processing_audio(data, sample_rate, funcs)
    f2 = processing_audio(f1, sample_rate, funcs)
    V_feature = feature(f1, sample_rate)

    if V_feature.shape == (20, 104):
        features.append(V_feature)
    return features

# Load the encoder
encoder = OneHotEncoder()
encoder.fit_transform(np.array([1, 2, 3, 4, 1, 5, 6, 7, 8]).reshape(-1, 1)).toarray()

# Function to process audio and make predictions
def process_and_predict(path):
    data, sample_rate = librosa.load(path, duration=2.4, offset=0.6)
    features = get_Feature(path)  # Assuming get_Feature function is defined elsewhere
    features = np.expand_dims(features, axis=3)  # Add channel dimension
    features = np.expand_dims(features, axis=3)  # Add time dimension
    features = np.swapaxes(features, 1, 2)  # Reshape for model input
    predictions = CryModel.predict(features)
    predicted_classes = encoder.inverse_transform(predictions)
    predicted_classes = [get_CryDataSet(value) for value in predicted_classes.flatten()]
    most_common_class = Counter(predicted_classes).most_common(1)[0][0]
    return most_common_class


# Load the model
CryModel = load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Check file extension
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ['wav', 'mp3']:
            return jsonify({'error': 'Unsupported file format'})

        # Save the file to a temporary location
        temp_file_path = f'temp_audio.{file_extension}'
        file.save(temp_file_path)

        # Process and predict
        predicted_classes = process_and_predict(temp_file_path)

        # Delete the temporary file
        os.remove(temp_file_path)

        return jsonify({'predicted_classes': predicted_classes})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "main":
    app.run(host='127.0.0.1', port=9988)