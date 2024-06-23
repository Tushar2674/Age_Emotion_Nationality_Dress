import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

# Load models without optimizer state
def load_model_without_optimizer(path):
    model = load_model(path, compile=False)
    # Compile the model to match its original setup if necessary
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the trained models
age_model = load_model('best_age_model.keras')
nationality_model = load_model('nation_model.keras')
emotion_model = load_model('emotion_detection_model.keras')
dress_color_model = load_model('dress_model.keras')

# Load class indices from JSON files for making labels 
with open('nationality_class_indices.json', 'r') as f:
    nationality_classes = {v: k for k, v in json.load(f).items()}

with open('emotion_class_indices.json', 'r') as f:
    emotion_classes = {v: k for k, v in json.load(f).items()}

with open('dress_color_class_indices.json', 'r') as f:
    dress_color_classes = {v: k for k, v in json.load(f).items()}

# Initialize Tkinter
root = tk.Tk()
root.title("Image Prediction")

# Function to preprocess the uploaded image
def preprocess_image(image_path, target_size=(150,150)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to handle image upload and prediction
def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img = Image.open(file_path)
            img = img.resize((150,150), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            image_label.config(image=img)
            image_label.image = img  
            
            img_array = preprocess_image(file_path)
            
            # Predict nationality 
            nationality_pred_idx = np.argmax(nationality_model.predict(img_array))
            nationality_pred = nationality_classes[nationality_pred_idx]
            
            # Predict age
            age_pred_probabilities = age_model.predict(img_array)[0]
            age_pred_old = np.argmax(age_pred_probabilities)
            age_pred=int(age_pred_old)+10
            if age_pred < 10 or age_pred > 60:
                raise ValueError("Age prediction out of range (10-60)")
            
            # Predict emotion
            emotion_pred_probabilities = emotion_model.predict(img_array)[0]
            emotion_pred_idx = np.argmax(emotion_pred_probabilities)
            emotion_pred = emotion_classes[emotion_pred_idx]
                                 
            # Predict dress color
            dress_color_pred_idx = np.argmax(dress_color_model.predict(img_array))
            dress_color_pred = dress_color_classes[dress_color_pred_idx]
            
            # Display predictions based on nationality
            if nationality_pred == 'Indian':
                result_label.config(text=f"Nationality: {nationality_pred}\n"
                                         f"Age: {age_pred}\n"
                                         f"Emotion: {emotion_pred}\n"
                                         f"Dress Color: {dress_color_pred}")
            elif nationality_pred == 'United States':
                result_label.config(text=f"Nationality: {nationality_pred}\n"
                                         f"Age: {age_pred}\n"
                                         f"Emotion: {emotion_pred}")
            elif nationality_pred == 'African':
                result_label.config(text=f"Nationality: {nationality_pred}\n"
                                         f"Emotion: {emotion_pred}\n"
                                         f"Dress Color: {dress_color_pred}")
            else:
                result_label.config(text=f"Nationality: {nationality_pred}\n"
                                         f"Emotion: {emotion_pred}")
        
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Create GUI components
upload_button = tk.Button(root, text="Upload Image", command=predict_image)
upload_button.pack(pady=20)

image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="")
result_label.pack(pady=20)

# Run the Tkinter main loop
root.mainloop()
