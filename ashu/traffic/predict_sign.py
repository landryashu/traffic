import tkinter as tk
from tkinter import filedialog, Label, Button, Canvas
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import sys

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43

def preprocess_image(image_path):
    """Preprocess the image for prediction."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def classify_image():
    """Open file dialog, classify image, and display result."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return
    
    # Preprocess and predict
    img = preprocess_image(file_path)
    prediction = model.predict(img)
    class_label = np.argmax(prediction)
    
    # Display image
    img_pil = Image.open(file_path)
    img_pil = img_pil.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img_pil)
    img_canvas.create_image(100, 100, image=img_tk, anchor=tk.CENTER)
    img_canvas.image = img_tk
    
    # Display result
    result_label.config(text=f"Predicted Category: {class_label}")

def reset_app():
    """Reset the application state."""
    img_canvas.delete("all")
    result_label.config(text="Prediction will appear here")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python traffic_gui.py model.h5")
    
    MODEL_PATH = sys.argv[1]
    model = load_model(MODEL_PATH)
    
    # Create GUI window
    root = tk.Tk()
    root.title("Traffic Sign Classifier")
    root.geometry("400x500")
    
    # UI Elements
    Button(root, text="Upload Image", command=classify_image).pack(pady=10)
    
    # Canvas for displaying image (200x200)
    img_canvas = Canvas(root, width=200, height=200, bg="lightgray")
    img_canvas.pack()
    
    result_label = Label(root, text="Prediction will appear here", font=("Arial", 14))
    result_label.pack(pady=10)
    
    # Reset Button
    Button(root, text="Reset", command=reset_app).pack(pady=10)
    
    # Run GUI loop
    root.mainloop()
