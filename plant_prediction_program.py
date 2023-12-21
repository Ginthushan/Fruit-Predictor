import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import customtkinter
import numpy as np
import cv2
import tensorflow as tf

# Load the TensorFlow model
model = tf.keras.models.load_model('modelcnn.h5')
   
# Save image data into a variable
global image_data

# Loads the image
def load_image(max_size=(400, 400)):
    global image_data
    file_path = filedialog.askopenfilename(title="Pick an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image = cv2.imread(file_path)
        if image is not None:
            img_height, img_width = image.shape[:2]
            max_height, max_width = max_size
            if img_height > max_height or img_width > max_width:
                scale = min(max_height / img_height, max_width / img_width)
                new_height = int(img_height * scale)
                new_width = int(img_width * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            image_data = image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = ImageTk.PhotoImage(image=Image.fromarray(image))
            image_panel.image = image
            image_panel.create_image(0, 0, anchor="nw", image=image)

# Clear the displayed image and reset image_data
def clear_image():
    global image_data
    image_panel.delete("all")
    image_data = None

# Predict the plant type using the loaded model
def predict():
    global image_data
    if image_data is not None:
        # Preprocess the image
        img = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (50, 50))
        img = img / 255.0  # Normalize pixel values

        # Make prediction
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        prediction = model.predict(img)
        predicted_class = get_category_name(np.argmax(prediction))
        confidence_level = np.max(prediction)

        # Show the prediction in a pop-up window
        pop_ui(predicted_class, confidence_level)
    else:
        print("Please load an image first")


#Pop up window
def pop_ui(predicted_class, confidence_level):
    top = customtkinter.CTkToplevel()
    top.geometry("200x100")
    top.title("Prediction Result")

    frame_1 = customtkinter.CTkFrame(master=top)
    frame_1.pack(pady=10, padx=10, fill="both", expand=True)

    label_1 = customtkinter.CTkLabel(frame_1, text=f"Predicted Plant: {predicted_class}")
    label_2 = customtkinter.CTkLabel(frame_1, text=f"Confidence Level: {confidence_level:.2%}")

    label_1.pack()
    label_2.pack()


#Categorie Names
def get_category_name(index):
    categories = [
        "pineapple", "waterapple", "cassava", "watermelon", "eggplant",
        "peperchili", "cucumber", "curcuma", "aloevera", "kale",
        "paddy", "longbeans", "soybeans", "cantaloupe", "banana",
        "galangal", "ginger", "papaya", "coconut", "orange",
        "tobacco", "mango", "guava", "spinach", "melon",
        "pomelo", "bilimbi", "shallot", "sweetpotatoes", "corn"
    ]

    if 0 <= index < len(categories):
        return categories[index]
    else:
        return "Invalid Index"

# Setting Custom Appearance for the application
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

# Create Window
root = customtkinter.CTk()

# Title Window
root.title("Plant Type Classification")

# Size of the window
root.geometry("600x600")

# Creating the Frames
frame_1 = customtkinter.CTkFrame(master=root)
frame_2 = customtkinter.CTkFrame(master=root)
frame_1.pack(pady=10, padx=10, fill="both", expand=True)
frame_2.pack(pady=10, padx=10, fill="both", expand=True)

# First Frame Items
image_panel = customtkinter.CTkCanvas(master=frame_1, width=400, height=400, bg='white')
image_panel.pack(pady=30, padx=10)
image_panel_image = image_panel.create_image(0, 0, anchor="nw")

# Second Frame Buttons
load_button = customtkinter.CTkButton(frame_2, text="Load Image", command=load_image)
load_button.pack(side=tk.LEFT, pady=10, padx=20)

predict_button = customtkinter.CTkButton(frame_2, text="Predict", command=predict)
predict_button.pack(side=tk.LEFT, pady=10, padx=20)

clear_image_button = customtkinter.CTkButton(frame_2, text="Clear Image", command=clear_image)
clear_image_button.pack(side=tk.LEFT, pady=10, padx=20)

# Run Root Window
root.mainloop()

