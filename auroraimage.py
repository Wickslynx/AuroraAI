import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import customtkinter as ctk
import tkinter as tk 
from tkinter import filedialog

model = tf.keras.applications.MobileNetV2(weights='imagenet') #Load a already pretrained model.

#Func to preprocces the image.
def pre_proc_img(img_path):
  image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224)) #Load the image with the size 224 by 244.

  in_arr = tf.keras.preprocessing.image.img_to_array(image) #Make the image into a array.

  in_arr = np.expand_dims(in_arr, axis=0) #Add extra dimension.
  
  return tf.keras.applications.mobilenet_v2.preprocess_input(in_arr) #return the image array with the mode.




def decode_pred(pred): 
  return tf.keras.applications.mobilenet_v2.decode_predictions(pred, top=3)[0] #Decode the prediction.



def predict_image(img_path):
  in_arr = pre_proc_img(img_path) #Make the array.
  predict = model.predict(in_arr) # Predict the result.
  decoded_predict = decode_pred(predict) #Decode the prediction.

  
  for i, (imagenet_id, label, score) in enumerate(decoded_predict):  #For all predictions
    out_txt += f"{i+1}. {label} {score * 100: .2f}" #Use i+1 as python is 0 based indexed, take the score times 100 as it comes in decimal form and shorten it down to two decimals.

  return out_txt #Return the prediction.
  

class AuroraImageUI:
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.window = ctk.CTk()
        self.window.title("AuroraAI-Image")
        self.window.geometry("1000x800") 

        self.header1 = ctk.CTkLabel(self.window, text="Image prediction software.", font=("Roboto", 100, "bold"))
        self.header1.pack(padx=10, pady=10)
      
        self.button1 = ctk.CTKButton(self.window, text="Choose file", command=open_file_ui)
        self.button1.pack(padx=10, pady=10)

        self.result_txt = tk.text(self.window, height=10, width=80, state="disabled")
        self.result_text.pack(padx=10, pady=10)

       
        self.window.mainloop()
      
    def open_file_ui():
      file_path = filedialog.askopenfilename(
        title="Select file:", 
        filetype=(("Image files", "*.jpg *.jpeg *.png *.gif *.bmp"), ("All files", "*.*"))
    )  
    
    if file_path: 
       self.result_txt.configure(state="normal")
       self.result_txt.delete(1.0, tk.END) 
       self.result_txt.insert(tk.END, predict_image(file_path)) 
       self.result_txt.configure(state="disabled")

    
 
