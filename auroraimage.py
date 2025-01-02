import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.applications.MobileNetV2(weights='imagenet') #Load a already pretrained model.

#Func to preprocces the image.

def pre_proc_img(img_path) { 
  image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224)) #Load the image with the size 224 by 244.

  in_arr = tf.keras.preprocessing.image.img_to_array(image) #Make the image into a array.
  in_arr = np.expand_dims(in_arr, axis=0) #
  return  tf.keras.applications.mobilenet_v2.preprocess_input(in_arr) #return the image array with the mode.

}

def decode_pred(pred): 
  return tf.keras.applications.mobilenet_v2.decode_predictions(pred, top=3)[0]



def predict_image(img_path):
  in_arr = pre_proc_img(img_path) #Make the array.
  predict = model.predict(in_arr) # Predict the result.
  decoded_predict = decode_pred(predict) #Decode the prediction.

  for i, (imagenet_id, label, score) in enumerate(decoded_predict):  #For all predictions
    #Print all prediction, with the label and the percentage.
    print(f"{i+1}. {label} {score * 100: .2f}") #Use i+1 as python is 0 based indexed, take the score times 100 as it comes in decimal form and shorten it down to two decimals.
    
 
