import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.h5')

image=cv2.imread(r'C:\Users\ansar\OneDrive\Desktop\mini project\BrainTumor Classification DL\datasets\yes\y1.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=np.argmax(model.predict(input_img))

if result == 0:
    print("NO BRAIN TUMOR DETECTED")
else:
    print("BRAIN TUMOR DETECTED")




