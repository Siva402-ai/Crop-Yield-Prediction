import numpy as np
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# Load pretrained CNN model
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def extract_image_features(img_path):

    # load image
    img = image.load_img(img_path, target_size=(224,224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    # extract features
    features = model.predict(img_array)

    # convert vector to single score
    feature_score = float(np.mean(features))

    return feature_score