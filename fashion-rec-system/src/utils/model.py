from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.layers import GlobalMaxPooling2D
from tensorflow import keras  
import numpy as np
import pandas as pd

def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    return model

def get_embedding(model, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).reshape(-1)

def recommend_items(embeddings, idx, top_n=5):
    cosine_sim = 1 - pairwise_distances(embeddings, metric='cosine')
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    idx_rec = [i[0] for i in sim_scores[1:top_n+1]]
    return idx_rec