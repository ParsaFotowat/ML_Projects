import pandas as pd
import numpy as np
import os

def load_data(metadata_path, embeddings_path):
    # Load metadata (e.g., item IDs, names, image paths)
    metadata = pd.read_excel(metadata_path)
    
    # Load embeddings (e.g., precomputed feature vectors)
    embeddings = np.load(embeddings_path)
    
    # Add embeddings as a new column in the metadata DataFrame
    metadata['embedding'] = list(embeddings)
    
    return metadata

def preprocess_data(df):
    """Preprocess the dataset for the model."""
    df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
    df = df.reset_index(drop=True)
    return df

def get_image_path(img_name, base_path):
    """Generate the full image path."""
    return os.path.join(base_path, "images", img_name)

def load_image(img_name, base_path, resized_fac=0.1):
    """Load and resize an image."""
    img_path = get_image_path(img_name, base_path)
    img = cv2.imread(img_path)
    w, h, _ = img.shape
    resized = cv2.resize(img, (int(h * resized_fac), int(w * resized_fac)), interpolation=cv2.INTER_AREA)
    return resized

df = load_data(
    metadata_path="C:/Users/admin/Desktop/Apps/Resume and Shit/Portfolio/Projects/ML Apps/ML_Projects/fashion-rec-system/Data/metadados.csv",
    embeddings_path="C:/Users/admin/Desktop/Apps/Resume and Shit/Portfolio/Projects/ML Apps/ML_Projects/fashion-rec-system/Data/embeddings.npy"
)