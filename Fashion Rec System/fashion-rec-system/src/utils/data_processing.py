import pandas as pd
import os

def load_data(file_path):
    """Load the dataset from a CSV file."""
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        return data
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

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