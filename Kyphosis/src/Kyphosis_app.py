import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os

# Title of the app
st.title("Kyphosis Classification App")

# Load default dataset or allow user upload
st.sidebar.header("Dataset Options")
default_dataset = st.sidebar.checkbox("Use default Kyphosis dataset", value=True)

if default_dataset:
    # Load the default dataset from the data folder
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'kyphosis.csv')
    if not os.path.exists(dataset_path):
        st.error(f"Dataset not found at {dataset_path}. Please check the file path.")
        st.stop()
    df = pd.read_csv(dataset_path)
    st.write("### Default Kyphosis Dataset Loaded")
else:
    # Allow user to upload their own dataset
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset Preview")
    else:
        st.warning("Please upload a dataset to proceed.")
        st.stop()

# Display dataset preview
st.write(df.head())

# Data visualization
st.write("### Pairplot")
hue_column = st.selectbox("Select the target column for hue", df.columns)
if st.button("Generate Pairplot"):
    fig = sns.pairplot(df, hue=hue_column, palette='Set1')
    st.pyplot(fig)

# Splitting data
if "Kyphosis" not in df.columns:
    st.error("The dataset must contain a 'Kyphosis' column for prediction.")
    st.stop()

x = df.drop("Kyphosis", axis=1)
y = df["Kyphosis"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

# Decision Tree Classifier
st.write("### Decision Tree Classifier")
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)
dtree_predictions = dtree.predict(x_test)
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, dtree_predictions))
st.write("Classification Report:")
st.text(classification_report(y_test, dtree_predictions))

# Random Forest Classifier
st.write("### Random Forest Classifier")
n_estimators = st.slider("Number of estimators", min_value=10, max_value=500, value=200, step=10)
rfc = RandomForestClassifier(n_estimators=n_estimators)
rfc.fit(x_train, y_train)
rfc_predictions = rfc.predict(x_test)
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, rfc_predictions))
st.write("Classification Report:")
st.text(classification_report(y_test, rfc_predictions))