import streamlit as st
import pandas as pd
from utils.data_processing import load_data
from utils.model import recommend_items, load_model
from utils.visualization import plot_figures

# Load the model and data
model = load_model()
df = load_data()

st.title("Fashion Recommendation System")

# User input for selecting an item
item_id = st.selectbox("Select an item ID for recommendations:", df['id'].values)

if st.button("Get Recommendations"):
    idx_rec = recommend_items(df['embedding'].values, item_id, top_n=6)

    # Display the selected item with its name
    selected_item = df.loc[df['id'] == item_id]
    st.image(selected_item['image'].values[0], caption=f"Selected Item: {selected_item['name'].values[0]}")

    # Display recommended items
    figures = {f'Recommended Item {i+1}': df.loc[idx_rec[i], 'image'] for i in range(len(idx_rec))}
    plot_figures(figures, nrows=2, ncols=3)

st.sidebar.header("About")
st.sidebar.text("This application recommends fashion items based on user-selected input.")