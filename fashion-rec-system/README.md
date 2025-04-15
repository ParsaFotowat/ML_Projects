# Fashion Recommendation System

This project is a Fashion Recommendation System built using Streamlit and TensorFlow. It leverages a pre-trained model to provide fashion item recommendations based on user input images.

## Project Structure

```
fashion-rec-system
├── src
│   ├── app.py                # Main entry point for the Streamlit application
│   ├── utils
│   │   ├── data_processing.py # Functions for loading and processing the dataset
│   │   ├── model.py          # Model architecture and recommendation functions
│   │   └── visualization.py   # Functions for visualizing data and recommendations
├── requirements.txt          # List of dependencies
├── README.md                 # Project documentation
└── .streamlit
    └── config.toml          # Configuration settings for the Streamlit application
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd fashion-rec-system
   ```

2. **Install the required packages:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   Execute the following command in your terminal:
   ```bash
   streamlit run src/app.py
   ```

## Usage

- Upload an image of a fashion item to receive recommendations for similar items.
- The application will display the recommended items along with their images.

## Acknowledgments

- This project utilizes TensorFlow for model training and inference.
- Special thanks to the contributors of the datasets used in this project.