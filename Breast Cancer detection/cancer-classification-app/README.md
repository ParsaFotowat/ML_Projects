# Cancer Classification App

This project is a Streamlit application for classifying breast cancer using a neural network model. The app allows users to input features related to breast cancer and receive predictions on whether the cancer is malignant or benign.

## Project Structure

```
cancer-classification-app
├── src
│   ├── app.py                # Main entry point for the Streamlit application
│   ├── data
│   │   └── B_cancer_data.csv # Dataset used for training the model
│   ├── models
│   │   └── model.pkl         # Serialized machine learning model
│   └── utils
│       └── preprocessing.py   # Utility functions for data preprocessing
├── requirements.txt          # List of dependencies
├── README.md                 # Documentation for the project
└── .streamlit
    └── config.toml          # Configuration settings for the Streamlit app
```

## Installation

To run this application, you need to have Python installed on your machine. Follow these steps to set up the project:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd cancer-classification-app
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the App

To run the Streamlit app, execute the following command in your terminal:
```
streamlit run src/app.py
```

This will start the Streamlit server, and you can access the app in your web browser at `http://localhost:8501`.

## Usage

1. Input the required features in the provided fields.
2. Click on the "Predict" button to get the classification result.
3. The app will display whether the cancer is predicted to be malignant or benign based on the input features.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.