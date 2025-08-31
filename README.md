# Depression Risk Prediction Model

This project implements a deep learning model to predict depression risk based on mental health survey data. The model uses various personal, academic, professional, and lifestyle factors to assess the risk of depression.

## Project Overview

The Depression Risk Prediction Model aims to identify individuals at risk of depression using machine learning techniques. The project includes data preprocessing, model development, evaluation, and a user-friendly web application for making predictions.

## Project Structure

```
Mental_Health_Survey_Project/
├── app/                      # Streamlit web application
│   ├── app.py                # Main application file
│   └── utils.py              # Utility functions for the app
├── data/                     # Data directory
│   ├── raw/                  # Raw data files
│   │   ├── train.csv         # Training data
│   │   └── test.csv          # Test data
│   └── processed/            # Processed data files
├── models/                   # Saved model files
├── notebooks/                # Jupyter notebooks for exploration
├── src/                      # Source code
│   ├── data/                 # Data processing modules
│   │   └── preprocess.py     # Data preprocessing
│   ├── models/               # Model definition modules
│   │   └── model.py          # Neural network model
│   ├── evaluation/           # Evaluation modules
│   │   └── metrics.py        # Evaluation metrics
│   └── visualization/        # Visualization modules
│       └── visualize.py      # Data visualization
├── visualizations/           # Generated visualizations
├── results/                  # Evaluation results
├── config.yaml               # Configuration file
├── train_model.py            # Main training script
└── requirements.txt          # Project dependencies
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Mental_Health_Survey_Project
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

The data preprocessing pipeline handles missing values, categorical encoding, and feature scaling:

```python
from src.data.preprocess import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Load and preprocess data
data_dict = preprocessor.load_and_preprocess()
```

### Model Training

To train the model, run the main training script:

```bash
python train_model.py
```

This script will:
1. Load and preprocess the data
2. Generate data visualizations
3. Train the deep learning model
4. Evaluate model performance
5. Generate a submission file for test predictions

### Streamlit Application

To run the web application:

```bash
cd app
streamlit run app.py
```

The application provides a user-friendly interface for making depression risk predictions based on user input.

## Model Evaluation

The model is evaluated using the following metrics:
- Accuracy: Overall prediction accuracy
- Precision: Proportion of positive identifications that were actually correct
- Recall: Proportion of actual positives that were correctly identified
- F1 Score: Harmonic mean of precision and recall

Fairness evaluation is also performed to ensure the model performs consistently across different demographic groups.

## Deployment

The Streamlit application can be deployed to various platforms:

### AWS Deployment

1. Set up an EC2 instance
2. Install required dependencies
3. Configure security groups to allow traffic on port 8501
4. Run the Streamlit app with `streamlit run app.py --server.port=8501`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The dataset used for this project contains synthetic mental health survey data
- PyTorch for deep learning framework
- Streamlit for the web application framework
