import pandas as pd
import numpy as np
import os
import sys
import torch
from unittest.mock import patch, MagicMock

# Add the app directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Mock streamlit
sys.modules['streamlit'] = MagicMock()
import streamlit as st

# Now we can import from app
from src.data.preprocess import DataPreprocessor
from src.models.model import ModelTrainer

# Create a mock for the load_model function
def mock_load_model():
    trainer = MagicMock()
    trainer.predict.return_value = [0]  # Low risk of depression
    trainer.model.return_value = torch.tensor([0.2])  # 20% probability
    
    preprocessor = DataPreprocessor()
    # Create sample data to fit the preprocessor
    train_data = pd.DataFrame({
        'Gender': ['Male', 'Female'],
        'Age': [25, 30],
        'City': ['Urban', 'Rural'],
        'Working Professional or Student': ['Student', 'Working Professional'],
        'Profession': ['Student', 'Engineer'],
        'Academic Pressure': [3, 4],
        'Work Pressure': [2, 5],
        'CGPA': [3.5, None],
        'Study Satisfaction': [4, None],
        'Job Satisfaction': [None, 3],
        'Sleep Duration': ['6-8 hours', '4-6 hours'],
        'Dietary Habits': ['Healthy', 'Average'],
        'Degree': ['Bachelors', 'Masters'],
        'Have you ever had suicidal thoughts ?': ['No', 'Yes'],
        'Work/Study Hours': [6, 8],
        'Financial Stress': [2, 4],
        'Family History of Mental Illness': ['No', 'Yes'],
        'Depression': [0, 1]
    })
    preprocessor.fit_transform(train_data)
    
    return trainer, preprocessor

def test_student_input_processing():
    """Test processing of student input data"""
    # Create student input data
    user_data = pd.DataFrame({
        'Gender': ['Male'],
        'Age': [22],
        'City': ['Urban'],
        'Working Professional or Student': ['Student'],
        'Profession': ['Student'],
        'Academic Pressure': [4],
        'Work Pressure': [3],
        'CGPA': [3.8],
        'Study Satisfaction': [3],
        'Job Satisfaction': [None],
        'Sleep Duration': ['6-8 hours'],
        'Dietary Habits': ['Healthy'],
        'Degree': ['Bachelors'],
        'Have you ever had suicidal thoughts ?': ['No'],
        'Work/Study Hours': [7],
        'Financial Stress': [2],
        'Family History of Mental Illness': ['No']
    })
    
    # Get trainer and preprocessor
    trainer, preprocessor = mock_load_model()
    
    # Process the data
    try:
        result = preprocessor.transform(user_data)
        # Handle both cases: with and without target column
        if isinstance(result, tuple):
            X_user, _ = result
        else:
            X_user = result
            
        # Make prediction
        prediction = trainer.predict(X_user)
        probability = trainer.model(torch.FloatTensor(X_user)).item()
        
        print(f"Student input processed successfully. Prediction: {prediction[0]}, Probability: {probability:.2f}")
        return True
    except Exception as e:
        print(f"Error processing student input: {e}")
        return False

def test_professional_input_processing():
    """Test processing of professional input data"""
    # Create professional input data
    user_data = pd.DataFrame({
        'Gender': ['Female'],
        'Age': [35],
        'City': ['Urban'],
        'Working Professional or Student': ['Working Professional'],
        'Profession': ['Engineer'],
        'Academic Pressure': [None],
        'Work Pressure': [4],
        'CGPA': [None],
        'Study Satisfaction': [None],
        'Job Satisfaction': [3],
        'Sleep Duration': ['4-6 hours'],
        'Dietary Habits': ['Average'],
        'Degree': ['Masters'],
        'Have you ever had suicidal thoughts ?': ['No'],
        'Work/Study Hours': [9],
        'Financial Stress': [3],
        'Family History of Mental Illness': ['No']
    })
    
    # Get trainer and preprocessor
    trainer, preprocessor = mock_load_model()
    
    # Process the data
    try:
        result = preprocessor.transform(user_data)
        # Handle both cases: with and without target column
        if isinstance(result, tuple):
            X_user, _ = result
        else:
            X_user = result
            
        # Make prediction
        prediction = trainer.predict(X_user)
        probability = trainer.model(torch.FloatTensor(X_user)).item()
        
        print(f"Professional input processed successfully. Prediction: {prediction[0]}, Probability: {probability:.2f}")
        return True
    except Exception as e:
        print(f"Error processing professional input: {e}")
        return False

if __name__ == "__main__":
    print("Testing student input processing:")
    student_success = test_student_input_processing()
    
    print("\nTesting professional input processing:")
    professional_success = test_professional_input_processing()
    
    if student_success and professional_success:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed.")