import pandas as pd
import numpy as np
import os
import sys
from src.data.preprocess import DataPreprocessor

def test_preprocessor_transform_with_target():
    """Test transform method when target column exists"""
    # Create a sample dataframe with target column
    data = pd.DataFrame({
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
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Test fit_transform
    X_train, y_train = preprocessor.fit_transform(data)
    print(f"fit_transform with target: X shape = {X_train.shape}, y shape = {y_train.shape}")
    
    # Test transform
    X_test, y_test = preprocessor.transform(data)
    print(f"transform with target: X shape = {X_test.shape}, y shape = {y_test.shape}")

def test_preprocessor_transform_without_target():
    """Test transform method when target column doesn't exist"""
    # Create a sample dataframe without target column
    data = pd.DataFrame({
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
        'Family History of Mental Illness': ['No', 'Yes']
    })
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # First fit the preprocessor with data that has target
    train_data = data.copy()
    train_data['Depression'] = [0, 1]
    preprocessor.fit_transform(train_data)
    
    # Test transform without target
    try:
        result = preprocessor.transform(data)
        if isinstance(result, tuple):
            X_test, y_test = result
            print(f"transform without target returned tuple: X shape = {X_test.shape}, y shape = {y_test.shape}")
        else:
            print(f"transform without target returned single value: shape = {result.shape}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing preprocessor with target column:")
    test_preprocessor_transform_with_target()
    print("\nTesting preprocessor without target column:")
    test_preprocessor_transform_without_target()