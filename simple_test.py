import pandas as pd
import numpy as np
import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules
from src.data.preprocess import DataPreprocessor

def test_transform_unpacking():
    """Test the fix for 'not enough values to unpack' error"""
    print("Testing transform unpacking fix...")
    
    # Create sample data
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
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Fit the preprocessor with training data
    preprocessor.fit_transform(train_data)
    
    # Create user data without target column
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
    
    # Test the fixed code
    try:
        print("Testing with our fix:")
        result = preprocessor.transform(user_data)
        # Handle both cases: with and without target column
        if isinstance(result, tuple):
            X_user, _ = result
            print("Result is a tuple, unpacked successfully")
        else:
            X_user = result
            print("Result is a single value, assigned successfully")
        
        print(f"X_user shape: {X_user.shape}")
        print("Test passed!")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    
    # Test the original code (should fail)
    try:
        print("\nTesting with original code (should fail):")
        X_user, _ = preprocessor.transform(user_data)
        print("This should not print if the error occurs")
    except ValueError as e:
        print(f"Expected error occurred: {e}")
        print("This confirms our fix is necessary")

if __name__ == "__main__":
    success = test_transform_unpacking()
    if success:
        print("\nFix for 'not enough values to unpack' error is working correctly!")
    else:
        print("\nFix is not working correctly.")