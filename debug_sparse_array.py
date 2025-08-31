import pandas as pd
import numpy as np
import os
import sys
import torch
from scipy import sparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules
from src.data.preprocess import DataPreprocessor
from src.models.model import ModelTrainer

def debug_sparse_array_issue():
    """Debug the sparse array length ambiguous issue"""
    print("Debugging sparse array issue...")
    
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
    
    # Transform user data
    print("\nTransforming user data...")
    result = preprocessor.transform(user_data)
    print(f"Result type: {type(result)}")
    
    # Handle both cases: with and without target column
    if isinstance(result, tuple):
        X_user, _ = result
        print("Result is a tuple")
    else:
        X_user = result
        print("Result is not a tuple")
    
    print(f"X_user type: {type(X_user)}")
    
    # Check if it's a sparse matrix
    if sparse.issparse(X_user):
        print("X_user is a sparse matrix")
        print(f"X_user shape: {X_user.shape}")
        print(f"X_user nnz: {X_user.nnz}")
        
        # Convert sparse matrix to dense
        print("\nConverting sparse matrix to dense array...")
        X_user_dense = X_user.toarray()
        print(f"X_user_dense shape: {X_user_dense.shape}")
        
        # Try to use it with torch
        print("\nConverting to torch tensor...")
        X_user_tensor = torch.FloatTensor(X_user_dense)
        print(f"X_user_tensor shape: {X_user_tensor.shape}")
    else:
        print("X_user is not a sparse matrix")
        print(f"X_user shape: {X_user.shape}")
        
        # Try to use it with torch
        print("\nConverting to torch tensor...")
        X_user_tensor = torch.FloatTensor(X_user)
        print(f"X_user_tensor shape: {X_user_tensor.shape}")
    
    print("\nDebug complete!")
    return True

if __name__ == "__main__":
    success = debug_sparse_array_issue()
    if success:
        print("\nDebugging completed successfully!")
    else:
        print("\nDebugging failed.")