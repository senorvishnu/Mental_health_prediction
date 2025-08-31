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

def test_end_to_end():
    """Test the entire prediction pipeline end-to-end"""
    print("Testing end-to-end prediction pipeline...")
    
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
    
    # Initialize preprocessor and model
    print("\nInitializing preprocessor and model...")
    preprocessor = DataPreprocessor()
    
    # Fit the preprocessor with training data
    print("Fitting preprocessor with training data...")
    X_train, y_train = preprocessor.fit_transform(train_data)
    print(f"X_train shape: {X_train.shape}")
    
    # Create a simple model for testing
    print("\nCreating a simple model for testing...")
    input_size = X_train.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
        torch.nn.Sigmoid()
    )
    
    # Test student input
    print("\nTesting student input...")
    student_data = pd.DataFrame({
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
    
    # Transform student data
    print("Transforming student data...")
    result = preprocessor.transform(student_data)
    print(f"Result type: {type(result)}")
    
    # Handle both cases: with and without target column
    if isinstance(result, tuple):
        X_student, _ = result
        print("Result is a tuple")
    else:
        X_student = result
        print("Result is not a tuple")
    
    print(f"X_student type: {type(X_student)}")
    
    # Check if it's a sparse matrix
    if sparse.issparse(X_student):
        print("X_student is a sparse matrix")
        print(f"X_student shape: {X_student.shape}")
        print(f"X_student nnz: {X_student.nnz}")
        
        # Convert sparse matrix to dense
        print("Converting sparse matrix to dense array...")
        X_student_dense = X_student.toarray()
        print(f"X_student_dense shape: {X_student_dense.shape}")
        
        # Try to use it with torch
        print("Converting to torch tensor...")
        X_student_tensor = torch.FloatTensor(X_student_dense)
    else:
        print("X_student is not a sparse matrix")
        print(f"X_student shape: {X_student.shape}")
        
        # Try to use it with torch
        print("Converting to torch tensor...")
        X_student_tensor = torch.FloatTensor(X_student)
    
    print(f"X_student_tensor shape: {X_student_tensor.shape}")
    
    # Make prediction
    print("\nMaking prediction...")
    with torch.no_grad():
        output = model(X_student_tensor)
        prediction = (output >= 0.5).float().numpy()
    
    print(f"Prediction: {prediction}")
    print(f"Probability: {output.item():.4f}")
    
    # Test professional input
    print("\nTesting professional input...")
    professional_data = pd.DataFrame({
        'Gender': ['Female'],
        'Age': [35],
        'City': ['Urban'],
        'Working Professional or Student': ['Working Professional'],
        'Profession': ['Engineer'],
        'Academic Pressure': [2],
        'Work Pressure': [5],
        'CGPA': [None],
        'Study Satisfaction': [None],
        'Job Satisfaction': [4],
        'Sleep Duration': ['4-6 hours'],
        'Dietary Habits': ['Average'],
        'Degree': ['Masters'],
        'Have you ever had suicidal thoughts ?': ['No'],
        'Work/Study Hours': [9],
        'Financial Stress': [3],
        'Family History of Mental Illness': ['No']
    })
    
    # Transform professional data
    print("Transforming professional data...")
    result = preprocessor.transform(professional_data)
    print(f"Result type: {type(result)}")
    
    # Handle both cases: with and without target column
    if isinstance(result, tuple):
        X_professional, _ = result
        print("Result is a tuple")
    else:
        X_professional = result
        print("Result is not a tuple")
    
    print(f"X_professional type: {type(X_professional)}")
    
    # Check if it's a sparse matrix
    if sparse.issparse(X_professional):
        print("X_professional is a sparse matrix")
        print(f"X_professional shape: {X_professional.shape}")
        print(f"X_professional nnz: {X_professional.nnz}")
        
        # Convert sparse matrix to dense
        print("Converting sparse matrix to dense array...")
        X_professional_dense = X_professional.toarray()
        print(f"X_professional_dense shape: {X_professional_dense.shape}")
        
        # Try to use it with torch
        print("Converting to torch tensor...")
        X_professional_tensor = torch.FloatTensor(X_professional_dense)
    else:
        print("X_professional is not a sparse matrix")
        print(f"X_professional shape: {X_professional.shape}")
        
        # Try to use it with torch
        print("Converting to torch tensor...")
        X_professional_tensor = torch.FloatTensor(X_professional)
    
    print(f"X_professional_tensor shape: {X_professional_tensor.shape}")
    
    # Make prediction
    print("\nMaking prediction...")
    with torch.no_grad():
        output = model(X_professional_tensor)
        prediction = (output >= 0.5).float().numpy()
    
    print(f"Prediction: {prediction}")
    print(f"Probability: {output.item():.4f}")
    
    print("\nTest complete!")
    return True

if __name__ == "__main__":
    success = test_end_to_end()
    if success:
        print("\nAll tests passed successfully!")
    else:
        print("\nTests failed.")