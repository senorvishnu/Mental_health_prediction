import pandas as pd
import numpy as np
import os
import sys
import torch
from unittest.mock import patch, MagicMock
import unittest

# Add the app directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Mock streamlit
sys.modules['streamlit'] = MagicMock()
import streamlit as st

# Import necessary modules
from src.data.preprocess import DataPreprocessor
from src.models.model import ModelTrainer

class TestMentalHealthApp(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment before each test"""
        # Create a sample training dataset
        self.train_data = pd.DataFrame({
            'Gender': ['Male', 'Female', 'Male', 'Female'],
            'Age': [25, 30, 22, 35],
            'City': ['Urban', 'Rural', 'Urban', 'Rural'],
            'Working Professional or Student': ['Student', 'Working Professional', 'Student', 'Working Professional'],
            'Profession': ['Student', 'Engineer', 'Student', 'Doctor'],
            'Academic Pressure': [3, 4, 5, 2],
            'Work Pressure': [2, 5, 3, 4],
            'CGPA': [3.5, None, 3.8, None],
            'Study Satisfaction': [4, None, 3, None],
            'Job Satisfaction': [None, 3, None, 4],
            'Sleep Duration': ['6-8 hours', '4-6 hours', '8+ hours', '<4 hours'],
            'Dietary Habits': ['Healthy', 'Average', 'Healthy', 'Poor'],
            'Degree': ['Bachelors', 'Masters', 'Bachelors', 'PhD'],
            'Have you ever had suicidal thoughts ?': ['No', 'Yes', 'No', 'No'],
            'Work/Study Hours': [6, 8, 7, 10],
            'Financial Stress': [2, 4, 3, 5],
            'Family History of Mental Illness': ['No', 'Yes', 'No', 'Yes'],
            'Depression': [0, 1, 0, 1]
        })
        
        # Initialize preprocessor and fit it with training data
        self.preprocessor = DataPreprocessor()
        self.X_train, self.y_train = self.preprocessor.fit_transform(self.train_data)
        
        # Create a mock trainer
        self.trainer = MagicMock()
        self.trainer.predict.return_value = [0]  # Default to low risk
        self.trainer.model.return_value = torch.tensor([0.2])  # Default to 20% probability
    
    def test_student_input_processing(self):
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
        
        # Process the data
        result = self.preprocessor.transform(user_data)
        # Handle both cases: with and without target column
        if isinstance(result, tuple):
            X_user, _ = result
        else:
            X_user = result
        
        # Verify shape
        self.assertEqual(X_user.shape[0], 1, "Should have 1 sample")
        self.assertEqual(X_user.shape[1], self.X_train.shape[1], "Feature dimensions should match training data")
    
    def test_professional_input_processing(self):
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
        
        # Process the data
        result = self.preprocessor.transform(user_data)
        # Handle both cases: with and without target column
        if isinstance(result, tuple):
            X_user, _ = result
        else:
            X_user = result
        
        # Verify shape
        self.assertEqual(X_user.shape[0], 1, "Should have 1 sample")
        self.assertEqual(X_user.shape[1], self.X_train.shape[1], "Feature dimensions should match training data")
    
    def test_missing_columns(self):
        """Test handling of missing columns"""
        # Create data with missing columns
        user_data = pd.DataFrame({
            'Gender': ['Male'],
            'Age': [22],
            'City': ['Urban'],
            'Working Professional or Student': ['Student'],
            'Profession': ['Student'],
            'Academic Pressure': [4],
            'Work Pressure': [3],
            'CGPA': [3.8],
            # Missing Study Satisfaction
            'Job Satisfaction': [None],
            'Sleep Duration': ['6-8 hours'],
            'Dietary Habits': ['Healthy'],
            'Degree': ['Bachelors'],
            'Have you ever had suicidal thoughts ?': ['No'],
            'Work/Study Hours': [7],
            'Financial Stress': [2],
            'Family History of Mental Illness': ['No']
        })
        
        # Process the data - should raise an error
        with self.assertRaises(Exception):
            result = self.preprocessor.transform(user_data)
    
    def test_both_satisfaction_columns(self):
        """Test with both satisfaction columns present"""
        # Create data with both satisfaction columns
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
            'Job Satisfaction': [None],  # None for student
            'Sleep Duration': ['6-8 hours'],
            'Dietary Habits': ['Healthy'],
            'Degree': ['Bachelors'],
            'Have you ever had suicidal thoughts ?': ['No'],
            'Work/Study Hours': [7],
            'Financial Stress': [2],
            'Family History of Mental Illness': ['No']
        })
        
        # Process the data
        result = self.preprocessor.transform(user_data)
        # Handle both cases: with and without target column
        if isinstance(result, tuple):
            X_user, _ = result
        else:
            X_user = result
        
        # Verify shape
        self.assertEqual(X_user.shape[0], 1, "Should have 1 sample")
        self.assertEqual(X_user.shape[1], self.X_train.shape[1], "Feature dimensions should match training data")
    
    def test_edge_case_values(self):
        """Test with edge case values"""
        # Create data with edge case values
        user_data = pd.DataFrame({
            'Gender': ['Other'],  # Non-standard gender
            'Age': [99],  # Very old age
            'City': ['Unknown'],  # Unknown city
            'Working Professional or Student': ['Retired'],  # Non-standard occupation
            'Profession': ['Other'],  # Non-standard profession
            'Academic Pressure': [10],  # Maximum value
            'Work Pressure': [10],  # Maximum value
            'CGPA': [4.0],  # Maximum value
            'Study Satisfaction': [5],  # Maximum value
            'Job Satisfaction': [5],  # Maximum value
            'Sleep Duration': ['<4 hours'],  # Minimum sleep
            'Dietary Habits': ['Poor'],  # Poor diet
            'Degree': ['Other'],  # Non-standard degree
            'Have you ever had suicidal thoughts ?': ['Yes'],  # Yes to suicidal thoughts
            'Work/Study Hours': [24],  # Maximum hours
            'Financial Stress': [10],  # Maximum stress
            'Family History of Mental Illness': ['Yes']  # Yes to family history
        })
        
        # Process the data - should handle edge cases
        try:
            result = self.preprocessor.transform(user_data)
            # Handle both cases: with and without target column
            if isinstance(result, tuple):
                X_user, _ = result
            else:
                X_user = result
            
            # Verify shape
            self.assertEqual(X_user.shape[0], 1, "Should have 1 sample")
            self.assertEqual(X_user.shape[1], self.X_train.shape[1], "Feature dimensions should match training data")
        except Exception as e:
            self.fail(f"Failed to process edge case values: {e}")

if __name__ == "__main__":
    unittest.main()