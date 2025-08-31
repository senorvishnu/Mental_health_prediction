import os
import shutil
import sys

def setup_data_directories():
    """Set up data directories and copy data files to the appropriate locations."""
    print("Setting up data directories...")
    
    # Create data directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Source data files
    source_files = {
        'train.csv': 'E:/Python/Mental_Health_Survey_Project/train.csv',
        'test.csv': 'E:/Python/Mental_Health_Survey_Project/test.csv',
        'sample_submission.csv': 'E:/Python/Mental_Health_Survey_Project/sample_submission.csv'
    }
    
    # Copy files to data/raw directory
    for filename, source_path in source_files.items():
        if os.path.exists(source_path):
            destination_path = os.path.join('data/raw', filename)
            try:
                shutil.copy2(source_path, destination_path)
                print(f"Copied {filename} to {destination_path}")
            except Exception as e:
                print(f"Error copying {filename}: {str(e)}")
        else:
            print(f"Warning: Source file {source_path} not found")
    
    print("Data setup complete!")

if __name__ == "__main__":
    setup_data_directories()