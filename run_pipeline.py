#!/usr/bin/env python

import os
import subprocess
import sys
import time

def run_command(command, description):
    """Run a command and print its output."""
    print(f"\n{'-' * 80}\n{description}\n{'-' * 80}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        print(f"Error: Command failed with return code {process.returncode}")
        return False
    return True

def main():
    """Run the complete pipeline."""
    print("Starting Depression Risk Prediction Pipeline")
    start_time = time.time()
    
    # Setup data directories and copy files
    if not run_command("python setup_data.py", "Setting up data directories"):
        return
    
    # Train the model
    if not run_command("python train_model.py", "Training and evaluating model"):
        return
    
    # Run the Streamlit app
    print("\nAll steps completed successfully!")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
    
    print("\nTo run the Streamlit app, use the following command:")
    print("cd app && streamlit run app.py")

if __name__ == "__main__":
    main()