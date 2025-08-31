import os
import pandas as pd
import numpy as np
import yaml
import torch
import json
from sklearn.model_selection import train_test_split

# Import project modules
from src.data.preprocess import DataPreprocessor
from src.models.model import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.visualization.visualize import DataVisualizer

def main():
    """Main function to orchestrate the workflow."""
    print("Starting Depression Risk Prediction Model Training")
    
    # Load configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Create directories if they don't exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs(config['output']['visualizations_dir'], exist_ok=True)
    
    # Initialize components
    preprocessor = DataPreprocessor()
    evaluator = ModelEvaluator()
    visualizer = DataVisualizer(output_dir=config['output']['visualizations_dir'])
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    X_train, y_train, X_test, test_ids = preprocessor.load_and_preprocess(
        train_path=config['data']['train_path'],
        test_path=config['data']['test_path']
    )
    
    if X_train is None or y_train is None:
        print("Error: Training data not found or could not be processed.")
        return
    
    # Update input size in config based on preprocessed data
    input_size = X_train.shape[1]
    config['model']['input_size'] = input_size
    
    # Save updated config
    with open('config.yaml', 'w') as file:
        yaml.dump(config, file)
    
    # Visualize data
    print("\nGenerating data visualizations...")
    try:
        # Load raw data for visualization
        train_data = pd.read_csv(config['data']['train_path'])
        
        # Plot feature distributions
        visualizer.plot_feature_distributions(
            train_data, 
            categorical_features=config['features']['categorical'],
            numerical_features=config['features']['numerical']
        )
        
        # Plot correlation matrix
        visualizer.plot_correlation_matrix(
            train_data,
            features=config['features']['numerical'] + [config['features']['target']]
        )
        
        print(f"Visualizations saved to {config['output']['visualizations_dir']}")
    except Exception as e:
        print(f"Warning: Could not generate all visualizations. Error: {str(e)}")
    
    # Initialize and train model
    print("\nInitializing model...")
    trainer = ModelTrainer(
        input_size=input_size,
        hidden_sizes=config['model']['hidden_layers'],
        dropout_rate=config['model']['dropout_rate'],
        learning_rate=config['model']['learning_rate'],
        batch_size=config['model']['batch_size']
    )
    
    print("\nPreparing data loaders...")
    train_loader, val_loader = trainer.prepare_data_loaders(
        X_train, y_train, val_split=config['model']['validation_split']
    )
    
    print("\nTraining model...")
    history = trainer.train(
        train_loader, 
        val_loader, 
        epochs=config['model']['epochs'],
        early_stopping_patience=config['model']['early_stopping_patience']
    )
    
    # Visualize training history
    visualizer.plot_training_history(history)
    
    # Make predictions on test data if available
    if X_test is not None and test_ids is not None:
        print("\nMaking predictions on test data...")
        
        predictions = trainer.predict(X_test)
        
        # Create submission file
        submission = pd.DataFrame({
            'id': test_ids,
            'Depression': predictions.astype(int)
        })
        
        submission_path = config['data']['submission_path']
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission.to_csv(submission_path, index=False)
        print(f"Submission file saved to {submission_path}")
    
    # Evaluate model on validation data
    print("\nEvaluating model...")
    # Split training data for evaluation
    X_train_subset, X_val, y_train_subset, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Make predictions on validation data
    val_predictions = trainer.predict(X_val)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_val, val_predictions)
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        y_val, 
        val_predictions,
        save_path=os.path.join(config['output']['visualizations_dir'], 'confusion_matrix.png')
    )
    
    # Evaluate fairness if possible
    try:
        # Load raw data again for fairness evaluation
        train_data = pd.read_csv(config['data']['train_path'])
        
        # Create validation data with original features for fairness evaluation
        _, val_indices = train_test_split(
            range(len(train_data)), test_size=0.2, random_state=42
        )
        val_data = train_data.iloc[val_indices].copy()
        val_data['predicted'] = val_predictions
        
        # Evaluate fairness
        fairness_metrics = evaluator.evaluate_fairness(
            val_data,
            val_predictions,
            sensitive_features=config['evaluation']['sensitive_features']
        )
        
        # Plot fairness metrics
        for metric in config['evaluation']['metrics']:
            if metric in metrics:
                evaluator.plot_fairness_metrics(
                    fairness_metrics,
                    metric=metric,
                    save_path=os.path.join(config['output']['visualizations_dir'], f'fairness_{metric}.png')
                )
        
        # Save evaluation results
        results = {
            'metrics': metrics,
            'fairness': fairness_metrics
        }
        
        with open(config['output']['results_file'], 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nEvaluation results saved to {config['output']['results_file']}")
    except Exception as e:
        print(f"Warning: Could not complete fairness evaluation. Error: {str(e)}")
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()