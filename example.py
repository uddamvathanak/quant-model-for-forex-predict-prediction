from forex_predictor import ForexPredictor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_predictions(y_test, predictions, title="Actual vs Predicted Returns"):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual Returns', alpha=0.7)
    plt.plot(y_test.index, predictions, label='Predicted Returns', alpha=0.7)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('predictions_plot.png')
    plt.close()

def main():
    # Initialize the predictor for EUR/USD
    predictor = ForexPredictor(currency_pair="EURUSD=X", prediction_horizon=1)
    
    # Fetch historical data
    print("Fetching historical data...")
    data = predictor.fetch_data(start_date="2020-01-01")
    
    # Create features
    print("Creating technical indicators...")
    data_with_features = predictor.create_features(data)
    
    # Prepare data for training
    print("Preparing data for training...")
    X_train, X_test, y_train, y_test = predictor.prepare_data(data_with_features)
    
    # Train the model with hyperparameter optimization
    print("Training model with hyperparameter optimization...")
    predictor.train(X_train, y_train, optimize=True, n_trials=50)
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict(X_test)
    
    # Evaluate the model
    metrics = predictor.evaluate(X_test, y_test)
    print("\nModel Performance Metrics:")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"R2 Score: {metrics['R2']:.6f}")
    
    # Plot actual vs predicted returns
    plot_predictions(y_test, predictions)
    
    # Save the model
    print("\nSaving the model...")
    predictor.save_model('forex_predictor_model.joblib')
    
    print("\nModel training and evaluation completed!")
    print("Check 'predictions_plot.png' for the visualization of actual vs predicted returns.")

if __name__ == "__main__":
    main() 