import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_model():
    # Define the path to the data file
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/advertising.csv'))

    # Load the dataset
    data = pd.read_csv(data_path)

    # Define features and target
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Test MSE: {mse}')

    # Ensure the models directory exists
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Save the model
    model_path = os.path.join(models_dir, 'sales_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()
