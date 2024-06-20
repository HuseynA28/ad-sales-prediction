import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def train_model():
    # Load data
    data = pd.read_csv('../data/advertising.csv')
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Test MSE: {mse}')

    # Save the model
    joblib.dump(model, '../models/sales_model.pkl')

if __name__ == '__main__':
    train_model()
