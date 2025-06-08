from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train_model(data):
    try:
        # Select features and target
        X = data[['SquareFootage', 'Bedrooms', 'Bathrooms']]
        y = data['Price']

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions and calculate MSE
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        return model, mse, y_test, predictions
    except KeyError as e:
        print(f"Missing column in dataset: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return None, None, None, None
