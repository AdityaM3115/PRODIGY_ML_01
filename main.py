from data_loader import load_data
from model import train_model
import matplotlib.pyplot as plt


def plot_results(y_test, predictions):
    """Plots the actual vs predicted prices."""
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Prices")
    plt.show()


def main():
    # Path to the dataset
    filepath = "F://vscode//python//task1//data//house_prices.csv"

    # Load the dataset
    print("Loading data...")
    data = load_data(filepath)

    if data is None:
        print("Failed to load data. Exiting...")
        return

    # Train the model
    print("Training model...")
    model, mse, y_test, predictions = train_model(data)

    # Output results
    print(f"Model trained successfully.")
    print(f"Mean Squared Error: {mse}")
    print(f"Model Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")

    # Plot results
    print("Plotting results...")
    plot_results(y_test, predictions)


if __name__ == "__main__":
    main()
