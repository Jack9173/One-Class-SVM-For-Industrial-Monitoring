import numpy as np
from sklearn import svm

# Let's assume we have a dataset of temperature readings from a industrial process
# We'll use this dataset to train our anomaly detection model
temperature_data = np.array([
    [20.5, 30.2, 25.1, 28.5, 22.1],  # normal data
    [20.1, 30.5, 25.3, 28.2, 22.5],  # normal data
    [20.8, 31.2, 26.1, 29.5, 23.8],  # normal data
    [21.5, 32.5, 27.5, 31.2, 25.5],  # anomaly!
    [20.2, 30.1, 25.2, 28.1, 22.2],  # normal data
    [20.9, 31.5, 26.5, 30.2, 24.5],  # anomaly!
])

# Create a One-Class SVM model with a radial basis function (RBF) kernel
model = svm.OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)

# Train the model on the temperature data
model.fit(temperature_data)

# Define a function to detect anomalies in real-time data streams
def detect_anomaly(new_data):
    """
    Take in a new data point and return True if it's an anomaly, False otherwise
    """
    # Predict the anomaly score for the new data
    anomaly_score = model.decision_function([new_data])

    # If the anomaly score is negative, it's an anomaly!
    if anomaly_score < 0:
        print("Anomaly detected!")
        return True
    else:
        print("Normal data point.")
        return False

# Example usage:
new_data = [21.8, 32.8, 28.2, 32.5, 26.8]  # new data point to be classified
is_anomaly = detect_anomaly(new_data)
print("Is anomaly:", is_anomaly)

new_data = [20.3, 30.3, 25.5, 28.8, 22.8]  # new data point to be classified
is_anomaly = detect_anomaly(new_data)
print("Is anomaly:", is_anomaly)
