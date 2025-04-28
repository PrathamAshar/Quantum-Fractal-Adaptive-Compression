import numpy as np
from quantum_compression import (
    variational_quantum_compression,
    simulate_quantum_circuit_with_noise,
    quantum_autoencoder_compression,
    visualize_quantum_results
)
from classical_model import build_and_train_model, evaluate_model
from neural_network import build_and_train_neural_network, predict_compressibility
from quantum_compression import create_variational_circuit

def main():
    # Generate sample data
    data = [0, 1, 1, 0, 1]

    # Variational Quantum Compression
    optimal_params = variational_quantum_compression(data)
    print(f"Optimal parameters for quantum compression: {optimal_params}")

    # Create quantum circuit
    qc = create_variational_circuit(data)

    # 🛠 Bind optimal parameters into the circuit
    param_dict = dict(zip(qc.parameters, optimal_params))
    qc_bound = qc.assign_parameters(param_dict)
    qc_bound.measure_all()

    # Simulate bound circuit
    counts = simulate_quantum_circuit_with_noise(qc_bound)
    print("Quantum Compression Results with Noise Mitigation:", counts)
    visualize_quantum_results(counts)


    # Neural Network for Predicting Compressibility
    #nn_model = build_and_train_neural_network()
    #compressibility_prediction = predict_compressibility(nn_model, data)
    #print(f"Predicted compressibility: {compressibility_prediction:.2f}")

    # Quantum Autoencoder Compression
    qc_autoencoder = quantum_autoencoder_compression(data)
    autoencoder_counts = simulate_quantum_circuit_with_noise(qc_autoencoder)
    print("Quantum Autoencoder Compression Results:", autoencoder_counts)
    visualize_quantum_results(autoencoder_counts)

if __name__ == "__main__":
    main()

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def build_and_train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    param_grid = {'C': [0.1, 1, 10]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy