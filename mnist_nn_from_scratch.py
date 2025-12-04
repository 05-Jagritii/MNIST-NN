#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MNIST Neural Network (FROM SCRATCH using NumPy)

- Loads MNIST digits (0–9) using Keras
- Normalizes data
- One-hot encodes labels
- Implements:
    * Dense layers
    * ReLU / Sigmoid activation
    * Softmax
    * Cross-entropy loss
    * Forward + Backpropagation
    * Mini-batch SGD
- Trains & evaluates on MNIST
- Prints accuracy, confusion matrix, and classification report

Run:
    py mnist_nn_from_scratch.py
"""

import numpy as np
from typing import Dict, Tuple

from keras.datasets import mnist

from sklearn.metrics import confusion_matrix, classification_report


# ============================================================
# Utility functions
# ============================================================

def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels (N,) to one-hot encoded (N, num_classes)."""
    return np.eye(num_classes)[y]


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    z = z - np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_deriv(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


# ============================================================
# Neural Network Class
# ============================================================

class NeuralNetwork:
    """
    Simple fully-connected neural network implemented from scratch.

    Parameters
    ----------
    input_dim : int
        Number of input features (MNIST: 28*28=784).
    hidden_layers : list[int]
        Sizes of hidden layers.
    output_dim : int
        Number of output classes (MNIST: 10).
    activation : {'relu','sigmoid'}
        Activation function for hidden layers.
    lr : float
        Learning rate for SGD.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_layers = [128, 64],
        output_dim: int = 10,
        activation: str = "relu",
        lr: float = 0.01,
        seed: int = 42,
    ):
        self.layers = [input_dim] + list(hidden_layers) + [output_dim]
        self.lr = lr
        np.random.seed(seed)

        # Choose activation
        if activation == "relu":
            self.activation = relu
            self.activation_deriv = relu_deriv
        elif activation == "sigmoid":
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        else:
            raise ValueError("Unsupported activation: " + activation)

        # Initialize parameters: small random weights, zero biases
        self.params: Dict[str, np.ndarray] = {}
        for i in range(len(self.layers) - 1):
            self.params[f"W{i+1}"] = np.random.randn(self.layers[i], self.layers[i+1]) * 0.01
            self.params[f"b{i+1}"] = np.zeros((1, self.layers[i+1]))

    # ------------------ Forward pass ------------------

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Forward pass through all layers.

        Returns:
            probs: softmax outputs
            cache: intermediate values for backprop
        """
        cache: Dict[str, np.ndarray] = {}
        A = X
        cache["A0"] = A

        L = len(self.layers) - 1  # number of layers excluding input

        # Hidden layers
        for i in range(1, L):
            Z = A @ self.params[f"W{i}"] + self.params[f"b{i}"]
            A = self.activation(Z)
            cache[f"Z{i}"] = Z
            cache[f"A{i}"] = A

        # Output layer (softmax)
        ZL = A @ self.params[f"W{L}"] + self.params[f"b{L}"]
        AL = softmax(ZL)
        cache[f"Z{L}"] = ZL
        cache[f"A{L}"] = AL

        return AL, cache

    # ------------------ Backward pass ------------------

    def backward(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cache: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Backpropagation to compute gradients of weights and biases.
        y_true, y_pred are shape (N, num_classes).
        """
        grads: Dict[str, np.ndarray] = {}
        m = y_true.shape[0]
        L = len(self.layers) - 1

        # Output layer gradient
        dZ = y_pred - y_true  # (N, C)
        grads[f"W{L}"] = cache[f"A{L-1}"].T @ dZ / m
        grads[f"b{L}"] = np.sum(dZ, axis=0, keepdims=True) / m

        dA_prev = dZ @ self.params[f"W{L}"].T

        # Hidden layers (from last hidden to first)
        for i in range(L-1, 0, -1):
            A = cache[f"A{i}"]
            A_prev = cache[f"A{i-1}"]
            dZ = dA_prev * self.activation_deriv(A)  # derivative wrt pre-activation
            grads[f"W{i}"] = A_prev.T @ dZ / m
            grads[f"b{i}"] = np.sum(dZ, axis=0, keepdims=True) / m
            dA_prev = dZ @ self.params[f"W{i}"].T

        return grads

    # ------------------ Parameter update ------------------

    def update(self, grads: Dict[str, np.ndarray]) -> None:
        for key in grads:
            self.params[key] -= self.lr * grads[key]

    # ------------------ Training ------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64,
    ) -> None:
        """
        Train using mini-batch gradient descent.
        y_train must be one-hot encoded.
        """
        n = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle
            idx = np.random.permutation(n)
            X_train = X_train[idx]
            y_train = y_train[idx]

            # Mini-batch training
            for start in range(0, n, batch_size):
                end = start + batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                y_pred, cache = self.forward(X_batch)
                grads = self.backward(y_batch, y_pred, cache)
                self.update(grads)

            # Epoch metrics
            y_pred_full, _ = self.forward(X_train)
            loss = -np.mean(np.sum(y_train * np.log(y_pred_full + 1e-9), axis=1))
            acc = np.mean(
                np.argmax(y_pred_full, axis=1) == np.argmax(y_train, axis=1)
            )

            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

        print("Training complete.")

    # ------------------ Prediction ------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)


# ============================================================
# MNIST loading (Keras)
# ============================================================

def load_mnist():
    """
    Load MNIST using Keras, flatten to 784-dim vectors, scale to [0,1].

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    print("Loading MNIST from Keras...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Flatten 28x28 -> 784
    X_train = X_train.reshape(-1, 28 * 28).astype("float32")
    X_test = X_test.reshape(-1, 28 * 28).astype("float32")

    # Scale to [0,1]
    X_train /= 255.0
    X_test /= 255.0

    return X_train, y_train, X_test, y_test


# ============================================================
# Main script
# ============================================================

def main():
    # 1. Load data
    X_train, y_train, X_test, y_test = load_mnist()

    # 2. One-hot encode labels for training
    y_train_oh = one_hot(y_train, num_classes=10)

    # 3. Initialize model
    model = NeuralNetwork(
        input_dim=784,
        hidden_layers=[128, 64],
        output_dim=10,
        activation="relu",
        lr=0.01,
    )
    print("\nTraining Neural Network...\n")

    # 4. Train
    model.fit(X_train, y_train_oh, epochs=10, batch_size=64)

    # 5. Evaluate
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test)

    acc = np.mean(y_pred == y_test)
    print(f"\nTest Accuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
