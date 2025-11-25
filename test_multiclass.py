import numpy as np
from numpy_nn.network import NeuralNetwork
from numpy_nn.layer import Dense
from numpy_nn.activation import Tanh, Softmax
from numpy_nn.loss import CategoricalCrossEntropy
from numpy_nn.initializers import Xavier

# Set seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("Multi-Class Classification with Softmax")
print("=" * 70)

# Create a simple 3-class classification problem
# Class 0: points near (0, 0)
# Class 1: points near (1, 0)
# Class 2: points near (0.5, 1)

X = np.array([
    [0.1, 0.1],   # Class 0
    [0.0, 0.2],   # Class 0
    [0.2, 0.0],   # Class 0
    [0.9, 0.1],   # Class 1
    [1.0, 0.0],   # Class 1
    [0.8, 0.2],   # Class 1
    [0.4, 0.9],   # Class 2
    [0.5, 1.0],   # Class 2
    [0.6, 0.8],   # Class 2
])

# One-hot encoded labels
y = np.array([
    [1, 0, 0],  # Class 0
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],  # Class 1
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],  # Class 2
    [0, 0, 1],
    [0, 0, 1],
])

print(f"\nDataset:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape} (one-hot encoded)")
print(f"Number of classes: 3")
print(f"Samples per class: 3")

# Build network for 3-class classification
print("\n" + "=" * 70)
print("Building Network: 2 → 5 (Tanh) → 3 (Softmax)")
print("=" * 70)

network = NeuralNetwork()
network.add_layer(Dense(2, 5, Tanh(), weight_initializer=Xavier()))
network.add_layer(Dense(5, 3, Softmax(), weight_initializer=Xavier()))

# Train with categorical cross-entropy
print("\nTraining...")
loss_fn = CategoricalCrossEntropy()
network.train(X, y, loss_fn, epochs=2000, learning_rate=0.1)

# Test predictions
print("\n" + "=" * 70)
print("Final Predictions")
print("=" * 70)

predictions = network.forward(X)

print(f"\n{'Input':<15} {'True Class':<12} {'Predicted':<30} {'Confidence':<12}")
print("-" * 70)

for i in range(len(X)):
    true_class = np.argmax(y[i])
    pred_probs = predictions[i]
    pred_class = np.argmax(pred_probs)
    confidence = pred_probs[pred_class]
    
    # Format probabilities
    probs_str = "[" + ", ".join([f"{p:.3f}" for p in pred_probs]) + "]"
    
    match = "✓" if true_class == pred_class else "✗"
    
    print(f"{str(X[i]):<15} Class {true_class:<6} {probs_str:<30} {confidence:.3f} {match}")

# Calculate accuracy
correct = np.sum(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))
accuracy = correct / len(y) * 100
print(f"\n{'=' * 70}")
print(f"Accuracy: {correct}/{len(y)} = {accuracy:.1f}%")

# Test on new points
print("\n" + "=" * 70)
print("Testing on New Points")
print("=" * 70)

test_points = np.array([
    [0.15, 0.15],  # Should be Class 0
    [0.85, 0.15],  # Should be Class 1
    [0.50, 0.85],  # Should be Class 2
])

test_predictions = network.forward(test_points)

print(f"\n{'Input':<15} {'Predicted Class':<18} {'Probabilities':<30}")
print("-" * 70)

for i in range(len(test_points)):
    pred_probs = test_predictions[i]
    pred_class = np.argmax(pred_probs)
    probs_str = "[" + ", ".join([f"{p:.3f}" for p in pred_probs]) + "]"
    
    print(f"{str(test_points[i]):<15} Class {pred_class:<12} {probs_str}")

print("\n" + "=" * 70)
print("Softmax Properties:")
print("=" * 70)
print("✓ All predictions sum to 1.0 (probability distribution)")
print("✓ Each output is between 0 and 1")
print("✓ Highest value = predicted class")

# Verify softmax properties
sample_pred = predictions[0]
print(f"\nExample prediction: {sample_pred}")
print(f"Sum: {np.sum(sample_pred):.6f} (should be 1.0)")
print(f"All positive: {np.all(sample_pred >= 0)}")
print(f"All <= 1: {np.all(sample_pred <= 1)}")
