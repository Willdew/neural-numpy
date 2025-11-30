import numpy as np
from neural_numpy.network import NeuralNetwork
from neural_numpy.layer import Dense
from neural_numpy.activation import Tanh, ReLU
from neural_numpy.loss import MSE
from neural_numpy.initializers import RandomNormal, Xavier, He

# Set seed for reproducibility
np.random.seed(42)

# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

print("=" * 70)
print("Comparing Weight Initialization Strategies on XOR Problem")
print("=" * 70)

# Test different initializers
initializers = {
    "Random Normal (0.1)": RandomNormal(std=0.1),
    "Xavier/Glorot": Xavier(),
    "He": He()
}

results = {}

for name, initializer in initializers.items():
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print('=' * 70)
    
    # Reset random seed for fair comparison
    np.random.seed(42)
    
    # Build network with this initializer
    network = NeuralNetwork()
    network.add_layer(Dense(2, 3, Tanh(), weight_initializer=initializer))
    network.add_layer(Dense(3, 1, Tanh(), weight_initializer=initializer))
    
    # Train
    loss_fn = MSE()
    print("\nTraining for 1000 epochs...")
    
    # Track initial and final loss
    initial_output = network.forward(X)
    initial_loss = loss_fn.forward(initial_output, y)
    
    network.train(X, y, loss_fn, epochs=1000, learning_rate=0.1)
    
    # Final predictions
    final_output = network.forward(X)
    final_loss = loss_fn.forward(final_output, y)
    
    results[name] = {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'predictions': final_output
    }
    
    print(f"\nInitial Loss: {initial_loss:.6f}")
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Improvement: {initial_loss - final_loss:.6f}")
    
    print("\nFinal Predictions:")
    for i, (input_data, target) in enumerate(zip(X, y)):
        pred = final_output[i, 0]
        print(f"  {input_data} → {pred:.4f} (target: {target[0]})")


# Summary comparison
print("\n\n" + "=" * 70)
print("SUMMARY COMPARISON")
print("=" * 70)
print(f"{'Initializer':<25} {'Initial Loss':>15} {'Final Loss':>15} {'Improvement':>15}")
print("-" * 70)

for name, result in results.items():
    improvement = result['initial_loss'] - result['final_loss']
    print(f"{name:<25} {result['initial_loss']:>15.6f} {result['final_loss']:>15.6f} {improvement:>15.6f}")

# Find best
best_name = min(results.items(), key=lambda x: x[1]['final_loss'])[0]
print(f"\n✓ Best: {best_name}")


# Test He initialization with ReLU
print("\n\n" + "=" * 70)
print("Testing He Initialization with ReLU (its intended use)")
print("=" * 70)

np.random.seed(42)
network_relu = NeuralNetwork()
network_relu.add_layer(Dense(2, 3, ReLU(), weight_initializer=He()))
network_relu.add_layer(Dense(3, 1, Tanh(), weight_initializer=Xavier()))  # Output can use Tanh

print("\nTraining...")
network_relu.train(X, y, MSE(), epochs=1000, learning_rate=0.1)

print("\nFinal Predictions:")
output_relu = network_relu.forward(X)
for i, (input_data, target) in enumerate(zip(X, y)):
    pred = output_relu[i, 0]
    print(f"  {input_data} → {pred:.4f} (target: {target[0]})")
