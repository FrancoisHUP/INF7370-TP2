import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Generate input values
x = np.linspace(-10, 10, 500)

# Compute ReLU activation using TensorFlow
relu = tf.keras.activations.relu(x)

# Compute LeakyReLU activation using negative_slope instead of alpha
leaky_relu = tf.keras.layers.LeakyReLU(negative_slope=0.1)(x)

# Create the plots
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(x, relu)
plt.title("ReLU Activation")
plt.xlabel("Input")
plt.ylabel("Activation")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x, leaky_relu)
plt.title("Leaky ReLU Activation (negative_slope=0.1)")
plt.xlabel("Input")
plt.ylabel("Activation")
plt.grid(True)

plt.tight_layout()
# Save the plot as a PNG file instead of showing it interactively
plt.savefig("assets/relu_comparison.png")
print("Plot saved to 'assets/relu_comparison.png'")
