import torch
import numpy as np

# Example one-hot vector
one_hot_vector = torch.tensor([0., 0., 1., 0., 0.])

# Standard deviation for the Gaussian distribution
std_dev = 1.0



def to_gaussian(one_hot_vector, std_dev: float = 1.0):
    # Find the index of 1.0 in the vector
    mean_idx = torch.argmax(one_hot_vector).item()

    # Create a tensor for the Gaussian distribution
    indices = torch.arange(one_hot_vector.size(0), dtype=torch.float32)
    gaussian_distribution = torch.exp(-0.5 * ((indices - mean_idx) / std_dev) ** 2)

    # Normalize the distribution
    gaussian_distribution /= gaussian_distribution.sum()

    return gaussian_distribution


if __name__ == '__main__':

    # Convert to Gaussian distribution
    gaussian_vector = to_gaussian(one_hot_vector, std_dev)
    print(gaussian_vector)