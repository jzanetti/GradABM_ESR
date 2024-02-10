import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch import tensor

num_categories = 3  # Number of categories
tau = 1.0  # Temperature parameter

num_samples = 10000  # Number of samples to generate
gumbel_samples = np.random.gumbel(loc=0, scale=1, size=(num_samples, num_categories))


softmax_samples = np.exp((gumbel_samples + np.log([[0.1, 0.8, 0.5]])) / tau)
softmax_samples /= np.sum(softmax_samples, axis=1, keepdims=True)


# x = F.gumbel_softmax(logits=tensor([0.0, 0.8, 0.5]), tau=1, hard=False, dim=1)[:, 0]

plt.figure(figsize=(8, 6))
for i in range(3):
    plt.hist(softmax_samples[:, i], bins=30, density=True, alpha=0.5, label=f"Category {i}")
plt.xlabel("Probability")
plt.ylabel("Density")
plt.title("Gumbel-Softmax Distribution")
plt.legend()
plt.savefig("test.png")
plt.close()
