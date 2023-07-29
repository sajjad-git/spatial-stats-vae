"""
To generate some weights
They don't add up to 1, but the point is to determine the importance
of content and style layers
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

def normal_dist_coefficients(layer):
    """
    Returns array of len (5,)
    """
    samples = np.random.normal(loc=layer, scale=1, size=100000)
    h = np.histogram(samples, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], density=True)
    return h[0]

"""
#To be used for testing
# %%

for i in range(1, 6):
    samples = np.random.normal(loc=i, scale=1, size=100000)
    plt.figure()
    plt.hist(samples, density=True, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    h = np.histogram(samples, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], density=True)
    print(h)

# %%
# tiangular dist
for i in range(1, 6):
    samples = np.random.triangular(0, i, 6, size=10000)
    plt.figure()
    h = np.histogram(samples, density=True, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    print(sum(h[0]))
    plt.hist(samples, density=True, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

# %%
"""
