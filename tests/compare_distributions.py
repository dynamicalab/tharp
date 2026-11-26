import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt

from mdl_hyperbolic.generators import sample_quasiuniform_radii_2d, sample_quasiuniform_radius


key = jax.random.PRNGKey(0)

num_dim = 2
alpha = 2.
r = 5.
sample_its = sample_quasiuniform_radii_2d(r, key, alpha=alpha, num_samples=5000) # Valid sample only when num_dim=2
sample_rs = sample_quasiuniform_radius(num_dim, r, key, alpha=alpha, num_samples=5000)

plt.hist(sample_its, density=True, alpha=0.5, label="Inverse transform", bins=20)
plt.hist(sample_rs, density=True, alpha=0.5, label="Rejection sampling", bins=20)
pdf = lambda x: (num_dim-1)*alpha*(jnp.exp((num_dim-1)*alpha*x)-1)/(jnp.exp((num_dim-1)*alpha*r)-1)
xvalues = jnp.linspace(0, r, 100)
plt.plot(xvalues, pdf(xvalues))
plt.legend()
plt.show()
