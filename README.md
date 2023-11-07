# permutationtest
Python implementation of permutation testing

Example
######

```python
from permutationtest import permutation_test
import numpy as np

# Generate some data
np_rng=np.random.default_rng(7)
treatment=np_rng.multivariate_normal(mean=(0.7,0.7), cov=np.eye(2), size=20)
vehicle=np_rng.multivariate_normal(mean=(0,0), cov=np.eye(2), size=100)

# Perform the test
permutation_test(treatment, vehicle)

```

Outputs a p-value for treatment being drawn from the same distribution as the vehicle of .0137
