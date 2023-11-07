import pytest
import numpy as np
@pytest.fixture
def treatment_vehicle_far_2_features_small():
    np_rng=np.random.default_rng(7)
    treatment=np_rng.multivariate_normal(mean=(0.7,0.7), cov=np.eye(2), size=5)
    vehicle=np_rng.multivariate_normal(mean=(0,0), cov=np.eye(2), size=10)
    return treatment, vehicle

@pytest.fixture
def treatment_vehicle_close_2_features_small():
    np_rng=np.random.default_rng(7)
    treatment=np_rng.multivariate_normal(mean=(0.3,0.3), cov=np.eye(2), size=5)
    vehicle=np_rng.multivariate_normal(mean=(0,0), cov=np.eye(2), size=10)
    return treatment, vehicle

@pytest.fixture
def treatment_vehicle_far_2_features_big():
    np_rng=np.random.default_rng(7)
    treatment=np_rng.multivariate_normal(mean=(0.7,0.7), cov=np.eye(2), size=20)
    vehicle=np_rng.multivariate_normal(mean=(0,0), cov=np.eye(2), size=100)
    return treatment, vehicle

@pytest.fixture
def treatment_vehicle_close_2_features_big():
    np_rng=np.random.default_rng(7)
    treatment=np_rng.multivariate_normal(mean=(0.5,0.5), cov=np.eye(2), size=20)
    vehicle=np_rng.multivariate_normal(mean=(0,0), cov=np.eye(2), size=100)
    return treatment, vehicle
