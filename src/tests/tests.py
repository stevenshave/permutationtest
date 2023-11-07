from permutationtest import permutation_test
import pytest

def test_permtest_close_small(treatment_vehicle_close_2_features_small):
    treatment, vehicle=treatment_vehicle_close_2_features_small
    assert permutation_test(treatment, vehicle) == pytest.approx(0.138195, 0.00001)

def test_permtest_far_small(treatment_vehicle_far_2_features_small):
    treatment, vehicle=treatment_vehicle_far_2_features_small
    assert permutation_test(treatment, vehicle) == pytest.approx(0.020979, 0.00001)
    
def test_permtest_close_big(treatment_vehicle_close_2_features_big):
    treatment, vehicle=treatment_vehicle_close_2_features_big
    assert permutation_test(treatment, vehicle) == pytest.approx(0.1519, 0.00001)

def test_permtest_far_big(treatment_vehicle_far_2_features_big):
    treatment, vehicle=treatment_vehicle_far_2_features_big
    assert permutation_test(treatment, vehicle) == pytest.approx(0.0137, 0.00001)
    
    