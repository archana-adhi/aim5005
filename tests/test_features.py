from aim5005.features import MinMaxScaler, StandardScaler
import numpy as np
import unittest
from unittest.case import TestCase

### TO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
        
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        
    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        
    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        
    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))

    # TODO: Add a test of your own below this line
    #Custom test cases    
        
    #1 To test std            
    def test_standard_scaler_get_std(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.std == expected).all(), "scaler fit does not return expected std {}. Got {}".format(expected, scaler.std)        

    #2 Testing standard scaler mean,std and transform with negative values 
    def test_standard_scaler_with_negative_values(self):
        scaler = StandardScaler()
        data = [[-1, -2], [-3, -4], [-5, -6], [-7, -8]]
        expected_mean = np.array([-4., -5.])
        expected_std = np.array([2.23606798, 2.23606798])
        expected_transformed_data = np.array([[ 1.34164079,  1.34164079], [ 0.4472136,  0.4472136], [-0.4472136, -0.4472136], [-1.34164079, -1.34164079]])
        
        transformed_data = scaler.fit_transform(data)
        
        assert (scaler.mean == expected_mean).all(), "Scaler mean does not match expected mean {}. Got {}".format(expected_mean, scaler.mean)
        assert np.allclose(scaler.std, expected_std), "Scaler standard deviation does not match expected std {}. Got {}".format(expected_std, scaler.std)
        assert np.allclose(transformed_data, expected_transformed_data), "Transformed data does not match expected value."  

    #3 Inverse transform
    def test_standard_scaler_inverse_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        scaler.fit(data)
        transformed_data = scaler.transform(data)
        # Perform inverse transform on the transformed data
        inverse_transformed_data = scaler.inverse_transform(transformed_data)
        assert (data == inverse_transformed_data).all(), "Inverse transform did not return the original data. Got: {}".format(inverse_transformed_data)
  
    
if __name__ == '__main__':
    unittest.main()