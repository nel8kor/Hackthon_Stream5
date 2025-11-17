# Build a unit test agent for the 3D CNN model
import unittest
import numpy as np
from keras.models import load_model

class Test3DCNNModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the trained model
        cls.model = load_model('model.h5')

    def test_model_input_shape(self):
        # Test the model input shape
        self.assertEqual(self.model.input_shape, (None, 10, 120, 160, 3))

    def test_model_output_shape(self):
        # Test the model output shape
        self.assertEqual(self.model.output_shape, (None, 2))

    def test_model_prediction(self):
        # Test the model prediction
        X_test = np.random.rand(1, 10, 120, 160, 3)
        y_pred = self.model.predict(X_test)
        self.assertEqual(y_pred.shape, (1, 2))

if __name__ == '__main__':
    unittest.main()
