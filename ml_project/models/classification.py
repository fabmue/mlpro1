# classification file

from sklearn.utils.validation import check_X_y, check_is_fitted
import numpy as np

class TestClass():
    def __init__(self, save_path=None):
        print("TestClass: init\n")
        self.save_path = save_path


    def fit(self, X, y):
        print("TestClass: fit\n")
        X, y = check_X_y(X, y)
        # add dummy feature
        self.dummyFeature = 93.0

        return self


    def predict(self, X):
        print("TestClass: predict\n")
        if (self.dummyFeature is None):
            print("dF is None")
        #check_is_fitted(self, ["dummyFeature"])

        print("dF: ", self.dummyFeature)
        nSamples = np.shape(X)[0]
        prediction = np.zeros((nSamples,1)) + self.dummyFeature

        return prediction


    def score(self, X, y, sample_weight=None):
        print("TestClass: score\n")

    def set_save_path(self, save_path):
        print("TestClass: set_save_path\n")
        self.save_path = save_path
