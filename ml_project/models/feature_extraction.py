import numpy as np
from tqdm import tqdm
from sklearn.utils.validation import check_array, check_is_fitted

import matplotlib.pyplot as plt


class CubeHisto():
    """ Make cubes and histogram on them """
    def __init__(self, n_cubes=9, n_bins=45):
        self.n_cubes = n_cubes
        self.n_bins = n_bins
        self.newFeatures = None

    def featuresFromSample(self, sampleNr, sample):

        for x in range(0, self.n_cubes):
            for y in range(0, self.n_cubes):
                for z in range(0, self.n_cubes):

                    cube = self.extractCube (x, y, z, sample)
                    histo = self.computeHisto(cube)

                    # if (x == 5 and y == 4 and z == 4):
                    #     plt.bar(range(len(histo)), histo)

                    self.newFeatures[sampleNr, x, y, z, :] = histo




    def extractCube(self, x, y, z, sample):
        """
        extract cube at position (x, y, z) from sample
        """
        (x_dim, y_dim, z_dim) = np.shape(sample)

        # compute cube sizes
        x_cube_size = int(x_dim / self.n_cubes)
        y_cube_size = int(y_dim / self.n_cubes)
        z_cube_size = int(z_dim / self.n_cubes)

        # compute lower index
        x_lower = x * x_cube_size
        y_lower = y * y_cube_size
        z_lower = z * z_cube_size

        # compute upper index (non-inclusive)
        x_upper = (x + 1) * x_cube_size
        y_upper = (y + 1) * y_cube_size
        z_upper = (z + 1) * z_cube_size

        # adapt upper index for last cubes (non-inclusive)
        if (x == self.n_cubes -1):
            x_upper = x_dim

        if (y == self.n_cubes -1):
            y_upper = y_dim

        if (z == self.n_cubes -1):
            z_upper = z_dim

        # print("x:", x_lower, x_upper)
        # print("y:", y_lower, y_upper)
        # print("z:", z_lower, z_upper)

        cube = sample[x_lower:x_upper, y_lower:y_upper, z_lower:z_upper]
        # print(np.shape(cube))
        return cube


    def computeHisto(self, cube):
        bins = np.linspace(0, 4500, self.n_bins+1)
        hist = np.histogram(cube, bins)[0]

        return hist


    def fit(self, X, y=None):
        X = check_array(X)
        (n_samp, _) = np.shape(X)
        X = np.reshape(X, (-1, 176, 208, 176))

        self.newFeatures = np.zeros((n_samp, self.n_cubes, self.n_cubes,
                                    self.n_cubes, self.n_bins))

        for sampleNr in tqdm(range(0, n_samp)):
            sample = X[sampleNr,:,:,:]
            self.featuresFromSample(sampleNr, sample)

        self.newFeatures = np.reshape(self.newFeatures, (n_samp, -1))
        return self


    def transform(self, X, y=None):
        check_is_fitted(self, ["newFeatures"])

        return self.newFeatures





