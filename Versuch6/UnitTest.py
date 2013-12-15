__author__ = 'janhorak'

import unittest

import numpy as np
import facerecognitionTemplate as face

class JUnitTestCases(unittest.TestCase):

    def test_image(self):
        list = face.generateListOfImgs(face.parseDirectory('res/training', 'png'))
        self.assertTrue(list is not None)
        # size of the pitcure is 41750 - so we need a length of 41750 for the matrix
        matrix = face.convertImgListToNumpyData(41750, list)
        self.assertIsNotNone(matrix)
        normedArrayOfFaces = face.calculateNormedArrayOfFaces(41750, matrix)
        self.assertIsNotNone(normedArrayOfFaces)
        
    def test_eigenfaces(self):
        eigenfaces = face.calculateEigenfaces(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]), 3, 1)
        self.assertIsNotNone(eigenfaces)
        
    def test_projectImage(self):
        image1 = [1, 2, 3, 4]
        image2 = [9, 8, 7, 6] 
        eigenfaces = face.calculateEigenfaces(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), 2, 2)
        self.assert_(np.all(np.equal(face.projectImageOnEigenspace(eigenfaces, image1), face.projectImagesOfSamePersonOnEigenspace(eigenfaces, [image1, image1]))), "Projection results must be equal.")
        self.assertIsNotNone(face.projectImagesOfSamePersonOnEigenspace(eigenfaces, [image1, image2]))
         
    def test_saveEigenvectors(self):
        images = face.generateListOfImgs(face.parseDirectory('res/test', 'png'))
        matrix = face.convertImgListToNumpyData(41750, images)
        normedArrayOfFaces = face.calculateNormedArrayOfFaces(41750, matrix)
        face.saveEigenvektorsAsImage(matrix, normedArrayOfFaces)
        

suite = unittest.TestLoader().loadTestsFromTestCase(JUnitTestCases)
unittest.TextTestRunner(verbosity=2).run(suite)