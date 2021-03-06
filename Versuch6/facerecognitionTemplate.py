from os.path import isdir,join,normpath,abspath
from os import listdir
from os import startfile
from os import unlink

import Image

import numpy as np

import tkFileDialog
from nltk.chunk.named_entity import shape
from decimal import Clamped
import math
import webbrowser


def parseDirectory(directoryName,extension):
    '''This method returns a list of all filenames in the Directory directoryName. 
    For each file the complete absolute path is given in a normalized manner (with 
    double backslashes). Moreover only files with the specified extension are returned in 
    the list.
    '''
    if not isdir(directoryName): return
    imagefilenameslist=sorted([
        normpath(join(directoryName, fname))
        for fname in listdir(directoryName)
        if fname.lower().endswith('.'+extension)            
        ])
    return imagefilenameslist

#####################################################################################
# Implement required functions here
#

def generateListOfImgs(list):
    '''
    This Function returns the list of images (Type Image) for the passed list of paths.

    Parameters:
    ===========

    list -> List of paths of Images (Type= String)

    Returns:
    ========
    resultList -> List of Images (Type = Image)
    '''
    resultList = []
    for i in range(len(list)):
        try:
            image = Image.open(list[i]).convert('L')
            print str('size of ' +str(list[i]) + ': '+ str(image.size[0]) + "x" + str(image.size[1]))
            resultList.append(image)
        except StandardError:
            print 'Error: Can\'t read file: ' + str(image)
    return resultList

def convertImgListToNumpyData(length, list):
    '''
    This Function returns a matrix (Type= numpy.asfarray) with the length of an Image
    (per Default 41750) and a depth of all images (per Default 21).

    A single value of Matrix[x][y] is the value of the Pixel y in the Image x.
    The Matrix has the values:
    x (0:20)
    y (0:41750)
    at 21 Pictures in the testdata and an imagesize of 167x250 pixels

    IMPORTANT
    =========
    If you want to set an other Image (with other size) you have to pass a new value for
    length

    Parameters:
    ===========

    length -> Size of the image (per Default 14750 | for an image with size: 167x250)

    list -> list of images

    Returns:
    ========
    resultMatrix -> Matrix (description in header)
    '''

    # creates the resultmatrix of the function
    resultMatrix = np.asfarray(np.zeros(shape=(len(list), len(range(length)))), dtype='float')

    for k in range(len(list)):
        # creates the important values per image or per main-loop
        highestValue = 0
        image = list[k]
        width = image.size[0]
        height = image.size[1]
        pixelData = []

        # splits the image and iterates over the pixel-data
        for i in range(height):
            for j in range(width):
                # get the value of the pixel a i(height), j(width) - Position
                pixelValue = image.getpixel((j, i))
                # checks if we get a new highest value
                if pixelValue > highestValue:
                    highestValue = pixelValue
                # adding the value of the pixel to the pixeldata-Array
                pixelData.append(pixelValue)

        # if the pixeldata-array is completed, we are dividing every pixel through the
        # highest value
        for i in range(len(pixelData)):
            newValue = float(float(pixelData[i]) / float(highestValue))
            pixelData[i] = newValue

        # writing the new pixeldata of a single image in a new row of the matrix
        counter = 0
        for x in pixelData:
            resultMatrix[k][counter] = x
            counter += 1

    return resultMatrix
    
def calculateEuclideanDistance(A, B):
    '''
    This Function calculates the euclidean distance between two Matrices

    Parameters:
    ===========

    A --> Matrix A
    B --> Matrix B

    Returns:
    ========
    float-value of the euclidean distance
    '''
    
    flatA = A.flatten()
    flatB = B.flatten()
    
    costs = 0
    
    for i in range(len(flatA)):
        costs = costs + math.pow(flatA[i]-flatB[i],2)
    
    costs = math.sqrt(costs)
    return costs


def calculateAverageArrayOfFaces(matrix):
    '''
    This Function calculates the average of a matrix

    Parameters:
    ===========

    matrix -> matrix for the images
    
    Returns:
    ========
    Array, average image. Each value represents a pixel.
    '''
        
    imageSum = np.zeros(np.shape(matrix[0]))
    
    for row in range(len(matrix)):
        imageSum = imageSum + matrix[row]
    averageImage = imageSum / len(matrix)
    
    return averageImage
   
   
def subtractAverage(imageArray, averageImage):
    '''
    This Function substracts a value from a matrix

    Parameters:
    ===========

    imageArray -> array, each value represents a pixel
    averageImage: substract values

    Returns:
    ========
    matrix with substracted value in each column/row
    '''
    # defines the resultMatrix
    resultMatrix = np.empty(np.shape(imageArray))
    
    # calculates the normed matrix
    for idx in range(len(imageArray)):
        resultMatrix[idx] = imageArray[idx]
        resultMatrix[idx] -= averageImage[idx]
        if resultMatrix[idx] < 0:
            resultMatrix[idx] = 0
            
    return resultMatrix

def calculateNormedArrayOfFaces(length, matrix):
    '''
    This function returns the normed values of the matrix.

    Parameters:
    ===========

    length -> Size of the image (per Default 14750 | for an image with size: 167x250)

    matrix -> matrix for the images

    Returns:
    ========
    resultMatrix -> normed matrix of pixelvalues
    '''
    # defines the resultMatrix
    resultMatrix = np.empty(np.shape(matrix))
    
    # calculates the average of the images
    averageImage = calculateAverageArrayOfFaces(matrix)
    
    # calculates the normed matrix
    for idx in range(len(matrix)):
        resultMatrix[idx] = subtractAverage(matrix[idx], averageImage)
            
    return resultMatrix



def calculateEigenfaces(adjfaces,width = None,height = None):
    '''
    Calculates all possible eigenfaces for a given matrix.
    The result will be ordered by eingenvalues in descending order.
    
    Parameter:
    adjfaces: numpy array, normed picture arrays (normed return value of convertImgListToNumpyData()) 
    width:    integer, width of each picture (all pictures must be the same size)
    height:   integer, height of each picture (all pictures must be the same size)
    
    Return:
    Numpy array, each row is a normed eigenface.
    '''
    
    transposedAdjfaces = np.transpose(adjfaces)
    
    xTmultiX = np.dot(adjfaces, transposedAdjfaces)
    
    eigenValues, eigenVectors = np.linalg.eigh(xTmultiX)
    
    sortedEigenValueIndices = np.argsort(eigenValues)[::-1]
    
    eigenfaces = np.zeros((eigenVectors.shape[0], adjfaces.shape[1]))
    
    eigenVectorIdx = 0
    
    for eigenVector in eigenVectors:
        
        dimensionIdx = 0
        
        for adjface in adjfaces:
            
            eigenfaces[eigenVectorIdx] = eigenfaces[eigenVectorIdx] + (eigenVector[dimensionIdx] * adjface)
            
            dimensionIdx = dimensionIdx + 1
        
        eigenVectorIdx = eigenVectorIdx + 1
    
    return eigenfaces[sortedEigenValueIndices]


def projectImageOnEigenspace(eigenVectors, image):
    '''
    Project the image (with the same dimension than its pixel count)
    to an eigenspace that is defined by the passed eigenVectors.
    
    Parameters:
    eigenVectors: 2 dimensional List/array, used to define eigenspace, each row represents a eigenvector
    image:        list/array, each value represents a pixel
    
    Return:
    Numpy array, point in eigenspace that represents the image.
    '''    
    
    eigenspacePoint = np.empty(eigenVectors.shape[0])
    
    idx = 0
    
    for eigenVector in eigenVectors:
        
        eigenspacePoint[idx] = np.dot(np.transpose(eigenVector), image)
        idx = idx + 1
    
    return eigenspacePoint


def projectImagesOfSamePersonOnEigenspace(eigenVectors, images):
    '''
    It's recommended to train with more than one picture for
    one person. To calculate the distances to this person,
    that is represented by multiple pictures, it's required
    to calculate the average of all pictures of the person.
    The average should be used to compare with instead of each 
    single picture.
    
    Parameters:
    eigenVectors: 2 dimensional List/array, used to define eigenspace, each row represents a eigenvector
    images:       2 dimensional list/array, each value represents a image, each image is a array of pixels
    
    Return:
    Numpy array, point in eigenspace that represents the average image.    
    '''
    
    eigenspacePointSum = projectImageOnEigenspace(eigenVectors, images[0])
    
    for image in images[1:]:
        eigenspacePointSum = eigenspacePointSum + projectImageOnEigenspace(eigenVectors, image)
        
    return eigenspacePointSum / len(images) 



def saveEigenvektorsAsImage(imageAverage, eigenVectors, imageRatio = (167, 250)):
    '''
    Write a picture representation of each eigenvector
    to the disk ('res/eigenvectors'). This makes it 
    possible to see how the eigenfaces look like.
    The normalization cut a little bit of the darker
    areas in the pictures, so this is not fully 
    compareable to the original images.
    
    Parameters:
    imageAverage:   return value of calculateAverageArrayOfFaces()
    eigenVectors:   2 dimensional List/array, used to define eigenspace, each row represents a eigenvector
    imageRatio:     list/tupel, dimension of the images, first value represents width, second height
    
    '''    
    
    targetDir = "res/eigenvectors/"
    
    printableEigenVectors = calculateNormedArrayOfFacesReverse(imageAverage, eigenVectors)
    
    removePngsInDirectory(abspath(targetDir))
    
    vectorIdx = 0
    for eigenVector in printableEigenVectors:      
        
        imageArray = np.empty(len(eigenVector), dtype="uint8")
        
        pixelIdx = 0        
        for pixel in eigenVector:
            imageArray[pixelIdx] = pixel * 255
            pixelIdx = pixelIdx + 1
        
        imageArray = np.reshape(imageArray, (imageRatio[1], imageRatio[0]))
                
        image = Image.fromarray(imageArray, "L")        
        image.save(targetDir + "vector_" + str(vectorIdx) + ".png", "png")
        
        vectorIdx = vectorIdx + 1
        

def calculateNormedArrayOfFacesReverse(imageAverage, normedArray):
    '''
    This function tries to undo the changes performed by calculateNormedArrayOfFaces().
    This is required if you changed the faces and want to print them out.
    A lot of information gets lost in the normalization. The result of this
    function is not as good as you may be want it to be.

    Parameters:
    originalMatrix: numpy array, same imageAverage that was passed to calculateNormedArrayOfFaces()
    normedArray:    numpy array, return value of calculateNormedArrayOfFaces()

    Returns:
    Numpy array, like originalMatrix.
    '''
    pictureLength = len(normedArray[0])
    
    resultMatrix = np.empty(np.shape(normedArray))
    
    for row in range(len(normedArray)):
        for coulumn in range(pictureLength):
            resultMatrix[row][coulumn] = normedArray[row][coulumn] + imageAverage[row]
            if resultMatrix[row][coulumn] > 1:
                resultMatrix[row][coulumn] = 1
                  
    return resultMatrix

def removePngsInDirectory(path):
    '''
    Delete all png files in a directory.
    
    Parameter:
    path: String, absolute path to directory
    '''
    for file in parseDirectory(path, "png"):
        try:
            unlink(file)
        except Exception, e:
            print e            
    
    
def mainTest(eigenfaceCount = 3, maximumDistance = 50):
    
    #Choose Directory which contains all training images
    TrainDir=tkFileDialog.askdirectory(title="Choose Directory of training images")
    
    #Choose the file extension of the image files
    Extension='png'
    
    #Choose the image which shall be recognized       
    testImageDirAndFilename=tkFileDialog.askopenfilename(title="Choose Image to detect")
    
    imagelist = parseDirectory(TrainDir, Extension)
    faceList = generateListOfImgs(imagelist)
    faceList2 = [1]
    faceList2[0] = Image.open(testImageDirAndFilename).convert('L')

    # all images must be the same size
    imageSize = faceList[0].size[0] * faceList[0].size[1]
    
    # size of the pitcure is 41750 - so we need a length of 41750 for the matrix
    matrix = convertImgListToNumpyData(imageSize, faceList)
    matrix2 = convertImgListToNumpyData(imageSize, faceList2)
    
    # calculate the average of trainingpictures
    imageAverage = calculateAverageArrayOfFaces(matrix)
    
    normedArrayOfFaces = calculateNormedArrayOfFaces(imageSize, matrix)
    
    eigenfaces = calculateEigenfaces(normedArrayOfFaces)
    
    # set the dimension of Eigenfaces
    relevantEigenfaces = eigenfaces[:eigenfaceCount]
    
    saveEigenvektorsAsImage(imageAverage, relevantEigenfaces)
    
    # substract the average from the testpicture
    NormedTestFace = subtractAverage(matrix2[0], imageAverage)
    
    distances = np.empty(len(normedArrayOfFaces))
    
    eigenspaceBild = projectImageOnEigenspace(relevantEigenfaces, NormedTestFace)
    
    for j in range(len(normedArrayOfFaces)):
        
        tmpeigenspaceBild = projectImageOnEigenspace(relevantEigenfaces, normedArrayOfFaces[j])
        
        # calculate the euclidean distance between xthe two eigenfaces
        distances[j] = calculateEuclideanDistance(eigenspaceBild, tmpeigenspaceBild)
    
    keysSortedDistances = np.argsort(distances)
    
    if distances[keysSortedDistances[0]] > maximumDistance:
        print "Keine Uebereinstimmung gefunden - Distanz: " + str(distances[keysSortedDistances[0]])
    else:
        print "Testbild: " + str(testImageDirAndFilename)
        print "Aehnlichstes Bild: " + str(imagelist[keysSortedDistances[0]])
        print "Distanz: " + str(round(distances[keysSortedDistances[0]],2))
        startfile(imagelist[keysSortedDistances[0]])
    
    
# Don't execute this if this file is used as a module in another file. 
if __name__ == '__main__':    
    mainTest()        
