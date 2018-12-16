import cv2
import _pickle as cPickle
import os, glob
import shutil
from matplotlib import pyplot as plt
import numpy as np
from collections import Counter


"""
    How to install Opencv version containing SIFT and SURF image descriptor
    pip install opencv-python==3.4.2.16

    pip install opencv-contrib-python==3.4.2.16
"""

# Object detection

# 1 divide your data set into test en trainning


def test():

    img = cv2.imread('butterfly.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    print(sift)

def createTestAndTrainningData(datasetFolder, test = "./test/", training = "./training/"):
    files = os.listdir(datasetFolder)
    print("Number of dataset : " + str(len(files)))

    # Create target Directory if don't exist
    if not os.path.exists(test) and not os.path.exists(training):
        os.mkdir(test)
        print("Directory ", test, " Created ")
        os.mkdir(training)
        print("Directory ", training, " Created ")
    else:
        print("Directories  already exists")

    count = 0
    dataset = os.path.dirname(datasetFolder)+"/"+os.path.basename(datasetFolder)
    for file in files:
        if count % 2 == 0:
            shutil.copy(dataset+'/'+file, test)
        else:
            shutil.copy(dataset + '/' + file, training)

        count = count + 1

    print("Total files copied : "+str(count))


# Create SIFT descritor file
def saveSIFTDescriptorAndKeypointFile(imageFile, test="./training", model="./model"):
    image = os.path.join(test, imageFile)
    img = cv2.imread(image, 0)
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img, None)

    keyDescriptor = {'kp': kp1, 'des':des1}
    image_w_ext = os.path.basename(image)
    filename, file_extension = os.path.splitext(image_w_ext)

    # Dump the keypoints
    f = open(filename, "wb+")
    f.write(cPickle.dumps(keyDescriptor))
    f.close()

    # Store file in model directory
    if not os.path.exists(model):
        os.mkdir(model)
        print("Directory ", model, " Created ")
    else:
        print("Directories  already exists")

    shutil.move(filename, model)


def createSIFTDescriptorFile(imageFile, test="./training", model="./model"):
    image = os.path.join(test, imageFile)
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)

    index = []
    for point in kp:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
                point.class_id)
        index.append(temp)
    image_w_ext = os.path.basename(image)
    filename, file_extension = os.path.splitext(image_w_ext)

    # Dump the keypoints
    f = open(filename, "wb+")
    f.write(cPickle.dumps(index))
    f.close()

    # Store file in model directory
    if not os.path.exists(model):
        os.mkdir(model)
        print("Directory ", model, " Created ")
    else:
        print("Directories  already exists")

    shutil.move(filename, model)


# create SIFT file for all training set

def createTrainingSIFTFiles(trainingFolder = "./training"):
    for file in os.listdir(trainingFolder):
        saveSIFTDescriptorAndKeypointFile(file)


# read descriptor from file with keypoint

def readDescriptorFileAndDrawKp(filename, model= "./model"):
    keyPointDescriptor = []
    filenameWithPath = os.path.join(model,filename)
    index = cPickle.loads(open(filenameWithPath, "rb").read())
    keyPointDescriptor.append(index.get('kp'))
    keyPointDescriptor.append(index.get('des'))
    return keyPointDescriptor


'''
    Keypoints matcher, takes an image as queryImage and looks in the model folder descriptors tha match better 
    this image.
    queryImage should be taken from the ./test folder
    distancePercentage = 0.75  is the distance that is selected for each image corresponding image in the model
'''

def keypointsMatcher(queryImage, testFolder = "./test", modelFolder= "./model", distancePercentage=0.75):
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = cv2.imread(os.path.join(testFolder, queryImage), 0)  # queryImage

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)

    # read through the model folder
    listOfModel = os.listdir(modelFolder)
    print("Number of models "+str(len(listOfModel)))
    print(len(des1))
    print(len(kp1))
    numberKpToSelect = 2*(len(kp1)//3)
    lilstOfSelectedModel = []

    for model in listOfModel:
        keyAndDescriptor = readDescriptorFileAndDrawKp(model)
        des2 = keyAndDescriptor[1]

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des2, des1, k=2)

        # Apply ratio test
        good = []

        if len(np.array(matches).shape) == 2 and np.array(matches).shape[1] == 2:
            for m, n in matches:
                if (m.distance/n.distance) > distancePercentage:
                    good.append([m])
            if len(good) > numberKpToSelect:
                lilstOfSelectedModel.append(model)
                print(model, " ", "selected")

    #print(lilstOfSelectedModel)
    seen = []
    for ob in lilstOfSelectedModel:
        filename = ob.split("__")[0]
        seen.append(filename)
    listSeen = Counter(seen)
    for element in listSeen:
        print(element,' ', listSeen[element], "  ",countCategoryElement(element), listSeen[element]/countCategoryElement(element))
    print(len(lilstOfSelectedModel))



def calculateCorrespond(desQuery, desModel, numberMatches):
    return numberMatches/(len(desQuery) + len(desModel))


def countCategoryElement(catname, modelFolder="./model"):
    # read through the model folder
    listOfModel = os.listdir(modelFolder)
    count = 0
    for filename in listOfModel:
        if catname+"__" in filename:
            count = count + 1
    return count

def denseSIFT(img, step_size=20, feature_scale=40, img_bound=20):
    # Create a dense feature detector
    detector = cv2.FeatureDetector_create("Dense")

    # Initialize it with all the required parameters
    detector.setInt("initXyStep", step_size)
    detector.setInt("initFeatureScale", feature_scale)
    detector.setInt("initImgBound", img_bound)
    # Run feature detector on the input image
    return detector.detect(img)



def main():
    #test()
    #createTestAndTrainningData("./dataset")
    #createTrainingSIFTFiles("./training")
    #createSIFTDescriptorFile("obj1__15.png")
    keypointsMatcher("obj91__180.png", distancePercentage=0.5)
    #print(readDescriptorFileAndDrawKp("obj3__125"))
    #print(denseSIFT("butterfly.jpg"))

if __name__ == "__main__":
    main()

