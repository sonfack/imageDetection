import cv2
import _pickle as cPickle
import os, glob
import shutil
from matplotlib import pyplot as plt
import numpy as np

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


def createTestAndTrainningData(datasetFolder, test="./test/", training="./training/"):
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
    dataset = os.path.dirname(datasetFolder) + "/" + os.path.basename(datasetFolder)
    for file in files:
        if count % 2 == 0:
            shutil.copy(dataset + '/' + file, test)
        else:
            shutil.copy(dataset + '/' + file, training)

        count = count + 1

    print("Total files copied : " + str(count))


# Create SIFT descritor file
def saveSIFTDescriptorAndKeypointFile(imageFile, test="./training", model="./model"):
    image = os.path.join(test, imageFile)
    img = cv2.imread(image, 0)
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img, None)

    keyDescriptor = {'kp': kp1, 'des': des1}
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

def createTrainingSIFTFiles(trainingFolder="./training"):
    for file in os.listdir(trainingFolder):
        saveSIFTDescriptorAndKeypointFile(file)


# read descriptor file

def readDescriptorFileAndDrawKp(filename, model="./model"):
    keyPointDescriptor = []
    '''
        for file in os.listdir(trainingFolder):
            filenameDir, file_extension = os.path.splitext(os.path.join(trainingFolder, file))

            f = filenameDir.split("/")

            if f[-1] == filename:
                image = filenameDir+file_extension
                print(image)
                break
        exit(0)
        img = cv2.imread(image)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''

    filenameWithPath = os.path.join(model, filename)
    # print(filenameWithPath)
    index = cPickle.loads(open(filenameWithPath, "rb").read())
    '''
    kp = []
    for point in index:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2],
                            _response=point[3], _octave=point[4], _class_id=point[5])
        kp.append(temp)

    '''
    keyPointDescriptor.append(index.get('kp'))
    keyPointDescriptor.append(index.get('des'))
    # print(keyPointDescriptor)

    return keyPointDescriptor
    '''
    # Draw the keypoints 
    imm = cv2.drawKeypoints(gray, kp, img)
    cv2.imshow("Key points", imm)
    cv2.waitKey(0)
    '''


# keypoints matcher

def keypointsMatcher(queryImage, testFolder="./test", modelFolder="./model"):
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = cv2.imread(os.path.join(testFolder, queryImage), 0)  # queryImage

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)

    # read through the model folder
    listOfModel = os.listdir(modelFolder)
    print("Number of models " + str(len(listOfModel)))

    for model in listOfModel:

        keyAndDescriptor = readDescriptorFileAndDrawKp(model)
        # print(keyAndDescriptor)

        kp2 = keyAndDescriptor
        des2 = keyAndDescriptor[1]

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        print(len(np.array(matches).shape))
        print(np.array(matches).shape[1])
        # Apply ratio test
        good = []
        # good_without_list = []

        if len(np.array(matches).shape) == 2 and np.array(matches).shape[1] == 2:
            print(model)
            for m, n in matches:
                if m.distance < 0.9 * n.distance:
                    good.append([m])
                    # good_without_list.append(m)

            # if len(good_without_list) > 50:
            # print(model)

    '''
        img2 = cv2.imread(os.path.join(testFolder, queryImage), 0)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_without_list, None)

        plt.imshow(img3)
        plt.show()

        count = count + 1

        if count == 3 :
            exit(0)
    '''

    # img1 = cv2.imread(os.path.join(testFolder, queryImage), 0)  # queryImage
    # img2 = cv2.imread(os.path.join(testFolder, searchImage), 0)  # trainImage

    # Initiate SIFT detector
    # sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1, None)
    # kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    # good = []
    # good_without_list = []

    # for m, n in matches:
    # if m.distance < 0.95 * n.distance:
    # good.append([m])
    # good_without_list.append(m)

    # img2 = cv2.imread(os.path.join(testFolder, queryImage), 0)
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_without_list, None)

    # plt.imshow(img3)
    # plt.show()


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
    # test()
    # createTestAndTrainningData("./dataset")
    # createTrainingSIFTFiles("./training")
    # createSIFTDescriptorFile("obj1__15.png")
    keypointsMatcher("obj2__185.png")
    # print(readDescriptorFileAndDrawKp("obj3__125"))
    # print(denseSIFT("butterfly.jpg"))


if __name__ == "__main__":
    main()

