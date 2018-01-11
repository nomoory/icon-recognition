import sys
from scipy import misc
from pdb import set_trace
import numpy as np
import tensorflow as tf
from PIL import Image
from os import listdir, getcwd, mkdir, chdir
from os.path import isdir, join
import random

NUM_REFINED_FILES_PER_LABEL = 5000
RAW_DIR = dirPath = join(getcwd(), 'iconDataset') 
TARGET_DIR = join(getcwd(), 'jpgImages')
PERCENTAGE_OF_TRAINING_DATASET = 0.8

# "targetLabels" is list of file names that you wanna train
def getFeedDataforClassification(targetLabels = 'none'):
    labels = listdir(TARGET_DIR)
    del labels[labels.index('negative')]
    labels.sort()

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    XYs= []
    for idx, label in enumerate(labels):
        print(label, "is on reading...(%d)" % idx)
        fileNames = listdir(join(TARGET_DIR, label))
        XYs = [[misc.imread(join(TARGET_DIR, label, fileName)), fileName.split('_')[0]] for fileName in fileNames if targetLabels == 'none' or fileName in targetLabels]

    # shuffling data to divide test and train data randomly
    random.shuffle(XYs)
    num_total = len(XYs) 
    num_train = int(PERCENTAGE_OF_TRAINING_DATASET * num_total)

    x = x_train
    y = y_train
    # for classification
    for idx, XY in enumerate(XYs):
        if num_train <= idx :
            x = x_test
            y = y_test
        x.append(XY[0])
        y.append(XY[1])
            
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    return (x_train, y_train), (x_test, y_test)

def getFeedDataforPrediction(targetFilePaths):
    x_prediction = []
    y_prediction = []    
    XYs = []

    # read image file
    for filePath in targetFilePaths:
        fileName = filePath.split('/')[-1]
        XYs.append([misc.imread(filePath), fileName.split('_')[0]])

    for XY in XYs:
        x_prediction.append(XY[0])
        y_prediction.append(XY[1])
    x_prediction = np.array(x_prediction)
    y_prediction = np.array(y_prediction)
    return (x_prediction, y_prediction)
    
def refineImageData(dirPath):
    labelDic = {}
    for labelName in listdir(dirPath):
        labelPath = join(dirPath, labelName)
        labelDic[labelName] = {
            'labelPath': labelPath,
            'fileNames': [ fileName for fileName in listdir(labelPath) if 'png' in fileName]
        }
    
    for idx, labelName in enumerate(labelDic):
        i = 0
        labelInfo = labelDic[labelName]
        print("{0} is on converting...".format(labelName))
        while i < NUM_REFINED_FILES_PER_LABEL:
            for fileName in labelInfo.get('fileNames'):
                filePath = join(labelInfo.get('labelPath'), fileName)
                backgroundColor = (random.randrange(0,255),random.randrange(0,255),random.randrange(0,255))
                newFileName = str(idx) + '_' + fileName.split('.')[0] + str(i)
                isConverted = convertPngToJpg(filePath, newFileName, labelName, backgroundColor)
                if isConverted: i = i+1
                if i == NUM_REFINED_FILES_PER_LABEL: 
                    break

def convertPngToJpg(filePath, fileName, label, backgroundColor): 
    with Image.open(filePath) as pilImg:
        pilImg.load()
        bg = Image.new("RGB", pilImg.size, backgroundColor)
        try:
            bg.paste(pilImg, mask=pilImg.split()[3])
        except:
            return False
        else:
            # bg.convert('RGB')
            # resize image
            basewidth = 128
            bg = bg.resize((basewidth, basewidth), Image.ANTIALIAS)

            target_label_dir = join(TARGET_DIR, label)
            if label not in listdir(TARGET_DIR):
                mkdir(target_label_dir)

            chdir(target_label_dir)
            bg.save('{0}.jpeg'.format(fileName))
            chdir(TARGET_DIR)

            return True

if __name__ == "__main__":
    refineImageData(RAW_DIR)