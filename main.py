from utils import split_data,orderTestSet
import os
from deepLearningModel import streetsSignModel

if __name__=="__main__":
    if False:
        pathToData = os.path.join(os.getcwd(),'Data/Train')
        pathToSaveTrain = os.path.join(os.getcwd(),'Data/TrainingData/Train')
        pathToSaveVal = os.path.join(os.getcwd(),'Data/TrainingData/Val')
        split_data(pathToData,pathToSaveTrain,pathToSaveVal)
    pathToImages = os.path.join(os.getcwd(),'Data/Test')
    pathToCSV = os.path.join(os.getcwd(),'Data/Test.csv')
    orderTestSet(pathToImages,pathToCSV)