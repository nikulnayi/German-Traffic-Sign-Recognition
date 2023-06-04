import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import csv
def split_data(pathToData,pathToSaveTrain,pathToSaveVal,splitSize=0.1):
    
    folders = os.listdir(pathToData)
    for folder in folders:
        fullPath = os.path.join(pathToData,folder)
        imagesPath = glob.glob(os.path.join(fullPath,'*.png'))
        x_train, x_val = train_test_split(imagesPath,test_size=splitSize)
        for x in x_train: 
            pathToFolder = os.path.join(pathToSaveTrain,folder)
            if not os.path.isdir(pathToFolder):
                os.makedirs(pathToFolder)
            shutil.copy(x, pathToFolder)
        for x in x_val:
            pathToFolder = os.path.join(pathToSaveVal,folder)
            if not os.path.isdir(pathToFolder):
                os.makedirs(pathToFolder)
            shutil.copy(x, pathToFolder)

def orderTestSet(pathToImages,pathToCSV):
    try:
        with open(pathToCSV,'r') as csvFile:
            reader = csv.reader(csvFile,delimiter=',')
            for i, row in enumerate(reader):
                if i==0:
                    continue
                imgName = os.path.basename(row[-1])
                label = row[-2]
                pathToFolder = os.path.join(pathToImages,label)
                if not os.path.isdir(pathToFolder):
                    os.makedirs(pathToFolder)
                imgFullPath = os.path.join(pathToImages,imgName)
                shutil.move(imgFullPath,pathToFolder)
    except:
        print('[INFO] : Error reading CSV file')
                    
