import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import split_data,orderTestSet
import os
from deepLearningModel import streetsSignModel, createGenerators

if __name__=="__main__":
   
    # pathToData = os.path.join(os.getcwd(),'Data/Train')
    # pathToSaveTrain = os.path.join(os.getcwd(),'Data/TrainingData/Train')
    # pathToSaveVal = os.path.join(os.getcwd(),'Data/TrainingData/Val')
    # split_data(pathToData,pathToSaveTrain,pathToSaveVal)

    
    # pathToImages = os.path.join(os.getcwd(),'Data/Test')
    # pathToCSV = os.path.join(os.getcwd(),'Data/Test.csv')
    # orderTestSet(pathToImages,pathToCSV)


    pathToTrain = os.path.join(os.getcwd(),'Data/TrainingData/Train')
    pathToVal = os.path.join(os.getcwd(),'Data/TrainingData/Val')
    pathToTest = os.path.join(os.getcwd(),'Data/Test')
    pathToSaveModel = 'Model/'
    batchSize = 64
    epochs = 15
    lr = 0.0001
    
    trainGenerator, valGenerator, testGenerator = createGenerators(batchSize,pathToTrain,pathToVal,pathToTest)
    numberOfClasses = trainGenerator.num_classes

    TRAIN=False
    TEST=True
    if TRAIN:
        pathToSaveModel = 'Model/'
        checkPointSaver = ModelCheckpoint(
            pathToSaveModel,
            monitor = 'val_accuracy',
            mode = 'max',
            save_best_only = True,
            save_freq = 'epoch',
            verbose = 1
        )   

        earlyStop = EarlyStopping(monitor='val_accuracy', patience = 10)

        model = streetsSignModel(numberOfClasses)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, amsgrad=True)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(trainGenerator, 
                epochs=epochs,
                batch_size=batchSize,
                validation_data=valGenerator,   
                callbacks=[checkPointSaver, earlyStop]
                )
    
    if TEST:
        model = tf.keras.models.load_model('./Model')
        model.summary()
        print('Evaluating Validation Set : ')
        model.evaluate(valGenerator)
        print('Evaluating Test Set : ')
        model.evaluate(testGenerator)
