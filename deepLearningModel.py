import tensorflow
from tensorflow.keras.layers import Conv2D, Input, Dense,  MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten 
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  

def createGenerators(batchSize,trainDataPath,valDataPath,testDataPath):
    preprocessor = 


def streetsSignModel(numberOfClasses):
    myInput = Input(shape=(60,60,3))

    x = Conv2D(32,(3,3), activation='relu')(myInput)
    x = Conv2D(64,(3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128,(3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10,activation='softmax')(x)

    model = Model(inputs=myInput,outputs = x)
    return model

if __name__=="__main__":
    model = streetsSignModel(10)
    model.summary()
