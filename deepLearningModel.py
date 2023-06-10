import tensorflow
from tensorflow.keras.layers import Conv2D, Input, Dense,  MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten 
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  

def createGenerators(batchSize,trainDataPath,valDataPath,testDataPath):
    trainPreprocessor = ImageDataGenerator(
        rescale = 1/255.,
        rotation_range = 10,
        width_shift_range = 0.1
    )
    testPreprocessor = ImageDataGenerator(
        rescale = 1/255.,
    )

    train_generator = trainPreprocessor.flow_from_directory(
        trainDataPath,
        class_mode="categorical",
        target_size=(60,60),
        color_mode='rgb',
        shuffle=True,
        batch_size=batchSize
    )

    val_generator = testPreprocessor.flow_from_directory(
        valDataPath,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batchSize,
    )

    test_generator = testPreprocessor.flow_from_directory(
        testDataPath,
        class_mode="categorical",
        target_size=(60,60),
        color_mode="rgb",
        shuffle=False,
        batch_size=batchSize,
    )

    return train_generator, val_generator, test_generator



def streetsSignModel(numberOfClasses):
    myInput = Input(shape=(60,60,3))

    x = Conv2D(32,(3,3), activation='relu')(myInput)
    x = Conv2D(64,(3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128,(3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    # x = Flatten()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(numberOfClasses,activation='softmax')(x)

    return Model(inputs=myInput,outputs = x)

# if __name__=="__main__":
#     model = streetsSignModel(10)
#     model.summary()
