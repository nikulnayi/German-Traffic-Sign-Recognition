# Traffic Sign Recognition using Deep Learning

This project aims to develop a deep learning model for traffic sign recognition using convolutional neural networks (CNNs). The model is trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset, which contains labeled images of various traffic signs.

## Dataset
The dataset used in this project can be found on Kaggle: [GTSRB - German Traffic Sign Recognition](https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## Requirements
- Python 3.x
- TensorFlow
- Keras
- scikit-learn

## Project Structure
The project consists of the following files:

- `deeplearningModel.py`: Contains the implementation of the deep learning model using TensorFlow and Keras. It includes functions to create data generators for training, validation, and testing, as well as the model architecture itself.
- `utils.py`: Provides utility functions for preprocessing the data, such as splitting the data into training and validation sets, and organizing the test set.
- `Model/`: Directory to store the saved model checkpoints.
- `Data/`: Directory to store the dataset and training data.
    - `Train/`: Contains the training images.
    - `Test/`: Contains the test images.
    - `Test.csv`: CSV file containing the test image labels and paths.
    - `TrainingData/`: Directory to store the preprocessed training data.
        - `Train/`: Contains the preprocessed training images.
        - `Val/`: Contains the preprocessed validation images.

## Usage
1. Download the dataset from Kaggle and place it in the `Data/` directory.
2. Preprocess the data by uncommenting the code in `deeplearningModel.py` to split the training data and organize the test set.
3. Modify the hyperparameters, such as batch size and number of epochs, in `deeplearningModel.py` if desired.
4. Run `deeplearningModel.py` to train the model and save the best model checkpoint.
5. To test the model on the validation and test sets, set the `TEST` flag to `True` in `deeplearningModel.py` and run the script.

Feel free to modify and experiment with the code to improve the model's performance.

## Credits
- GTSRB dataset: [Data](https://www.kaggle.com/meowmeowmeowmeowmeow)
