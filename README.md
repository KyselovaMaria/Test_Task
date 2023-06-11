# Data Science R&D center Winstars

## Test task in Data Science [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection)

## ![img_ship](https://storage.googleapis.com/kaggle-media/competitions/Airbus/ships.jpg)

## Project Structure:

```
├── LICENSE
├───README.md                           <- The top-level README for developers using this project
|
├───eda                                 <- Eda
│   └───eda.ipynb                       <- Eda notebook
|
├───train-infer                         <- Contain kaggle notebooks with training and inferencing
│   ├───inference.ipynb                 <- Model inference results
│   ├───train.ipynb                     <- Model training results
|
├───src                                 <- Code
|   ├───inference.py                    <- Model inference
|   ├───train.py                        <- Model training
|
└───requirements.txt                    <- The requirements file for reproducing the analysis environment, e.g.
                                           generated with `pip freeze > requirements.txt`
```

## Solution
In my work I have shown that I am familiar with CNN and related topics. For better result can be used deeper NN and tuning hyperparameters - it takes more time.
Accuracy of prediction about 99%.

## Code for training a model
# Let's go through the explanation step by step:

1. Data Connection: The code begins by importing necessary libraries and setting up the paths for the training and testing data directories. It also reads the training ship segmentations data from a CSV file.

2. Constants: Several constants are defined, including image dimensions, target dimensions, number of epochs, batch size, and other flags for faster development.

3. Run-length Encoding and Decoding: Two functions, `rle_encode` and `rle_decode`, are defined to encode and decode masks using the run-length encoding technique. These functions are used to convert masks from a binary image format to a string representation and vice versa.

4. Dice Coefficient: The `dice_coef` function calculates the Dice coefficient, which is a metric used to evaluate the performance of the segmentation model. It measures the similarity between the predicted and true masks.

5. Model Architecture: The code defines the architecture of the segmentation model using the Keras functional API. It consists of a series of convolutional and pooling layers for downsampling and upsampling, with skip connections to retain spatial information.

6. Model Compilation: The model is compiled with the RMSprop optimizer and binary cross-entropy loss function. The Dice coefficient is also used as a metric to monitor the model's performance during training.

7. Data Split: The training dataset is split into training and validation sets using the `train_test_split` function from scikit-learn.

8. Image and Mask Generation: Several functions are defined to generate images and masks in batches for training and validation. These functions read the images and masks, resize them to the target dimensions, and preprocess them for training.

9. Model Training: The model is trained using the `fit_generator` function, which takes the data generators, steps per epoch, validation data, validation steps, and other parameters. The best model weights are saved during training using the `ModelCheckpoint` callback.

10. Test Image Generation: A function is defined to generate test images for prediction. It reads the test images, resizes them to the target dimensions, and preprocesses them.

11. Making Predictions: The trained model is used to make predictions on the test images. The `predict_generator` function is used to generate predicted masks. The predicted masks are then visualized alongside the original images using matplotlib.

12. Training Progress Plot: Finally, a plot is generated to visualize the training progress over epochs. It shows the Dice coefficient values for both training and validation sets.

Overall, this code sets up and trains a U-Net-based model for image segmentation using ship segmentation data. It demonstrates the steps involved in loading and preprocessing the data, defining the model architecture, training the model, and making predictions.
