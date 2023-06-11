# Data Science R&D center Winstars

## Test task in Data Science [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection)

## ![img_ship](https://storage.googleapis.com/kaggle-media/competitions/Airbus/ships.jpg)

## Project Structure:

```
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


## Code that performs exploratory data analysis (EDA) on a dataset

1. Library Imports: The necessary libraries are imported, including numpy, pandas, matplotlib.pyplot, pathlib, tensorflow.keras.utils, skimage.io, seaborn, cv2, and os.

2. Checking Working Directory: The code prints the contents of the directory specified by the path "C:\Users\mariy\Desktop\test" using os.listdir. This is done to verify the current working directory.

3. Loading Data: The code loads the dataset from the CSV file "train_ship_segmentations_v2.csv" using pandas' read_csv function. The dataset is stored in the DataFrame called `df`.

4. Examining Data: The code displays the first few rows of the DataFrame `df` using the `head()` method to get an initial overview of the data.

5. Dataset Information: The code uses the `shape` attribute of the DataFrame to print the number of rows and columns in the dataset, and `info()` method to display detailed information about the DataFrame, including the column names, non-null count, and data types.

6. Visualizing Images: The code visualizes a subset of images from the dataset. It selects a random sample of images with ship annotations (`EncodedPixels` column is not NaN) and displays them in a grid using matplotlib subplots. The images are read using `imread` from skimage.io and shown using `imshow`. Two sets of images are displayed: one with ship annotations and another without ship annotations.

7. Analyzing Ship Existence: The code adds a new column to the DataFrame called 'Exist_ship' that indicates whether a ship is present in an image or not. It uses the `EncodedPixels` column to check for NaN values. The `apply` method is used with a lambda function to convert NaN values to 0 and non-NaN values to 1.

8. Ship Distribution: The code generates a pie chart to visualize the distribution of images with and without ships using the 'Exist_ship' column.

9. Ship Count Distribution: The code calculates the number of ships in each image using the `value_counts()` method on the 'ImageId' column and plots a bar chart to show the distribution of ship counts.

10. Ship Size Distribution: The code calculates the size (number of pixels) of each ship annotation using the 'EncodedPixels' column. It splits the run-length-encoded strings, counts the number of elements in each split, and creates a histogram using seaborn's `histplot` function to visualize the distribution of ship sizes.

11. Image and Mask Visualization: The code defines a function `rle_decode` to decode the run-length-encoded masks and a function `masks_as_color` to draw ships in different colors. It then randomly selects an image from the dataset and displays the original image, the combined mask image, and the overlay of the mask on the original image.

12. Pairwise Image Comparison: The code randomly selects four images from the dataset and displays each image along with its corresponding mask. It calls the `raw_prediction` function to load and preprocess the image and uses the `masks_as_color` function to create the color mask.


## Code for training a model

Let's go through the explanation step by step:

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


## Code for model inference

The explanation step by step:

1. Library Imports: The necessary libraries are imported, including TensorFlow, Keras, pandas, numpy, skimage, tqdm, matplotlib, and os.

2. Directory and Model Paths: The paths for the training directory, test directory, and model file are defined using the TRAIN_DIR, TEST_DIR, and MODEL_DIR variables, respectively.

3. Image and Target Dimensions: The code defines the image and target dimensions using IMG_WIDTH, IMG_HEIGHT, TARGET_WIDTH, and TARGET_HEIGHT variables.

4. Constants: Other constants are defined, including the number of epochs, batch size, image shape, and flags for faster development.

5. Dice Coefficient: The dice_coef function is defined, which calculates the Dice coefficient. This function is used as a custom metric during model training and is needed during inference.

6. Model Loading: The pre-trained model is loaded using the load_model function from Keras. The custom_objects parameter is used to pass the dice_coef function to ensure proper loading of the model.

7. Test Image Preparation: The code reads the sample submission file using pandas to get the test image IDs. Then, it initializes an empty array X_test to store the resized test images. It iterates over the test image IDs, reads each image using skimage's imread function, resizes it to the target dimensions using skimage's resize function, and appends it to the X_test array.

8. Model Prediction: The loaded model is used to make predictions on the test images. The model.predict function is called on the X_test array to obtain the predicted masks for each test image. The result is stored in the preds_test array.

9. Visualization: The code sets a threshold value and prepares to visualize the test images alongside their predicted masks. It creates a grid of subplots using matplotlib and iterates over a subset of predicted masks. For each predicted mask, it plots the corresponding test image, overlays the predicted mask on top of the image, and sets the alpha value to make the mask semi-transparent. The resulting visualization is shown without axis labels.

This code essentially loads a pre-trained model, performs inference on test images, and visualizes the predictions alongside the original images.
