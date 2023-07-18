# Iris Flower Classification using MATLAB
This project implements a classification model using the K-Nearest Neighbors (KNN) algorithm to classify Iris flowers into different species (setosa, versicolor, and virginica) based on their sepal length, sepal width, petal length, and petal width.

## Dataset
The project utilizes the popular Fisher's Iris dataset, which is available in the MATLAB environment. The dataset contains 150 samples with measurements of four features for three different Iris flower species.

## Features
1. Dataset Preparation:
   - The Fisher's Iris dataset is loaded into the MATLAB environment.
   - The dataset contains measurements of four features: sepal length, sepal width, petal length, and petal width.
   - The features are assigned to the variable `X`, and the corresponding labels (species) are assigned to the variable `y`.

2. Data Preprocessing:
   - Z-score normalization is applied to the feature data (`X`) to ensure that all features have zero mean and unit variance.
   - Mean and standard deviation of the feature data are computed and stored in `X_mean` and `X_std`, respectively.
   - The feature data is normalized using the calculated mean and standard deviation.

3. Model Training:
   - The K-Nearest Neighbors (KNN) algorithm is used for classification.
   - The number of nearest neighbors to consider (`k`) is set to 5.
   - The KNN model is trained using the preprocessed feature data (`X`) and corresponding labels (`y`).

4. Model Evaluation using Cross-validation:
   - The dataset is partitioned into 5 folds for cross-validation.
   - For each fold, the model is trained on the training data and evaluated on the test data.
   - The accuracy of the model is calculated for each fold, and the mean accuracy is computed as the overall performance metric.

5. Confusion Matrix Calculation:
   - The confusion matrix is computed to analyze the performance of the model for each class.
   - True positive, false positive, and false negative values are calculated for each class using the predicted and actual labels.
   - The confusion matrix is displayed, showing the counts of correctly and incorrectly classified instances for each class.

6. Performance Metrics Calculation:
   - Precision, recall, and F1 score are computed for each class using the values from the confusion matrix.
   - Precision represents the proportion of true positive predictions out of all positive predictions.
   - Recall (also known as sensitivity) represents the proportion of true positive predictions out of all actual positive instances.
   - F1 score is the harmonic mean of precision and recall, providing a balanced measure of the model's performance.

7. Model Saving and Loading:
   - The trained model, along with the mean and standard deviation of the feature data, is saved in a MATLAB MAT file (`iris_model.mat`).
   - The saved model can be loaded later for further use or deployment.

8. Visualization:
   - A scatter plot is displayed to visualize the distribution of Iris flowers based on the sepal length and sepal width.
   - Each data point is represented by a marker color corresponding to the actual species label.

9. User Interaction and Prediction:
   - A user interface is provided to interact with the trained model.
   - Users can enter the measurements of sepal length, sepal width, petal length, and petal width.
   - The user input is preprocessed using the stored mean and standard deviation values.
   - The trained model predicts the species of the Iris flower based on the user input.
   - The predicted species is displayed to the user.

## Usage

1. Load the MATLAB environment and run the provided code.

2. The code will preprocess the dataset, train the KNN model, perform model evaluation using cross-validation, and save the trained model.

3. Once the model is trained, a scatter plot of the dataset will be displayed to visualize the distribution of Iris flowers.

4. Users can interact with the trained model through the user interface by entering measurements of sepal length, sepal width, petal length, and petal width.

5. The model will predict the species of the Iris flower based on the user input and display the predicted species.

## Dependencies

- MATLAB (version 2020b or above)



