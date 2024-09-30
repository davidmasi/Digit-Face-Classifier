# Digit-Face-Classifier
This project implements machine learning classifiers to recognize digits and faces from image data. Using Naive Bayes and Perceptron algorithms, the classifier is trained on feature-extracted data for accurate image classification.

Features:
- Naive Bayes: A probabilistic model based on the Bayes theorem for classification.
- Perceptron: A linear classifier that learns the weights for each feature during training.
- Flexible Training: Classifiers can be trained on different datasets (faces or digits) with customizable data sizes.
- Evaluation Statistics: The project collects detailed accuracy, error rates, and timing statistics for model performance evaluation.

Project Structure:
- models/: Contains the implementations of Naive Bayes, Perceptron, and Neural Network classifiers.
- helpers/: Includes various utility functions like feature extraction, data loading, and performance metrics.
- data/: Stores the training and testing datasets for digits and faces.
- Main.py: Main script to run the classifiers and collect statistics.

Installation:
1. Clone this repository:
git clone https://github.com/your-username/face-digit-classifier.git
cd face-digit-classifier
2. Install dependencies:
pip install -r requirements.txt

Usage:
Run the classifier with different modes:
python Main.py

Options:
- --classifier: Specify which classifier to use (naivebayes, perceptron, neuralnetwork).
- --data: Choose the dataset to train on (digit, face).
- --iterations: Number of training iterations.
- --debug: Enable debug mode for smaller dataset sizes.

Example:
python Main.py --classifier naivebayes --data digit --iterations 5

Classifiers:
- Naive Bayes: This classifier calculates the likelihood of each label given the features of the image, using the assumption of feature independence.

- Perceptron: A linear classifier that iteratively updates the weight of each feature based on classification errors during training.
  
Data: 
- The project can be run on datasets of handwritten digits and facial images. You can train the model on either dataset using the --data flag.

Statistics:
- The project records accuracy, error rates, and training times for each run, making it easy to evaluate classifier performance across different configurations.

License:
This project is licensed under the MIT License.
