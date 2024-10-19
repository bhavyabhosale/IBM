

\
\# AgriAI

AgriAI is an innovative platform designed to empower farmers and agricultural enthusiasts with cutting-edge AI technologies. Our mission is to enhance agricultural productivity and sustainability through intelligent decision-making tools.

\## Features

- \*\*Crop Recommendation:\*\* Our advanced algorithms analyze soil and environmental parameters to recommend the most suitable crops for your farm.

- \*\*Plant Disease Recognition:\*\* Upload an image of your plant leaves, and our AI will quickly identify any diseases present, providing actionable insights to help you maintain healthy crops and prevent losses.

\## Technologies Used

- \*\*Python:\*\* Main programming language for backend logic.
- \*\*Streamlit:\*\* Framework for building the web application interface.
- \*\*TensorFlow:\*\* For building and using machine learning models for disease recognition and crop recommendation.
- \*\*Google Generative AI:\*\* To fetch detailed information about plant diseases.
- \*\*Pillow:\*\* For image processing and manipulation.
- \*\*NumPy:\*\* For numerical computations.
- \*\*Pickle:\*\* For loading machine learning models.
- \*\*scikit-learn:\*\* For model training, evaluation, and preprocessing.
- \*\*pandas:\*\* For data manipulation and analysis.
- \*\*Seaborn & Matplotlib:\*\* For data visualization.
- \*\*Plotly:\*\* For interactive graph plotting and visualization.
- \*\*pandas\_profiling:\*\* For generating data analysis reports.

\## Installation
1. Install the required packages:
pip install -r requirements.txt
1. Place your pre-trained model and class indices JSON file in the project directory.
1. Run the Streamlit app:
streamlit run app.py
1. Access the application in your web browser at `http://localhost:8501`.
1. Navigate through the dashboard to use the Crop Recommendation and Disease Recognition features.

\## Model Summary

The Plant Disease Prediction project involves building a Convolutional Neural Network (CNN) model to classify plant diseases from image data, using a dataset of 38 different plant disease classes. Below is a summary of the steps involved in your process:

\### 1. Data Import and Preprocessing

- The dataset was imported from the provided Kaggle link, containing over 70,000 training images and 17,000 validation images, categorized into 38 classes.
- TensorFlow was used to preprocess the image data, resizing all images to 128x128 pixels and batching them for training.

\### 2. CNN Model Building

- \*\*Model Architecture\*\*: A Sequential model with five convolutional blocks was designed, progressively increasing the number of filters from 32 to 512.
- \*\*Convolutional Layers\*\*: Multiple Conv2D layers with ReLU activation were added, followed by Max Pooling layers to reduce spatial dimensions.
- \*\*Dropout Layers\*\*: Used after the convolutional layers to prevent overfitting, with a dropout rate of 0.25 and another at the Dense layer with a rate of 0.4.
- \*\*Fully Connected Layers\*\*: A Dense layer with 1500 units was added before the final output layer.
- \*\*Output Layer\*\*: A softmax layer with 38 units was used to output the probabilities for each of the 38 classes.
- \*\*Learning Rate\*\*: To avoid overshooting, the learning rate was set to 0.0001.

\### 3. Model Compilation and Training

- The model was compiled using the Adam optimizer with categorical\_crossentropy loss and accuracy as the evaluation metric.
- The model was trained for 10 epochs, with the training accuracy reaching 97.82% and validation accuracy reaching 94.59% after 10 epochs.

\### 4. Model Evaluation

- Training set accuracy: 97.82%
- Validation set accuracy: 94.59%
- These results indicate that the model performs well on both the training and validation datasets, with no significant signs of overfitting.

\### 5. Saving the Model

- The trained model was saved as `trained\_plant\_disease\_model.keras` for future use.

\### 6. Model History and Visualization

- The training history, including loss and accuracy metrics, was stored in a JSON file (training\_hist.json).
- A plot visualizing the accuracy across epochs was generated to show the improvement in accuracy for both training and validation sets.

\### 7. Evaluation on the Validation Set

- Using the validation set, predictions were generated for all 17,572 images, and the true categories were compared with the predicted ones.
- A confusion matrix was created to analyze model performance, and the classification report showed precision, recall, and F1-score for each class.

\### 8. Metrics

- The precision, recall, and F1-score were calculated for each plant disease category, providing insights into the model's strengths and weaknesses in identifying specific diseases.

