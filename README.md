# Weather-Forecast-Predictor
This project is a Weather Forecast Predictor that uses a deep learning model to classify weather conditions based on images. The model is built with both a custom Convolutional Neural Network (CNN) and a transfer learning approach using MobileNetV2, achieving high accuracy. The project also includes a Gradio interface to allow easy interaction with the model, making it user-friendly and accessible for predictions.

# Project Overview
The Weather Forecast Predictor project aims to classify images into various weather conditions, such as sunny, rainy, foggy, cloudy, and more. The initial model was a custom CNN that provided a strong foundation, but later, MobileNetV2 was implemented through transfer learning to enhance performance and accuracy, reaching a test accuracy of 93%.

# Features
+ Image-based Weather Prediction: Predicts weather conditions based on an input image.
+ Custom Model and Transfer Learning: Comparison of custom CNN and MobileNetV2 for improved accuracy.
+ Interactive Gradio Interface: Provides a user-friendly interface for predictions.
+Deployment Ready: Compatible with deployment on platforms such as Hugging Face Spaces.

# Technologies Used
+ Programming Languages: Python
+ Deep Learning Libraries: TensorFlow, Keras
+ Frameworks: Gradio for interface, Docker for containerization
+ Model Architecture: Convolutional Neural Network (CNN), MobileNetV2
+ Data Handling: NumPy, Pandas
+ Visualization: Matplotlib, Seaborn
  
# Data Preprocessing
The dataset used for this project consists of weather images labeled into different weather categories. Preprocessing steps include:
+ Image Resizing: All images resized to 150x150 pixels.
+ Data Augmentation: Augmentation techniques like rotation, flipping, and zooming to increase dataset diversity.
+ Normalization: Pixel values normalized to improve model convergence.

# Model Training
+ Custom CNN: Initially, a custom CNN model was trained from scratch, achieving decent accuracy.
+ MobileNetV2 Transfer Learning: To improve performance, transfer learning with MobileNetV2 was applied, resulting in a significant accuracy boost.
+ Evaluation Metrics: Accuracy was used to evaluate model performance. MobileNetV2 achieved a test accuracy of 93%.

# Performance
+ The final MobileNetV2 model achieved:

+ Train Accuracy: ~95%
+ Validation Accuracy: ~92%
+ Test Accuracy: 93%

# Gradio Interface
A Gradio Interface was built to make the model accessible and interactive. Users can upload an image, and the model will predict the weather condition based on the image. The Gradio interface provides a seamless way to test the model in real-time.

# Installation
To run this project locally, follow these steps:

Clone the repository:
+ git clone https://github.com/yourusername/weather-forecast-predictor.git
cd weather-forecast-predictor
## Install the required dependencies:

+ pip install -r requirements.txt
+ Download the dataset ((https://www.kaggle.com/datasets/vijaygiitk/multiclass-weather-dataset)) and place it in the data/ directory.

# Usage
+ Run the weather_forecast_predictor.ipynb notebook to train the model.
+ Launch the Gradio interface to start making predictions:
+ gr.Interface(fn=predict_weather, inputs="image", outputs="text").launch()
  
# Results
The Weather Forecast Predictor provides reliable weather condition classification based on images. The high accuracy of the MobileNetV2 model makes it suitable for real-world applications, including automated weather stations, photography settings, and more.

# Future Work
+ Additional Weather Conditions: Expand the model to predict more granular weather categories.
+ Real-time Video Analysis: Implement real-time analysis on video feeds.
+ Model Optimization: Further fine-tune MobileNetV2 or experiment with other architectures to improve performance.
  
# Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.
