Road Condition Detection in Gilgit
Overview
This project aims to develop an object detection model to identify road conditions and turns on roads in Gilgit. The model is designed to detect right turns, left turns, straight roads, and unexpected conditions like landslides. By combining the power of CNN classification and YOLO object detection, this project leverages deep learning for real-world applications, specifically focusing on enhancing road safety through accurate detection of various road scenarios.

Table of Contents
Project Objectives
Project Phases
Data Collection
Data Labeling
Classifier Training
YOLO Implementation
Model Evaluation
Dataset
Model Training
Evaluation
Results
How to Use
Requirements
Installation
Contributing
Acknowledgments
Project Objectives
Collect a dataset of road images in Gilgit, including various road types and conditions.
Annotate images to label road conditions such as right turn, left turn, straight road, and unexpected conditions.
Train a classification model (CNN) to categorize images into predefined classes.
Implement the YOLO model to detect and classify road turns and conditions accurately.
Evaluate the model’s performance and improve accuracy with fine-tuning.
Project Phases
Phase 1: Data Collection
Students collected a diverse set of road images from Gilgit, including different road types, conditions, and weather scenarios. Each group was required to gather at least 50 images covering:

Right Turn
Left Turn
Straight Road
Unexpected Conditions (e.g., Landslides)
Phase 2: Data Labeling
The collected images were annotated and labeled with the appropriate categories:

Right Turn
Left Turn
Straight Road
Unexpected Condition
Annotation tools like LabelImg were used to make the labeling process efficient.

Phase 3: Classifier Training
A Convolutional Neural Network (CNN) was trained using the labeled dataset to categorize images. Evaluation metrics like accuracy, precision, recall, and F1-score were used to assess the performance of the model.

Phase 4: YOLO Implementation
The YOLO (You Only Look Once) object detection model was implemented to detect and classify road turns and unexpected conditions:

The dataset was preprocessed to be compatible with YOLO format.
YOLO was trained to identify specific road features, enhancing detection accuracy.
Phase 5: Model Evaluation
The YOLO model's performance was evaluated using:

Mean Average Precision (mAP)
Intersection over Union (IoU)
The model was fine-tuned to achieve better accuracy and robustness. Example images were used to showcase the detection capabilities.

Dataset
The dataset consists of road images from Gilgit, captured and labeled by the students. It includes different types of road conditions and turns, aiming to provide a comprehensive dataset for model training.

Model Training
CNN Classifier: A CNN was trained on the labeled dataset to classify images into four categories: right turn, left turn, straight road, and unexpected conditions.
YOLO Object Detection: The YOLO model was trained on the preprocessed dataset to precisely locate and categorize road conditions.
Evaluation
Both the CNN and YOLO models were evaluated using standard performance metrics:

CNN Metrics: Accuracy, Precision, Recall, F1-score
YOLO Metrics: Mean Average Precision (mAP), Intersection over Union (IoU)
Results
The trained models showed good performance in identifying road conditions and turns:

Example images with predictions for each category are included in the project repository.
The YOLO model effectively detected unexpected conditions, making it suitable for practical applications.
Requirements
Python 3.8+
PyTorch
TensorFlow
OpenCV
Roboflow
YOLOv5
Annotation tools (e.g., LabelImg)
 This project involves data collection, labeling, training, and evaluation, providing a hands-on experience in applying computer vision techniques to real-world challenges.
