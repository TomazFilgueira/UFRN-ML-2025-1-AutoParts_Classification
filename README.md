# UFRN-ML-2025-1-Iracing_Classification

# ❤️ XXXXX Introduction

This project focuses on predicting Simracing cornening phases using machine learning models. It includes data preparation, exploratory data analysis (EDA), model selection, parameter tuning.

---

## 🗂️ Table of Contents
1. [📌 Project Overview](#-project-overview)
2. [📁 Directory Structure](#-directory-structure)
3. [❓ Problem Description](#-problem-description)
4. [🔢 Exploratory Data Analysis-EDA](#eda-checkpoints)
5. [🏎️ Iracing Image Classification](#classification-heart-disease)
6. [☝️  Model 1 Conclusion](#model-1-conclusion)
7. [✌️ Model 2 Implementation](#model-2)
8. [⚡Metrics for model 2](#metrics-for-model-2)
9. [☝️✌️Comparing models with different threshold](#comparing-models-with-different-threshold)
10. [🏁Project Conclusion](#project-conclusion)


---

## 📌 Project Overview

Simracing is becoming even more popular nowadays. Even professionais drivers from Formula 1 uses this hardware in order to train before F1 sessions. This project uses deep learning techniques (Pytorch) to classify cornering phases in a virtual racing. 

Key features include:
- 🧹 Data preparation and cleaning.  
- 🔍 Exploratory Data Analysis (EDA).  
- 🧠 Model training, evaluation, and parameter optimization.  
- 🌐 Comparing models
- ☁️ Change hyper parameters

---

## 📁 Directory Structure

```plaintext
Heart-Disease-Classification/
│
├── data/                          	# Contains the dataset
├── data_preparation               	# Scripts for creating pytorch tensors
├── model_configuration            	# configure sequencial linear models
├── model_training                 	# script for train model    
├── images                         	# Contains images generates by model ouputs
├── EDA.ipynp                       # Jupyter notebook with Exploratory Data Analysis
├── Iracing_classification.ipynb  # Jupyter notebook using model 1 and 2 for classification
└── README.md                      	# Project description and instructions
```

---
## ❓ Problem Description

Driving analysis in simulators is crucial for performance improvement. This project was born from a passion for sim racing and aims to create an AI tool capable of automatically identifying which cornering phase a driver is in, based solely on the game's image.

The four classified phases are:

1.  **Braking:** The moment of approaching and decelerating before the corner.
2.  **Mid-corner:** The main cornering phase. This category combines the initial steering input (`Turn-in`) and passing the tightest point of the corner (`Apex`). The car is rotating and navigating the turn at its lowest speed.
3.  **Corner Exit:** The moment the driver unwinds the steering wheel and resumes acceleration towards the next straight.
4.  **Straight:** Driving on a straight section of the track, typically under full throttle with minimal steering input, connecting one corner to the next.

Below are image examples for each of the 4 classes used to train the model.

<p align="center">
  <img src="https://github.com/TomazFilgueira/UFRN-ML-2025-1-Iracing_Classification/blob/main/train_dataset_iracing/freada/freada%20(1000).jpg" width="24%" alt="Braking Example">
  <img src="https://github.com/TomazFilgueira/UFRN-ML-2025-1-Iracing_Classification/blob/main/train_dataset_iracing/curva_apex/curva%20(1002).jpg" width="24%" alt="MidCorner Example">
  <img src="https://github.com/TomazFilgueira/UFRN-ML-2025-1-Iracing_Classification/blob/main/train_dataset_iracing/saida_curva/saida_curva%20(1006).jpg" width="24%" alt="Corner Exit Example">
  <img src="https://github.com/TomazFilgueira/UFRN-ML-2025-1-Iracing_Classification/blob/main/train_dataset_iracing/reta/reta%20(1002).jpg" width="24%" alt="Straight Example">  
</p>
<p align="center">
  <em>From left to right: Braking, Mid corner,Corner Exit and Straight.</em>
</p>

## 🚗 Dataset Generation

The foundation of this project is a custom dataset meticulously generated using the iRacing simulation software. The primary goal was to create a diverse and robust collection of images that captures a wide variety of driving scenarios. To achieve this, a carefully selected combination of vehicles and circuits was used, ensuring the model is exposed to different visual cues, cockpit layouts, lighting conditions, and track characteristics.

### Vehicles Used

Two cars with distinct physics and visual profiles were chosen to enhance the dataset's variety, providing the model with different reference points for the driver's field of view.

| <p align="center">Global Mazda MX-5 Cup</p> | <p align="center">Toyota GR86</p> |
| :---: | :---: |
| <br> <img src="https://github.com/TomazFilgueira/UFRN-ML-2025-1-Iracing_Classification/blob/main/images/global_mazda.png" width="400"> | <br> <img src="https://github.com/TomazFilgueira/UFRN-ML-2025-1-Iracing_Classification/blob/main/images/toyota_gr86.png" width="400"> |
| A momentum-based car known for its predictable handling, widely used in introductory racing series. | A modern sports car offering a different handling challenge and a more contemporary cockpit environment. |

### Circuits Raced

Data was collected on two internationally renowned circuits, each presenting unique corner types, elevation changes, and environmental textures.

| <p align="center">Oulton Park Circuit</p> | <p align="center">WeatherTech Raceway Laguna Seca</p> |
| :---: | :---: |
| <br> <img src="https://github.com/TomazFilgueira/UFRN-ML-2025-1-Iracing_Classification/blob/main/images/oulton_park.png" width="400">`* | <img src="https://github.com/TomazFilgueira/UFRN-ML-2025-1-Iracing_Classification/blob/main/images/laguna_seca.png" width="400"> |
| A narrow, undulating track in the UK, famous for its blind crests and technically demanding sections. | A classic American circuit in California, known for its iconic "Corkscrew" chicane and significant elevation changes. |

### Data Capture and Labeling Methodology

The dataset was built using a custom, two-step process to ensure both efficiency and accuracy.

1.  **Automated Image Capture**: A Python script, **`printscreen_generator.ipynb`**, was executed during gameplay on the circuits and with the cars mentioned above. This script was configured to automatically capture a high-resolution screenshot every one second, saving all raw images into a temporary folder named `print screen folder`.

2.  **Manual Classification Tool**: After the capture phase, a second Python script was used to facilitate a streamlined manual labeling process. This tool would open each image from the `print screen folder` individually, allowing for manual categorization. After an image was assigned to one of the four driving phases (Braking, Mid Corner , Corner Exit, or Straight), the script would automatically move the image file into its respective final directory within the dataset structure.

This two-step methodology allowed for the rapid collection of thousands of images, which were then carefully and accurately labeled by hand to create a high-quality dataset for training the model.

# 📊 Exploratory Data Analysis (EDA)
Before training the model, an Exploratory Data Analysis (EDA) was performed to understand the dataset's composition. This step is crucial for identifying characteristics like class imbalance, which can significantly influence model training and performance. The key findings are detailed below.

## Overall Class Distribution
First, we analyzed the total number of images for each of the four classes across the entire dataset. The distribution is clearly imbalanced.

As the graph shows, the Straight class is heavily over-represented compared to the others, containing more than double the images of the least-represented class, Braking. This imbalance is expected, as a car spends more time on straights than in specific braking zones, but it's a critical factor to consider. A model trained on this data might develop a bias towards predicting the majority class.

![Image Distribution per Class](https://github.com/TomazFilgueira/UFRN-ML-2025-1-Iracing_Classification/blob/main/images/classes_distribution.png)

## Train vs. Validation Set Comparison
To ensure that our validation results are a reliable measure of performance, we verified that the class distribution is consistent between the training and validation sets.

The percentage distribution is remarkably similar across both sets. For example, the Braking class constitutes approximately 16% of the training set and 14% of the validation set, while the Straight class accounts for roughly 42% and 50%, respectively. This consistency is excellent, as it confirms that the validation set is a representative sample of the training data.

![Image Distribution per Class](https://github.com/TomazFilgueira/UFRN-ML-2025-1-Iracing_Classification/blob/main/images/classes_distribution_percentages.png)

## Key Takeaways from EDA

**Imbalanced Dataset**: The dataset is significantly imbalanced, with a majority of images belonging to the Straight class.

**Representative Validation Set**: The proportional distribution of classes is consistent between the training and validation splits, ensuring that our evaluation metrics will be reliable.

**Modeling Strategy**: The class imbalance must be addressed during the modeling phase. Techniques such as using class weights in the loss function or applying data augmentation strategies like oversampling the minority classes should be considered to prevent model bias and improve performance on less-represented classes.
















   



















