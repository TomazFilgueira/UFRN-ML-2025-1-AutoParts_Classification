# UFRN-ML-2025-1-Iracing_Classification

#  🏎️🏎️🏎️ Introduction

This project focuses on predicting Simracing cornening phases using machine learning models. It includes data preparation, exploratory data analysis (EDA), model selection, parameter tuning.

---

# 🗂️ Table of Contents
1. [📌 Project Overview](#-project-overview)
2. [📁 Directory Structure](#-directory-structure)
3. [❓ Problem Description](#-problem-description)
4. [🚗 Dataset Generation](#-Dataset-Generation)
5. [🔢 Exploratory Data Analysis-EDA](#Exploratory-Data-Analysis-(EDA))
6. [ Data Pipeline](#-Data-Pipeline:-From-Raw-Images-to-Model-Batches)
7. [🧠 Model Configuration and Training](#-Model-Configuration-and-Training)
8. [🏎️ Iracing Image Classification](#classification-heart-disease)
9. [☝️  Model 1 Conclusion](#model-1-conclusion)
10. [✌️ Model 2 Implementation](#model-2)
11. [⚡Metrics for model 2](#metrics-for-model-2)
12. [☝️✌️Comparing models with different threshold](#comparing-models-with-different-threshold)
13. [🏁Project Conclusion](#project-conclusion)


---

# 📌 Project Overview

Simracing is becoming even more popular nowadays. Even professionais drivers from Formula 1 uses this hardware in order to train before F1 sessions. This project uses deep learning techniques (Pytorch) to classify cornering phases in a virtual racing. 

Key features include:
- 🧹 Data preparation and cleaning.  
- 🔍 Exploratory Data Analysis (EDA).  
- 🧠 Model training, evaluation, and parameter optimization.  
- 🌐 Comparing models
- ☁️ Change hyper parameters

---

# 📁 Directory Structure

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
# ❓ Problem Description

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

# 🚗 Dataset Generation

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
| <br> <img src="https://github.com/TomazFilgueira/UFRN-ML-2025-1-Iracing_Classification/blob/main/images/oulton_park.png" width="400"> | <img src="https://github.com/TomazFilgueira/UFRN-ML-2025-1-Iracing_Classification/blob/main/images/laguna_seca.png" width="400"> |
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

# ⚙️ Data Pipeline: From Raw Images to Model Batches

The journey from raw gameplay screenshots to model-ready data is handled by a comprehensive data pipeline built with PyTorch's `transforms`, `ImageFolder`, and `DataLoader` classes. This process is designed to standardize the images and efficiently feed them to the model.

Our Exploratory Data Analysis (EDA) revealed a significant class imbalance, which can be addressed with techniques like class weighting during training. The primary goal of this pipeline, however, is to first create a clean and consistent data format.

The pipeline consists of two main stages:

**1. Transformation and Standardization**

Every image is processed through a sequential transformation pipeline created with **`transforms.Compose`**. This ensures that every input to the model is uniform.

* **Resizing & Formatting:** Images are resized to a standard **128x128 pixels** using **`transforms.Resize()`** and converted to the **RGB** color space with a `ToImage()` transform.
* **Scaling & Standardization:** Pixel values are scaled to a `[0.0, 1.0]` range using **`ToDtype(torch.float32, scale=True)`**. Then, a more rigorous **standardization** is applied. This is achieved with a custom **`Architecture.make_normalizer()`** function that calculates the dataset's mean and standard deviation to create a final **`transforms.Normalize()`** instance.

**2. Data Loading and Batching**

After the transformation pipeline is defined, the **`ImageFolder`** and **`DataLoader`** classes manage the data flow:

* **`ImageFolder`** automatically discovers the classes from the folder names (`train_dataset_iracing` and `test_dataset_iracing`) and applies the `Compose` pipeline to each image.
* **`DataLoader`** then wraps the dataset to create **mini-batches of 16 images**. For the training set, the data is **shuffled** each epoch to improve model generalization, while the validation loader does not, ensuring consistent evaluation.

# 🧠 Model Configuration and Training

To identify the optimal architecture for this classification task, a systematic, experimental approach was taken. Three distinct model configurations were trained and evaluated, starting with a baseline model and then exploring variations in network width and depth.

## 1. Base Model Configuration

The initial model, which serves as our **base model**, is a custom convolutional neural network (CNN) defined as **`arch.cnn2`**. This architecture was configured with a specific number of feature maps in its convolutional layers.

* **Architecture:** `arch.cnn2`
* **Number of Features:** `num_features = 5`

This configuration provides a benchmark against which all other experiments are measured.

## 2. Experiment 1: Varying Network Width

To analyze the impact of the number of feature maps (network width), the base model was re-trained with two variations, altering only the `num_features` parameter:

* **Simpler Model:** `num_features = 3`
* **More Complex Model:** `num_features = 10`

This experiment helps determine if a wider (more features) or narrower (fewer features) network is better suited for this specific image dataset.

## 3. Experiment 2: Increasing Network Depth

To test the effect of a deeper architecture, a third model was created by modifying the base `arch.cnn2` architecture.

* **Modification:** An additional convolutional block was inserted into the network.

This experiment investigates whether a deeper model, with more layers, can learn more complex hierarchical features from the images and improve classification accuracy.

## Training Process

To ensure a fair comparison, all model configurations were trained using an identical setup, managed by a custom `Architecture` class that encapsulates the model, loss function, and optimizer.

The key components and hyperparameters of the training pipeline are:

* **Reproducibility:** A fixed random seed, **`torch.manual_seed(13)`**, was set to ensure that experiments are reproducible.
* **Loss Function:** The **`nn.CrossEntropyLoss`** was used as the criterion. This loss function is standard for multi-class classification tasks.
* **Optimizer:** The **`Adam`** optimizer was chosen to update the model's weights, with a learning rate set to **`3e-4`**.
* **Regularization:** A dropout rate of **`p=0.3`** was applied within the `CNN2` model architecture to help prevent overfitting.
* **Epochs:** Each model was trained for a total of **8 epochs**.

After training was complete, the final state dictionary of the model was saved to a file (e.g., `base_model_cnn2.pth`) for evaluation and future use. The training and validation loss curves were also plotted to visually assess the model's learning progress.

  
# 📊 Results

This section presents the performance of the four trained model configurations. The goal is to compare their effectiveness and select the best-performing architecture for the final classification task. **Validation Accuracy** was used as the primary metric for comparison.

## Performance Metrics Comparison

The table below summarizes the final validation performance for each of the four experimental models after 8 epochs of training.

| Model Configuration | Validation Accuracy | Notes |
| :--- | :---: | :--- |
| **1. Base Model** (`n_feature=5`) | `65.4%` | The benchmark performance. |
| **2. Short Model** (`n_feature=3`) | `63.9%` | Performance of the model with fewer features. |
| **3. Wider Model** (`n_feature=10`) | `61.2%` | Performance of the model with more features. |
| **4. Deeper Model** (+1 Conv Block) | `62.4%` | Performance of the model with more layers. |

![Confusion Matrix for the Best Model](https://github.com/TomazFilgueira/UFRN-ML-2025-1-Iracing_Classification/blob/main/images/models_comparison.png)

**Analysis:** The results indicate that the **Base Model** achieved the highest validation accuracy, while the XXXXX had the lowest. However, it has to bear in mind that base model has got 65% of accuracy, which is very low!

This suggests that for this task, no models had greate performance. Indicating that we need to evaluate a better CNN Architecture.

### Analysis of the Best Model

Based on the results, the **[Your Best Model's Name, e.g., Deeper Model]** was selected as the final model. A confusion matrix was generated to provide a more detailed look at its performance across the four classes.

![Confusion Matrix for the Best Model](https://github.com/TomazFilgueira/UFRN-ML-2025-1-Iracing_Classification/blob/main/images/cm_base_model.png)

*(**Add your analysis of the confusion matrix here.** For example: "The confusion matrix reveals that the model performs exceptionally well on the `Straight` and `Braking` classes. The primary source of confusion occurs between `Mid Corner` and `Exit Corner`, which is expected due to their visual similarity as the car begins to accelerate.")*

# Project Conclusion

The experimental results demonstrate that the **[Your Best Model's Name]** provides the best balance of accuracy and performance, and it has been selected as the final model for this project.

















   



















