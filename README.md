# Cork Oak Tree Detection and Biomass Estimation

## Internship Project with MUST University

This repository contains the code and documentation for a computer vision and research 2 months internship project focused on detecting cork oak trees and estimating their biomass using deep learning techniques.

## Project Overview

The goal of this project is to develop an advanced system that can accurately detect cork oak trees in aerial or satellite imagery and estimate their biomass. This project utilizes the DeepForest library and custom-trained models to achieve these objectives, contributing to more efficient forest management and environmental monitoring.

### Significance

Cork oak forests play a crucial role in Mediterranean ecosystems and economies. Accurate detection and biomass estimation of these trees can:
- Aid in sustainable forest management
- Support conservation efforts
- Contribute to carbon sequestration studies
- Assist in monitoring forest health and biodiversity

## Repository Structure
#### InternshipProject/
│
##### ├── Posttraining_Images/
##### ├── Predicted_Images/
##### ├── Pretraining_Images/
##### ├── Test_Images/
##### ├── Validation_Images/
##### ├── lightning_logs/
│
##### ├── Validation.csv
##### ├── annotation.csv
##### ├── annotations.csv
##### ├── main.ipynb
##### ├── training.ipynb
##### └── validations.CSV

## Methodology

### 1. Data Collection and Preparation

- **Image Acquisition**: Utilized Google Earth Pro to capture high-resolution images of cork oak forests.
- **Annotation**: 
  - Employed VGG Image Annotator (VIA) for precise tree labeling.
  - Focused on identifying individual cork oak trees in diverse landscapes.
- **Data Conversion**: 
  - Developed custom scripts to convert VIA annotations to DeepForest compatible format.
  - Ensured proper formatting of bounding box coordinates and labels.

### 2. Model Architecture and Training

- **Base Model**: Leveraged DeepForest, a deep learning library built on PyTorch, specializing in tree crown detection.
- **Fine-tuning Process**: 
  - Adapted the pre-trained DeepForest model to specifically recognize cork oak trees.
  - Implemented in `training.ipynb` with detailed steps for reproducibility.
- **Training Parameters**:
  - Epochs: 10 (adjustable based on performance)
  - Utilized custom loss functions and optimization techniques for improved accuracy.

### 3. Prediction and Evaluation

- **Model Application**: 
  - Applied the trained model to test images, showcasing its ability to detect cork oak trees in new, unseen data.
- **Visualization**: 
  - Developed functions to visually represent detected trees with bounding boxes.
  - Generated annotated images for easy interpretation of results.
- **Metrics Calculation**: 
  - Implemented a robust evaluation pipeline to assess model performance.
  - Calculated key metrics including accuracy, precision, recall, and F1 score.

### 4. Biomass Estimation

- **Algorithm Development**: 
  - Created a custom function to estimate tree biomass based on detected canopy area.
  - Utilized allometric equations specific to cork oak trees for accurate estimation.
- **Integration**: 
  - Seamlessly integrated biomass calculation with the detection pipeline.
  - Provided total biomass estimates for entire images, useful for forest inventory purposes.

Note: The `deepforest_model.pth` file is not included in the repository due to size limitations. Please see the "Model Weights" section for download instructions.

## Key Files and Their Functions

- `main.ipynb`: 
  - Core script for model inference and result visualization.
  - Contains functions for loading images, making predictions, and displaying results.
- `training.ipynb`: 
  - Comprehensive notebook detailing the model training process.
  - Includes data loading, model configuration, training loops, and model saving.

## Model Weights

The trained model weights (`deepforest_model.pth`) are too large to be included directly in this repository. You can download them from the following Google Drive link:

[Download deepforest_model.pth] 
([https://drive.google.com/file/d/1fEzPdyQT-9otiOFCyNJ7KBYkzhbrB7C1/view?usp=sharing])

After downloading, place the `deepforest_model.pth` file in the root directory of the project before running the inference scripts.

## How to Use This Repository

1. Clone the repository: `git clone [repository URL]`
2. Download the `deepforest_model.pth` file from the Google Drive link provided above and place it in the project's root directory.
3. Follow the notebooks in order: `training.ipynb` for model training, `main.ipynb` for inference and evaluation.
## Results and Performance

The project achieved impressive results on the validation set:

- Accuracy: 0.8431
- Precision: 0.86
- Recall: 0.9773
- F1 Score: 0.9149

These metrics indicate strong performance in detecting cork oak trees, with particularly high recall suggesting the model rarely misses trees present in the images.

### Visualization Examples

![image](https://github.com/user-attachments/assets/15ba1392-2025-45f2-938a-b775b857f25f) ![image](https://github.com/user-attachments/assets/38938a32-8451-4742-a473-422eba51d894)


## Challenges and Solutions

- **Data Variability**: Addressed the challenge of diverse tree appearances and landscapes by incorporating a wide range of images in the training set.
- **Annotation Precision**: Developed strict annotation guidelines to ensure consistency across the dataset.
- **Model Tuning**: Experimented with various hyperparameters and data augmentation techniques to optimize model performance.

## Future Work and Potential Improvements

1. **Dataset Expansion**: 
   - Incorporate images from different seasons and geographical locations to improve model generalization.
   - Consider adding multispectral imagery for enhanced feature detection.

2. **Model Optimization**: 
   - Explore advanced architectures like Mask R-CNN for instance segmentation.
   - Implement ensemble methods to boost accuracy and robustness.

3. **Integration and Deployment**:
   - Develop a user-friendly interface for non-technical users.
   - Create APIs for easy integration with existing forest management systems.

4. **Extended Analysis**:
   - Incorporate temporal analysis to study forest growth and changes over time.
   - Explore the possibility of species differentiation within mixed forests.

## How to Use This Repository

1. Clone the repository: `git clone https://github.com/maram77/Summer_Internship_2024.git`
2. Follow the notebooks in order: `training.ipynb` for model training, `main.ipynb` for inference and evaluation.

