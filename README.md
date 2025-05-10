# Card Reader - Playing Card Classifier 

This project uses deep learning to classify playing cards into 14 classes based on images. It leverages transfer learning with a pretrained convolutional neural network (CNN), applies data augmentation, and includes visualizations of model performance and feature maps.

## Dataset
The model was trained on a Kaggle dataset containing labeled images of playing cards (number and face cards). The dataset is preprocessed into training and validation sets with class balancing and augmentation.

## Architecture
- Pretrained base: VGG16 (ImageNet weights)
- Classification head: Flatten → Dense → Dropout → Dense (Softmax)
- Final layer: 14-class softmax output
- Loss function: Categorical Crossentropy
- Optimizer: Adam

## Evaluation
The notebook performs:
- 5-fold cross-validation
- Accuracy and loss tracking over epochs
- Classification report (precision, recall, F1-score)
- Confusion matrix heatmap

## Feature Visualization
Includes feature map visualizations for face and number cards to analyze what the model is learning.

## Requirements
- Python 3.x
- TensorFlow / Keras
- NumPy
- scikit-learn
- matplotlib
- seaborn

## Usage
Clone this repo and run the notebook:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
jupyter notebook Card_Reader.ipynb
