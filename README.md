# CS6964-malaria_detector

# Malaria Cell Image Classification  
Binary Classification of Parasitized vs. Uninfected Thin Blood Smear Images

This project builds a full machine learning pipeline to classify malaria-infected red blood cells using traditional algorithms and a custom convolutional neural network (CNN).

---

## Dataset
- Source: Thin blood smear cell images (Parasitized / Uninfected)
- https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-project.html
- ~27,000 PNG images
- Varying dimensions and quality (all normalized during preprocessing)

---

## Pipeline Overview

### 1. **Image Preprocessing**
- Read raw PNGs using **OpenCV**
- Two preprocessing tracks:
  - **Grayscale (50×50)** for traditional ML models  
  - **RGB Color (128×128)** for CNN models
- Steps:
  - Resize  
  - Normalize pixel values  
  - Label encoding (infected = 1, uninfected = 0)
- Saved as NumPy arrays for fast reload:
  - `Cells_bw.npy`, `Labels_bw.npy`
  - `Cells_color.npy`, `Labels_color.npy`

---

## Models Trained

### **Classical ML (on grayscale flattened inputs)**
- **Random Forest Classifier**
  - Default model + tuned model (n_estimators, depth, features)
- **Logistic Regression**
  - Default model + tuned model (C, solver, penalty)
- **Support Vector Machine (RBF Kernel)**
  - Default model + tuned model (C=5, gamma=0.1)

### **Deep Learning (on color images)**
- **Custom CNN (Keras/TensorFlow)**
  - Conv–Pool stacks
  - Dense layer classifier
  - Tuned version with deeper architecture

---

## Evaluation Metrics
For each model:
- **Accuracy**
- **F1-score**
- **Training time**
- **Confusion Matrix**

All metrics compiled into a Pandas dataframe; `eval_df` for easy comparison.

---

## Skills Demonstrated
- Multiclass medical image preprocessing  
- Feature scaling + normalization  
- Train–test splitting & leakage prevention  
- Hyperparameter tuning (GridSearchCV)  
- CNN architecture design in TensorFlow/Keras  
- ML model comparison and reporting best model  
- Medical ML considerations (balanced data, F1 emphasis)  
