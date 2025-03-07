# Infyma AI Hackathon 2025
# Diabetes Detection through Retinopathy

## 🏆 Introduction

Welcome to the **AI/ML Hackathon on Diabetes Detection through Retinopathy**! This competition challenges participants to develop machine learning models for detecting diabetic retinopathy from retinal images. The goal is to create an accurate and efficient AI-based diagnostic tool to assist in early detection and treatment planning.

## 📌 Problem Statement

Diabetic Retinopathy (DR) is a complication of diabetes that affects the eyes and can lead to vision loss. Your task is to develop an AI model that classifies retinal images into different DR severity levels.

## 📂 Dataset Details

- **Categories:**
  - No DR (Healthy)
  - Mild DR
  - Moderate DR
  - Severe DR
  - Proliferative DR
- **Formats:** JPEG/PNG images with structured CSV metadata.

## 📜 Rules & Guidelines

- **Team Size:** 2-4 members per team.
- **Duration:** 24-48 hours.
- **Allowed Frameworks:** TensorFlow, PyTorch, OpenCV, FastAI, Scikit-Learn.
- **Computational Resources:** Google Colab, Kaggle Kernels, or personal GPU setups.
- **Originality:** Plagiarism or unauthorized use of existing solutions will lead to disqualification.
- **Submission Format:**
  - Jupyter Notebook / Python Script (.ipynb or .py)
  - Model Weights (.h5 or .pt)
  - A short **report** explaining methodology.

## 📊 Evaluation Criteria

Your model will be evaluated based on:

- **Accuracy & Performance (40%)**: F1-score, Precision, Recall.
- **Explainability (20%)**: Use of visualization techniques (Grad-CAM, SHAP).
- **Computational Efficiency (20%)**: Inference speed, optimization techniques.
- **Innovation (20%)**: Novel architectures, hybrid models.

## 🛠 Technical Instructions

### 🔹 File Structure

```plaintext
├── team_name/
    ├── model/
    │   ├── trained_model.h5 or .pt
    ├── notebooks/
    │   ├── model_training.ipynb
    ├── report.pdf (explaining approach and results)
    ├── README.md (instructions for running the model)
    ├── requirements.txt (dependencies)
```

### 🔹 Install Dependencies

Run the following command to install dependencies:
```bash
pip install -r requirements.txt
```
Recommended libraries: TensorFlow, PyTorch, OpenCV, Scikit-learn, Matplotlib, Pandas, NumPy.

### 🔹 Prepare the Dataset

- **Preprocessing:**
  - Resize images to match model input requirements.
  - Apply normalization and augmentation (rotation, flipping, contrast adjustments).
- **Model Training:**
  - Use transfer learning or train from scratch with CNNs, Transformers, or hybrid models.
  - Implement early stopping, dropout, batch normalization.
  - Optimize performance with hyperparameter tuning (GridSearch, RandomizedSearch).
- **Evaluation & Explainability:**
  - Use proper metrics (F1-score, Precision, Recall).
  - Apply Grad-CAM or SHAP for model interpretability.
- **Deployment (Optional):**
  - Demonstrate inference using Flask, FastAPI, or Streamlit.
  - Optimize inference using model quantization.

## 🤖 Model Selection Hints

💡 Choose the right architecture for better results:

### **Beginner-Friendly CNNs**
- **VGG-16 / VGG-19**: Simple but computationally heavy.
- **ResNet-50**: Good baseline with residual connections.

### **Optimized CNNs for Medical Images**
- **EfficientNet-B3/B4**: Balances accuracy and efficiency.
- **DenseNet-121**: Prevents feature loss, ideal for small datasets.

### **Transformers for Advanced Users**
- **Vision Transformer (ViT)**: Works well with large datasets.
- **Swin Transformer**: Efficient for high-resolution medical images.

### **Explainability & Segmentation Models**
- **InceptionV3 + Grad-CAM**: Visualizes affected areas.
- **U-Net + ResNet**: Ideal for detecting affected regions.

### **Advanced Approaches**
- **Hybrid CNN-Transformer Models**: Combines CNNs and self-attention mechanisms.
- **Ensemble Learning**: Multiple models for improved accuracy.

### **Additional Architecture Options**
- **VGG-16**: Simple yet effective.
- **InceptionV3**: Captures spatial hierarchies.
- **MobileNet**: Lightweight, optimized for mobile devices.
- **GoogleNet (Inception)**: Lower computational cost.

## 📥 Submission Instructions

1. Submit by directly pushing it on your Github accounts. Add your Github account link in the submission form later provided by the organizing team.
3. Ensure your **code is well-documented** and dependencies are listed in `requirements.txt`.


## 📧 Support & Queries

For any queries, contact **zayan.rashidrana@studentambassadors.com**. Mentors will be available for guidance throughout the hackathon.

---

🚀 **Good Luck & Happy Coding!**

