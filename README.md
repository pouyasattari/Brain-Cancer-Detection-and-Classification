# 🧠 Brain Cancer Detection from MRI Scans  
*A Machine Learning and Deep Learning Approach for Tumor Classification*  

<img src="https://github.com/user-attachments/assets/efe347fa-6bec-45e0-9ed2-bfb03e310cd6" alt="Brain Tumor Classification" width="500">



## **📌 Overview**  
This project focuses on the detection and classification of brain tumors (Glioma, Meningioma, Pituitary Tumor, and No Tumor) from **MRI scans** using **Machine Learning & Deep Learning techniques**. The goal is to enhance diagnostic accuracy and assist medical professionals by automating tumor classification.  

## **⚙️ Tech Stack**  
**Languages & Libraries:**  
`Python, TensorFlow, OpenCV, Scikit-Learn, PCA, NMF, LASSO, Decision Tree, Random Forest, SVC, CNN (ResNet50, VGG16), Vision Transformer, Autoencoder, Isolation Forest`  

---  

## **🚀 Key Steps & Methods**  

### **1️⃣ Data Preprocessing 🏷️**  
✔ Converted MRI images to grayscale and resized to **224x224 pixels**  
✔ Applied **Principal Component Analysis (PCA)** and **Non-Negative Matrix Factorization (NMF)** for dimensionality reduction  
✔ Used **VGG16 preprocessing pipeline** to normalize and enhance image features  

<img src="https://github.com/user-attachments/assets/d51dfbfc-0e7c-468f-a478-4e02a3b3cd77" alt="Resizing" width="500">
<img src="https://github.com/user-attachments/assets/510ff2f7-04de-464c-9961-6f0da3f70e21" alt="PCA1" width="500">
<img src="https://github.com/user-attachments/assets/2d9f36fb-e5d5-42bc-9d92-a171169dd6c3" alt="PCA2" width="500">
<img src="https://github.com/user-attachments/assets/6c1c1847-580e-48e0-96d3-85d20d4ec649" alt="NMF" width="500">
<img src="https://github.com/user-attachments/assets/df35bf57-abd2-4a1c-83bb-13b149ce2496" alt="image" width="500">

### **2️⃣ Outlier Detection 🧐**  
✔ Implemented an **Autoencoder** to detect anomalies in MRI scans  
✔ Used **Mean Squared Error (MSE) thresholding** to flag outliers  
✔ Applied **Isolation Forest** and **NMF-based anomaly detection**  

<img src="https://github.com/user-attachments/assets/5142d57c-dcaa-4f7f-859b-fb76c82d9863" alt="Autoencoder2" width="500">
<img src="https://github.com/user-attachments/assets/66e95a3a-8245-4bdf-90ff-40537112e6cf" alt="notumor detection" width="500">

### **3️⃣ Feature Selection & Model Training 🔍**  
✔ Performed **Recursive Feature Elimination with Cross-Validation (RFECV)** to select optimal features  
✔ Used **LASSO Logistic Regression** for feature importance ranking  

### **4️⃣ Deep Learning Classification 🤖**  
✔ Implemented **CNN models (ResNet50, VGG16)** for automatic tumor classification  
✔ Experimented with **Vision Transformer (ViT)** for improved accuracy  
✔ Trained models using **Categorical Cross-Entropy Loss & Adam Optimizer**  

<img src="https://github.com/user-attachments/assets/cafacbea-54fa-47be-9a84-627d21a87632" alt="image" width="500">
<img src="https://github.com/user-attachments/assets/3ee5f42a-e750-445c-bae3-16571cd3e03b" alt="Decision Tree Classifier" width="500">
<img src="https://github.com/user-attachments/assets/50b8543c-b32f-463e-a1e4-7482e9f120aa" alt="pairplot" width="500">
<img src="https://github.com/user-attachments/assets/10b7b416-e902-4354-9331-eb636ab4b561" alt="pairplot2" width="500">
<img src="https://github.com/user-attachments/assets/4f16a62e-979a-4197-82b4-bd5ea9f24f29" alt="pairplot3" width="500">
<img src="https://github.com/user-attachments/assets/c7133fdc-0c3b-4e30-a238-f33fb7d27df4" alt="VGG16" width="500">
<img src="https://github.com/user-attachments/assets/8c045040-cfab-417a-9ecb-ebba9653ff4e" alt="VGG16" width="500">

### **5️⃣ Performance Evaluation 📊**  
✔ Measured model performance using **Precision, Recall, and F1-score**  
✔ Achieved **high accuracy in tumor classification**, with CNN models outperforming traditional classifiers  
✔ Evaluated confusion matrix and ROC curves to analyze classification performance  

---  

## **📈 Results & Findings**  
- **Autoencoder & Isolation Forest** successfully identified outliers based on MSE analysis  
- **Feature Selection (RFECV, LASSO) reduced dimensionality** while preserving classification accuracy  
- **ResNet50 & VGG16 achieved the best performance**, surpassing machine learning classifiers  
- **Vision Transformer (ViT) showed potential for further improvement in MRI-based tumor detection**  

---  
