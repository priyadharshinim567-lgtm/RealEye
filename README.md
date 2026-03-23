👁️ RealEye – AI vs Real Image Detection
📌 Overview

RealEye is a deep learning-based web application that detects whether an uploaded image is real (authentic) or AI-generated.
It also provides visual explanation (Grad-CAM heatmap) to show which parts of the image influenced the prediction.

🚀 Features
🔍 Detects AI-generated vs real images
🌡️ Grad-CAM heatmap for explainability
📊 Confidence score for predictions
🖥️ Interactive web app using Streamlit
⚡ Fast and user-friendly interface

🧠 Technologies Used
Python
TensorFlow / Keras (CNN)
Streamlit
NumPy, PIL
Grad-CAM (Explainable AI)

📂 Project Structure
RealEye/
│
├── dataset/                # Training images (real & AI)
├── model/
│   ├── cnn_model.py        # CNN architecture
│   ├── realeye_model.keras # Trained model
│
├── preprocessing/
│   ├── preprocess.py       # Image preprocessing
│
├── utils/
│   ├── gradcam.py          # Heatmap generation
│
├── train_model.py          # Model training script
├── app.py                  # Streamlit web app
└── README.md

⚙️ Installation
pip install -r requirements.txt

🏋️‍♀️ Train the Model
python train_model.py

🌐 Run the Application
streamlit run app.py

📊 How It Works
User uploads an image
Image is preprocessed (resized, normalized)
CNN model predicts probability

Output:
Label (Real / AI Generated)
Confidence score
Grad-CAM generates heatmap for explanation

🎯 Use Cases
Fake image detection
Social media verification
Digital content authentication
Cybersecurity applications

📈 Future Enhancements
🎥 Video deepfake detection
🌐 Browser extension integration
☁️ Cloud deployment (real-time API)
📱 Mobile app version

📌 Conclusion

RealEye demonstrates how AI can be used responsibly to detect AI-generated content, improving trust and authenticity in digital media.

⭐ Acknowledgment

Datasets sourced from public repositories (Kaggle & open sources).
- ❌ Web interface (planned for later weeks)
