# BioVision-Non-Intrusive-Hemoglobin-Detection-System
BioVision is an AI-powered system for classifying biomedical images and predicting hemoglobin-related conditions. 
It integrates deep learning (TensorFlow/Keras) with a MySQL database pipeline to automatically process images, generate predictions, 
and update results in real time.
Features
- Automated prediction loop that continuously fetches new image records from a MySQL database, classifies them, and updates results.
- Deep learning model implemented with TensorFlow/Keras for hemoglobin classification.
- Image preprocessing pipeline for resizing, normalization, and batch formatting.
- Database integration for storing and updating predictions seamlessly.
Tech Stack
- Languages: Python
- Libraries: TensorFlow/Keras, NumPy, OpenCV
- Database: MySQL (via PyMySQL)
- Environment: Linux/Windows compatible
Workflow
1. Fetch unclassified image records from the MySQL database.
2. Preprocess images (resize 150x150, normalize).
3. Run deep learning model predictions using TensorFlow/Keras.
4. Update the disease table in MySQL with predicted results.
5. Repeat in a continuous loop for real-time classification.
Installation & Setup
1. Clone the Repository:
   git clone https://github.com/your-username/BioVision.git
   cd BioVision

2. Create a Virtual Environment (optional):
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows

3. Install Dependencies:
   pip install -r requirements.txt

4. Setup MySQL Database:
   - Install MySQL and create a database named `tmss`.
   - Create a table `disease` with the following schema:
     CREATE TABLE disease (
         ids INT PRIMARY KEY AUTO_INCREMENT,
         pic VARCHAR(255),
         disease VARCHAR(255)
     );
   - Insert sample rows with image names for testing.

5. Run the Prediction Loop:
   python hemoglobin_predictor.py
Requirements File Example
tensorflow
keras
numpy
opencv-python
pymysql
Sample Code Snippet
import numpy as np
import pymysql as mdb
from keras.preprocessing import image
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("hemoglobin_model.h5")

# Database update function
def upd(disease, ids):
    mydb = mdb.connect(host="127.0.0.1", user="root", passwd="", database="tmss")
    cursor = mydb.cursor()
    sql = "UPDATE disease SET disease = %s WHERE ids = %s"
    cursor.execute(sql, (disease, ids))
    mydb.commit()
    mydb.close()

# Prediction example
img = image.load_img("sample.jpg", target_size=(150,150))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
predicted_class = np.argmax(prediction)
upd(str(predicted_class), 1)  # Example update
Future Enhancements
- Add a Streamlit/Flask dashboard for real-time monitoring of predictions.
- Extend support to multiple biomedical datasets.
- Incorporate explainable AI (Grad-CAM) for interpretability of results.

