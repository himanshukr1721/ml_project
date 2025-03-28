# Career Path Prediction App 🎓💼

A **Streamlit-based web application** that predicts the most suitable career path for users based on their skills, interests, and preferences. This app uses a **machine learning model** trained on a dataset of professionals to provide personalized career recommendations.

---

## 🚀 Features

- **User-Friendly Interface**: Simple and intuitive UI built with Streamlit.
- **Dynamic Inputs**: Collects user information such as skills, interests, and preferences through sliders, dropdowns, and text inputs.
- **Career Prediction**: Uses a pre-trained machine learning model to predict the best career path.
- **Interactive Elements**:
  - Progress bar for prediction process.
  - Balloons for a fun user experience.
  - Expandable sections for detailed explanations.

---

## 🛠️ Technologies Used

- **Python**: Core programming language.
- **Streamlit**: For building the web application.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical computations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For training the machine learning model.
- **Pickle**: For saving and loading the trained model.

---

---

## 📋 How It Works

1. **Input Collection**:
   - Users provide their details such as logical quotient rating, coding skills, public speaking points, and preferences for certifications, workshops, and career areas.

2. **Feature Encoding**:
   - Categorical inputs are encoded into numerical values using mappings.

3. **Prediction**:
   - The encoded inputs are passed to a pre-trained **Decision Tree Classifier** to predict the most suitable career path.

4. **Visualization**:
   - A correlation heatmap is displayed to show relationships between numerical features.

---

## 🖥️ How to Run the Project

### Prerequisites
- Python 3.8 or higher
- Required Python libraries:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/career-path-prediction.git
   cd career-path-prediction

2. Install the required libraries:
    ```pip install -r requirements.txt
3. Run the Streamlit app:
    ```streamlit run app.py

4. Open the app in your browser at http://localhost:8501.

📊 Dataset
The dataset (mldata.csv) contains information about professionals, including:

Skills (e.g., coding, public speaking)
Preferences (e.g., certifications, workshops)
Career paths (e.g., developer, cloud computing)
The dataset is preprocessed with:

Binary encoding for "Yes/No" features.
Numerical encoding for categorical features like "poor", "medium", "excellent".
Dummy variable encoding for multi-class features.
🧠 Machine Learning Model
Model: Decision Tree Classifier
Training: The model was trained on the mldata.csv dataset.
Input Features: 21 features including skills, preferences, and interests.
Output: Predicted career path.


🤝 Contributing
Contributions are welcome! If you'd like to improve this project:

Fork the repository.
Create a new branch (feature/your-feature).
Commit your changes.
Push to the branch.
Open a pull request.
📜 License
This project is licensed under the MIT License. See the LICENSE file for details.

❤️ Acknowledgments
Thanks to the creators of Streamlit for making web app development easy.
Special thanks to the contributors of Scikit-learn, Pandas, and Seaborn.

Made with ❤️ by Himanshu and Mrigyanshi