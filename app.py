import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from src.model import CareerPredictor

# Load the model
@st.cache_resource
def load_model(model_path="models/career_predictor_rf.joblib"):
    try:
        return CareerPredictor.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Main app
def main():
    st.title("Career Path Predictor")
    st.write("AI-based skill assessment and career guidance")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Choose a page", ["Home", "Predict", "Model Info"])
    
    if page == "Home":
        show_home_page()
    elif page == "Predict":
        show_prediction_page()
    elif page == "Model Info":
        show_model_info_page()

def show_home_page():
    st.header("Welcome to Career Path Predictor")
    st.write("""
    This application helps you find potential career paths based on your skills, 
    education, interests, and experience.
    
    ### How it works
    1. Navigate to the 'Predict' page from the sidebar
    2. Enter your information
    3. Get personalized career recommendations
    
    ### About
    This tool uses machine learning to analyze patterns in career paths and make predictions 
    based on your unique profile. The recommendations are based on patterns identified from 
    historical data of successful professionals in various fields.
    """)
    
    st.image("https://www.ies.ncsu.edu/wp-content/uploads/sites/26/2022/10/CareerPathways1200x628.jpg", 
             caption="Career Pathways", use_column_width=True)

def show_prediction_page():
    st.header("Career Prediction")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Model could not be loaded. Please check if the model file exists.")
        return
    
    # Create input form
    with st.form("prediction_form"):
        # These fields should match your actual model features
        # Update these based on your dataset's features
        
        col1, col2 = st.columns(2)
        
        with col1:
            education_level = st.selectbox(
                "Education Level",
                ["High School", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "PhD"]
            )
            
            experience_years = st.slider(
                "Years of Experience",
                min_value=0.0,
                max_value=20.0,
                value=2.0,
                step=0.5
            )
        
        with col2:
            # Multi-select fields for skills and interests
            technical_skills = st.multiselect(
                "Technical Skills",
                ["Programming", "Data Analysis", "Machine Learning", "Web Development", 
                 "Database Management", "Project Management", "Design", "Marketing",
                 "Communication", "Problem Solving", "Leadership"]
            )
            
            soft_skills = st.multiselect(
                "Soft Skills",
                ["Communication", "Teamwork", "Problem Solving", "Time Management",
                 "Leadership", "Adaptability", "Creativity", "Critical Thinking"]
            )
        
        interests = st.multiselect(
            "Career Interests",
            ["Technology", "Healthcare", "Education", "Finance", "Arts", "Science",
             "Engineering", "Business", "Social Work", "Entertainment"]
        )
        
        # Submit button
        submitted = st.form_submit_button("Predict Career Path")
    
    if submitted:
        # Process the input data
        # Convert multi-select fields to a format suitable for the model
        tech_skills_str = ", ".join(technical_skills)
        soft_skills_str = ", ".join(soft_skills)
        interests_str = ", ".join(interests)
        
        # Create input dataframe
        input_data = pd.DataFrame({
            "education_level": [education_level],
            "technical_skills": [tech_skills_str],
            "soft_skills": [soft_skills_str],
            "interests": [interests_str],
            "experience_years": [experience_years]
            # Add more fields as required by your model
        })
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            
            # Get top 3 predictions with probabilities
            class_probs = [(model.target_classes[i], float(prob)) for i, prob in enumerate(probabilities)]
            class_probs.sort(key=lambda x: x[1], reverse=True)
            top_3 = class_probs[:3]
            
            # Display results
            st.success(f"**Recommended Career Path: {top_3[0][0]}**")
            st.write(f"Confidence: {top_3[0][1]*100:.1f}%")
            
            st.subheader("Alternative Career Paths")
            
            # Display alternatives
            for career, prob in top_3[1:]:
                st.write(f"- {career} ({prob*100:.1f}%)")
            
            # Plot probabilities
            fig, ax = plt.subplots()
            careers = [career for career, _ in top_3]
            probs = [prob*100 for _, prob in top_3]
            
            sns.barplot(x=probs, y=careers, palette="viridis", ax=ax)
            ax.set_xlabel("Confidence (%)")
            ax.set_ylabel("Career Path")
            ax.set_title("Top Career Recommendations")
            
            st.pyplot(fig)
            
            # Additional guidance
            st.subheader("Next Steps")
            st.write(f"""
            Based on your profile, here are some next steps to pursue a career in **{top_3[0][0]}**:
            
            1. Research specific job requirements for this field
            2. Consider enhancing your skills in relevant areas
            3. Look for entry-level positions or internships
            4. Connect with professionals in this field for mentorship
            
            Remember, this is a recommendation based on your current profile. Your career path 
            may evolve as you gain more skills and experience.
            """)
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please make sure all required fields are filled correctly.")

def show_model_info_page():
    st.header("Model Information")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Model could not be loaded.")
        return
    
    # Display model information
    st.subheader("Model Type")
    st.write(f"This career prediction model uses a **{model.model_type.upper()}** algorithm.")
    
    st.subheader("Features Used")
    st.write("The model uses the following features to make predictions:")
    
    for i, feature in enumerate(model.feature_names):
        st.write(f"- {feature}")
    
    st.subheader("Possible Career Paths")
    st.write("The model can predict the following career paths:")
    
    # Display in a grid layout
    cols = st.columns(3)
    for i, career in enumerate(model.target_classes):
        cols[i % 3].write(f"- {career}")
    
    # Try to load and display feature importances
    try:
        import os
        importance_path = "models/feature_importances.csv"
        
        if os.path.exists(importance_path):
            importances = pd.read_csv(importance_path)
            
            st.subheader("Feature Importance")
            st.write("These are the most important factors in determining career paths:")
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importances.head(10), ax=ax)
            ax.set_title('Top 10 Feature Importances')
            st.pyplot(fig)
        else:
            st.info("Feature importance data is not available.")
    
    except Exception as e:
        st.warning(f"Could not load feature importance information: {e}")

if __name__ == "__main__":
    main()