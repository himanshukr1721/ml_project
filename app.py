import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import time
import streamlit as st
from db import *

# Load the ML model
pickleFile = open("weights.pkl", "rb")
regressor = pickle.load(pickleFile)

# Load dataset
df = pd.read_csv('./data/mldata.csv')
df['workshops'] = df['workshops'].replace(['testing'], 'Testing')

# Preprocessing - recreating the encoding steps that are in the original code
# Binary encoding
cols = df[["self-learning capability?", "Extra-courses did", 
           "Taken inputs from seniors or elders", "worked in teams ever?", "Introvert"]]
for i in cols:
    cleanup_nums = {i: {"yes": 1, "no": 0}}
    df = df.replace(cleanup_nums)

# Number encoding
mycol = df[["reading and writing skills", "memory capability score"]]
for i in mycol:
    cleanup_nums = {i: {"poor": 0, "medium": 1, "excellent": 2}}
    df = df.replace(cleanup_nums)

# Category encoding - THIS WAS MISSING
category_cols = df[['certifications', 'workshops', 'Interested subjects', 
                    'interested career area ', 'Type of company want to settle in?', 
                    'Interested Type of Books']]
for i in category_cols:
    df[i] = df[i].astype('category')
    df[i + "_code"] = df[i].cat.codes

# Dummy encoding
df = pd.get_dummies(df, columns=["Management or Technical", "hard/smart worker"], prefix=["A", "B"])

# Now create dictionaries for encoding
Certifi = list(df['certifications'].unique())
certi_code = list(df['certifications_code'].unique())
C = dict(zip(Certifi, certi_code))

Workshops = list(df['workshops'].unique())
Workshops_code = list(df['workshops_code'].unique())
W = dict(zip(Workshops, Workshops_code))

Interested_subjects = list(df['Interested subjects'].unique())
Interested_subjects_code = list(df['Interested subjects_code'].unique())
ISC = dict(zip(Interested_subjects, Interested_subjects_code))

interested_career_area = list(df['interested career area '].unique())
interested_career_area_code = list(df['interested career area _code'].unique())
ICA = dict(zip(interested_career_area, interested_career_area_code))

Typeofcompany = list(df['Type of company want to settle in?'].unique())
Typeofcompany_code = list(df['Type of company want to settle in?_code'].unique())
TOCO = dict(zip(Typeofcompany, Typeofcompany_code))

Interested_Books = list(df['Interested Type of Books'].unique())
Interested_Books_code = list(df['Interested Type of Books_code'].unique())
IB = dict(zip(Interested_Books, Interested_Books_code))

Range_dict = {"poor": 0, "medium": 1, "excellent": 2}

# Prediction function - same as original
def inputlist(Name, Contact_Number, Email_address,
      Logical_quotient_rating, coding_skills_rating, hackathons, 
      public_speaking_points, self_learning_capability, 
      Extra_courses_did, Taken_inputs_from_seniors_or_elders,
      worked_in_teams_ever, Introvert, reading_and_writing_skills,
      memory_capability_score, smart_or_hard_work, Management_or_Techinical,
      Interested_subjects, Interested_Type_of_Books, certifications, workshops, 
      Type_of_company_want_to_settle_in, interested_career_area):
    
    Afeed = [Logical_quotient_rating, coding_skills_rating, hackathons, public_speaking_points]

    input_list_col = [self_learning_capability, Extra_courses_did, Taken_inputs_from_seniors_or_elders,
                    worked_in_teams_ever, Introvert, reading_and_writing_skills, memory_capability_score,
                    smart_or_hard_work, Management_or_Techinical, Interested_subjects, Interested_Type_of_Books,
                    certifications, workshops, Type_of_company_want_to_settle_in, interested_career_area]
    feed = []
    K = 0
    j = 0
    
    for i in input_list_col:
        if(i == 'Yes'):
            j = 2
            feed.append(j)
        
        elif(i == "No"):
            j = 3
            feed.append(j)
        
        elif(i == 'Management'):
            j = 1
            k = 0
            feed.append(j)
            feed.append(K)
        
        elif(i == 'Technical'):
            j = 0
            k = 1
            feed.append(j)
            feed.append(K)
        
        elif(i == 'Smart worker'):
            j = 1
            k = 0
            feed.append(j)
            feed.append(K)
        
        elif(i == 'Hard Worker'):
            j = 0
            k = 1
            feed.append(j)
            feed.append(K)
        
        else:
            for key in Range_dict:
                if(i == key):
                    j = Range_dict[key]
                    feed.append(j)
            
            for key in C:
                if(i == key):
                    j = C[key]
                    feed.append(j)
            
            for key in W:
                if(i == key):
                    j = W[key]
                    feed.append(j)
            
            for key in ISC:
                if(i == key):
                    j = ISC[key]
                    feed.append(j)
            
            for key in ICA:
                if(i == key):
                    j = ICA[key]
                    feed.append(j)
            
            for key in TOCO:
                if(i == key):
                    j = TOCO[key]
                    feed.append(j)
            
            for key in IB:
                if(i == key):
                    j = IB[key]
                    feed.append(j)
    
    t = Afeed + feed    
    output = regressor.predict([t])
    
    return(output)

# Set page configuration
st.set_page_config(
    page_title="Career Path Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 5px;
        padding: 10px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e5de4 !important;
        color: white !important;
    }
    div.block-container {
        padding-top: 2rem;
    }
    div.stButton > button {
        background-color: #4e5de4;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #3a49d3;
        border-color: #3a49d3;
    }
    .header-container {
        display: flex;
        align-items: center;
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header-text {
        flex: 2;
    }
    .header-image {
        flex: 1;
        text-align: center;
    }
    .stSlider > div > div > div {
        background-color: #4e5de4;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .skills-card {
        background-color: #f8f9fa;
        border-left: 5px solid #4e5de4;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0 5px 5px 0;
    }
    .form-header {
        color: #4e5de4;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .result-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #6c757d;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style="padding: 20px 0;">
            <h1 style="color: #4e5de4; font-size: 2.5rem;">Career Path Predictor</h1>
            <p style="font-size: 1.2rem; color: #555;">Discover your ideal career path based on your skills, preferences, and personality.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("./assets/Career _Isometric.png", width=250)

    # Create tabs for better organization
    tab1, tab2 = st.tabs(["💼 Career Assessment", "📊 Results & Analytics"])
    
    # Store form inputs in session state to persist between tabs
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
    
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = ""

    # Tab 1: Career Assessment Form
    with tab1:
        st.markdown("<h3 class='form-header'>Personal Information</h3>", unsafe_allow_html=True)
        
        # Personal Info in columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            Name = st.text_input("Full Name")
            Email_address = st.text_input("Email Address")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            Contact_Number = st.text_input("Contact Number")
            st.markdown("<p style='color:#6c757d;font-size:0.9rem;'>Your information is secure and will only be used for this assessment.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Skills Assessment
        st.markdown("<h3 class='form-header'>Skills & Abilities</h3>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            Logical_quotient_rating = st.slider('Logical Quotient Skills', 0, 10, 5)
            coding_skills_rating = st.slider('Coding Skills', 0, 10, 5)
        
        with col2:
            hackathons = st.slider('Hackathons Participated', 0, 10, 1)
            public_speaking_points = st.slider('Public Speaking Skills', 0, 10, 5)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Personal Traits in columns
        st.markdown("<h3 class='form-header'>Personal Traits & Experiences</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<div class='skills-card'>", unsafe_allow_html=True)
            self_learning_capability = st.selectbox('Self-Learning Capability', ('Yes', 'No'))
            Extra_courses_did = st.selectbox('Completed Extra Courses', ('Yes', 'No'))
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='skills-card'>", unsafe_allow_html=True)
            Taken_inputs_from_seniors_or_elders = st.selectbox('Seek Advice from Seniors', ('Yes', 'No'))
            worked_in_teams_ever = st.selectbox('Team Collaboration Experience', ('Yes', 'No'))
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<div class='skills-card'>", unsafe_allow_html=True)
            Introvert = st.selectbox('Are You an Introvert?', ('Yes', 'No'))
            reading_and_writing_skills = st.selectbox('Reading & Writing Skills', ('poor', 'medium', 'excellent'))
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Work Style & Preferences
        st.markdown("<h3 class='form-header'>Work Style & Preferences</h3>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            memory_capability_score = st.selectbox('Memory Capability', ('poor', 'medium', 'excellent'))
            smart_or_hard_work = st.selectbox('Work Style', ('Smart worker', 'Hard Worker'))
            Management_or_Techinical = st.selectbox('Preferred Role Type', ('Management', 'Technical'))
        
        with col2:
            Interested_subjects = st.selectbox('Interested Subjects', 
                ('programming', 'Management', 'data engineering', 'networks', 
                'Software Engineering', 'cloud computing', 'parallel computing', 
                'IOT', 'Computer Architecture', 'hacking'))
            
            Type_of_company_want_to_settle_in = st.selectbox('Preferred Company Type', 
                ('BPA', 'Cloud Services', 'product development', 
                'Testing and Maintainance Services', 'SAaS services', 'Web Services', 
                'Finance', 'Sales and Marketing', 'Product based', 'Service Based'))
            
            interested_career_area = st.selectbox('Interested Career Area', 
                ('testing', 'system developer', 'Business process analyst', 
                'security', 'developer', 'cloud computing'))
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Additional Information
        st.markdown("<h3 class='form-header'>Additional Information</h3>", unsafe_allow_html=True)
        
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            certifications = st.selectbox('Certifications Completed', 
                ('information security', 'shell programming', 'r programming', 
                'distro making', 'machine learning', 'full stack', 
                'hadoop', 'app development', 'python'))
        
        with col2:
            workshops = st.selectbox('Workshops Attended', 
                ('Testing', 'database security', 'game development', 
                'data science', 'system designing', 'hacking', 
                'cloud computing', 'web technologies'))
        
        Interested_Type_of_Books = st.selectbox('Preferred Book Categories', 
            ('Series', 'Autobiographies', 'Travel', 'Guide', 'Health', 
            'Journals', 'Anthology', 'Dictionaries', 'Prayer books', 
            'Art', 'Encyclopedias', 'Religion-Spirituality', 
            'Action and Adventure', 'Comics', 'Horror', 'Satire', 
            'Self help', 'History', 'Cookbooks', 'Math', 'Biographies', 
            'Drama', 'Diaries', 'Science fiction', 'Poetry', 'Romance', 
            'Science', 'Trilogy', 'Fantasy', 'Childrens', 'Mystery'))
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Submit Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submit_button = st.button("Predict My Career Path", use_container_width=True)
        
        if submit_button:
            if not Name or not Email_address:
                st.error("Please provide your name and email address to continue.")
            else:
                with st.spinner("Analyzing your profile..."):
                    # Progress bar with a more professional look
                    progress_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.02)  # Faster animation
                        progress_bar.progress(percent_complete + 1)
                    
                    # Make prediction
                    result = inputlist(Name, Contact_Number, Email_address, 
                                     Logical_quotient_rating, coding_skills_rating, hackathons, 
                                     public_speaking_points, self_learning_capability, Extra_courses_did, 
                                     Taken_inputs_from_seniors_or_elders, worked_in_teams_ever, Introvert,
                                     reading_and_writing_skills, memory_capability_score, smart_or_hard_work, 
                                     Management_or_Techinical, Interested_subjects, Interested_Type_of_Books,
                                     certifications, workshops, Type_of_company_want_to_settle_in, interested_career_area)
                    
                    # Store result in session state
                    st.session_state.prediction_result = result
                    st.session_state.form_submitted = True
                    
                    # Database operations
                    create_table()
                    add_data(Name, Contact_Number, Email_address, Logical_quotient_rating, coding_skills_rating, hackathons, 
                           public_speaking_points, self_learning_capability, Extra_courses_did, 
                           Taken_inputs_from_seniors_or_elders, worked_in_teams_ever, Introvert,
                           reading_and_writing_skills, memory_capability_score, smart_or_hard_work, 
                           Management_or_Techinical, Interested_subjects, Interested_Type_of_Books,
                           certifications, workshops, Type_of_company_want_to_settle_in, interested_career_area)
                    
                    # Trigger balloons effect
                    st.balloons()

    # Tab 2: Results & Analytics
    with tab2:
        if st.session_state.form_submitted:
            st.markdown("""
            <div class="result-container">
                <img src="./assets/career.png" width="120">
                <h2 style="color: #4e5de4; margin-top: 1rem;">Your Ideal Career Path</h2>
                <h1 style="font-size: 2.2rem; font-weight: bold; color: #333; margin: 1rem 0;">
                    {0}
                </h1>
                <p style="color: #666; font-size: 1.1rem;">
                    Based on your skills, interests, and preferences, we've identified this as your ideal career path.
                </p>
            </div>
            """.format(st.session_state.prediction_result[0]), unsafe_allow_html=True)
            
            # Analytics Section
            st.markdown("<h3 class='form-header' style='margin-top: 2rem;'>Skills Analytics</h3>", unsafe_allow_html=True)
            
            # Feature correlation plot
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            corr = df[['Logical quotient rating', 'hackathons', 
           'coding skills rating', 'public speaking points']].corr()
            f,axes = plt.subplots(1,1,figsize = (10,10))
            sns.heatmap(corr,square=True,annot = True,linewidth = .4,center = 2,ax = axes)
            st.subheader("Here are some nerdy analytics 😁")
            st.text("Correlation Between Numerical Features")
            st.pyplot(f)

            
            with st.expander("What does this correlation mean?"):
                st.write("""
                This heatmap shows the relationships between different skills in our dataset:
                
                - **Positive correlation** (closer to 1): When one skill increases, the other tends to increase as well.
                - **Negative correlation** (closer to -1): When one skill increases, the other tends to decrease.
                - **No correlation** (closer to 0): The skills don't have a consistent relationship.
                
                Understanding these relationships can help you focus on complementary skills for your career development.
                """)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Career recommendations
            st.markdown("<h3 class='form-header'>Next Steps</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("""
                <h4 style="color: #4e5de4;">Recommended Skills to Develop</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="padding: 8px 0; border-bottom: 1px solid #eee;">✅ Advanced problem-solving techniques</li>
                    <li style="padding: 8px 0; border-bottom: 1px solid #eee;">✅ Communication and presentation skills</li>
                    <li style="padding: 8px 0; border-bottom: 1px solid #eee;">✅ Team collaboration and project management</li>
                    <li style="padding: 8px 0;">✅ Industry-specific technical knowledge</li>
                </ul>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("""
                <h4 style="color: #4e5de4;">Recommended Resources</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="padding: 8px 0; border-bottom: 1px solid #eee;">📚 Online courses and certifications</li>
                    <li style="padding: 8px 0; border-bottom: 1px solid #eee;">📚 Industry conferences and networking events</li>
                    <li style="padding: 8px 0; border-bottom: 1px solid #eee;">📚 Professional mentorship programs</li>
                    <li style="padding: 8px 0;">📚 Industry-specific projects and challenges</li>
                </ul>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
        else:
            st.info("Please complete the assessment form in the 'Career Assessment' tab to see your results.")
            if st.button("Go to Assessment Form"):
                pass  # The JavaScript redirect doesn't work in Streamlit, so we'll just rely on the tab UI
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Developed with ❤️ | © 2025 Career Path Predictor</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()