import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pandas as pd

# Load model and vectorizer
model = joblib.load("category_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# --- Page Config ---
st.set_page_config(page_title="News Headline Classifier", layout="centered")

# --- Custom Background Gradient ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #d3cce3, #e9e4f0);
    }
    .main {
        background-color: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin-top: 2rem;
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Navigation ---
with st.sidebar:
    selected = option_menu(
        "Navigation", ["🏠 Home", "📰 Classifier", "ℹ️ About"],
        icons=["house", "search", "info-circle"],
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "5px"},
            "icon": {"color": "black", "font-size": "16px"},
            "nav-link": {"font-size": "16px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#6c63ff", "color": "white"}
        }
    )

# --- Pages ---
if selected == "🏠 Home":
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("🧠 News Headline Category Classifier")
    st.markdown("""
    Welcome to the smart headline classifier app!  
    This tool uses Natural Language Processing and Machine Learning to detect the **topic** of any news headline.  

    Categories include:  
    - 🏛️ Politics  
    - 💼 Business  
    - 🏈 Sports  
    - 💻 Technology  
    - 🎬 Entertainment
    """)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "📰 Classifier":
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.header("🔎 Try Classifying a Headline")

    headline = st.text_input("Enter headline here", placeholder="e.g. Government passes new technology bill")
    if st.button("Classify"):
        if headline.strip():
            headline_vec = vectorizer.transform([headline.lower()])
            prediction = model.predict(headline_vec)[0]
            proba = model.predict_proba(headline_vec)[0]

            st.success(f"🎯 Predicted Category: **{prediction}**")

            # Show confidence scores
            proba_df = pd.DataFrame({
                "Category": model.classes_,
                "Confidence": proba
            }).sort_values("Confidence", ascending=False)
            st.bar_chart(proba_df.set_index("Category"))

        else:
            st.warning("⚠️ Please enter a headline to classify.")
    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "ℹ️ About":
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.header("ℹ️ About This Project")
    st.markdown("""
    This project is an NLP-powered web app that:
    - Uses **TF-IDF** vectorization
    - Applies a **Multinomial Naive Bayes** classifier
    - Classifies into 5 categories
    - Built using **Streamlit**

    Developed by: `Sourav Surendren`  
    GitHub Repo: [🔗 Click here](https://github.com/Souravsurendren/News_Headlines_Classifier.git)
    """)
    st.markdown("</div>", unsafe_allow_html=True)
