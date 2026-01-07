import streamlit as st
import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-2.5-flash")

if st.button("Test Gemini"):
    response = model.generate_content(
        "Write a short SEO description for a holiday rental"
    )
    st.write(response.text)