import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Generative AI based Data Analyst! 👋")
st.sidebar.success("Select either data cleaning (CSV Agent) or Data Analysis above.")
st.divider()

st.markdown(
    """
    Hi, my name is Tarun Arora, MPCS student at the University of Chicago. This is my final project for Generative AI.
    
    For the purpose of this project, I tried to build two things:
    
    A. Langchain based agent with memory and tools that can help users to clean their CSV files. 
    
    B. OpenAI model that suggests you different plots for data visualization, asks you for two columns you want to create a plot for, writes the code for it as well as it displays the graph for you! 

    Ideally, I have designed this multi-page streamlit application to allow users to first clean their CSV files by interacting with the langchain agent and tools using a chatbot.
    Followed by data analysis on the downloaded cleaned data. 
"""
)