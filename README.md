## Data Cleaning and Analysis using Generative AI and Langchain

The streamlit application has 3 pages

Welcome Page giving brief introduction to the project.
MY CSV Agent which is a chatbot built to clean the data. Only tools get access to the contents of CSV file.
Data Analysis which suggests users what plots to create for data visualization, generates code and displays the plot as per user inputs.
Additionally, I created a custom GPT to perform similar operations which is discussed in the report: Project_Report_Generative_AI.pdf

To run the project:

Delete the existing plot.png file.
Clear the contents of analytics.py file.
run streamlit run Welcome.py in the terminal
Welcome page should appear on your browser.
Go to MY CSV Agent page and upload special_rules_rental_data_100_rows.csv and interact with chatbot to clean the file.
Download the clean CSV file and upload it in Data Analysis page and provide required inputs to get plot suggestions, code and plot.
Unclean data is stored in special_rules_rental_data_100_rows.csv Cleaned data after using My CSV Agent as depicted in video is stored in cleaned_data (1).csv

Code generated for creating plot is stored in: analytics.py

Plot generated from that code is stored in: plot.png
