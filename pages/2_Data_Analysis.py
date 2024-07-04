from dotenv import load_dotenv
import pandas as pd
import numpy as np
import streamlit as st
import os
from openai import OpenAI

load_dotenv()

def get_code(prompt_template):
    client = OpenAI()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant who knows and understands data analytics very well."},
        {"role": "user", "content": prompt_template}
    ]
    )
    code = completion.choices[0].message.content
    start_index = code.find("python") + len("python")
    end_index = code.find("```", start_index)
    refined_code = code[start_index:end_index]

    python_file = open('analytics.py', 'w')
    python_file.write(refined_code)
    python_file.close()

    st.code(refined_code, language='python')

    file_path = 'plot.png'
    if os.path.exists(file_path):
        os.remove(file_path)

    import analytics
    analytics.plot()
    st.image("plot.png")

def get_columns():

    file_path = 'plot.png'

    if os.path.exists(file_path):
        os.remove(file_path)

    df = st.session_state.analysis_df
    list_of_available_columns = list(df.columns)

    if 'column1' not in st.session_state:
        st.session_state.column1 = list_of_available_columns[0]

    if 'column2' not in st.session_state:
        st.session_state.column2 = list_of_available_columns[0]

    st.session_state.column1 = st.selectbox("Select column 1", list_of_available_columns, index=list_of_available_columns.index(st.session_state.column1))
    st.session_state.column2 = st.selectbox("Select column 2", list_of_available_columns, index=list_of_available_columns.index(st.session_state.column2))
    st.session_state.type_graph = st.text_input("what type of graph do you want to plot?")

    column1=st.session_state.column1
    column2=st.session_state.column2
    type_graph = st.session_state.type_graph

    type1 = type(df[column1])
    type2 = type(df[column2])
    columns = np.array(df.columns)
    number_of_rows = len(df)

    context = f""

    question = f"""I have a dataset with following columns: {columns}

    I want to draw a graph to analyze the data between columns {column1} and {column2}. I want to have {column1} on x-axis and {column2} on y-axis. The type of data in {column1} is {type1} and the type of data in {column2} is {type2}. There are a total number of {number_of_rows} in the dataset. I want the graph to be {type_graph}.
    The file location is in the same directory and name is analysis.csv. The plot should be saved as plot.png

    Do not rush into writing the code, understand the above mentioned instructions and the information provided to decide how to perform data analysis and then proceed to write python code for the same."""

    prompt_template = f"""
    # CONTEXT #
    You will be helping me with python code to perform data visualization based using matplotlib graphs on the dataset and columns I define.

    # OBJECTIVE #
    Your objective throughout will be to provide me with high quality python code that can produce detailed graphs. Detailed graphs here refer to graphs which have labels on both axis, legends and title. Based on the description of the dataset and the columns, you should decide to code the graph that will be the most useful in analyzing the data. 

    # STYLE #
    You should provide the code for the data visualization and graphs with detailed comments explaining every step. The code should not have any errors and should high quality graphs. If there are multiple graphs you should consider using subplots making sure you generate one image including all the graphs. Also, at the end of the code add a line to save the plot generated as an image.

    # TONE #
    You should maintain an analytical tone and produce code that helps user get insights from the data.

    # AUDIENCE #
    The target audience are people who are currently working on data analytics projects and need help in producing high quality graphs to get data insights.

    # RESPONSE FORMAT #
    The whole code should be in a function called plot. Your code should not have plt.show() line in the code.
    You should make sure you should just provide the detailed code as described above.
    You should also analyze when does it make sense to actually take average when rows can have repeated column values (like county, bedrooms etc.)

    #############################################################################################################################

    # START DATA ANALYSIS #
    {context}
    {question}

    """
    button1 = st.button("Get me the code for analysis!", type="primary")
    if button1:
        get_code(prompt_template)
    else:
        st.warning("Please select two columns to perform analysis on.")

@st.cache_data
def get_plot_suggestions(user_input, columns):
    
    df = st.session_state.analysis_df
    if not df.empty:
        # columns = np.array(df.columns)
        types_columns = df.dtypes.apply(lambda x: x.name).to_dict()


        if (user_input):
            suggestion_prompt = f"""I have a dataset with following columns: {columns}. 
            The following are the respective columns and its types: {types_columns}. 
            Specifically, in the area of {user_input}.
            Based on this information, can you suggest me graphs with which column on x-axis and which column on y-axis that should be created to analyze the dataset.    
            You should present the output in bullet points format and analyze the different columns and their column types before rushing to suggestions.
            """
            client = OpenAI()
            completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who knows and understands data analytics very well."},
                {"role": "user", "content": suggestion_prompt}
            ]
            )
            st.write(completion.choices[0].message.content)
            # return True
        else:
            st.warning("Enter a suggestion or just enter None to get data analysis suggestions.")
    

def main():
    file_path = 'plot.png'

    if os.path.exists(file_path):
        os.remove(file_path)

    st.title('One Step Data Analysis!')
    st.subheader('Upload the csv document', divider='rainbow')

    uploaded_file = st.file_uploader("Upload the document", type='csv')

    if uploaded_file:

        if "analysis_df" not in st.session_state:
            with open("analysis.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.analysis_df = pd.read_csv(uploaded_file)

        suggestion_input = st.text_input("Any specific data analysis you want:")
        columns = np.array(st.session_state.analysis_df.columns)
        get_plot_suggestions(suggestion_input, columns)
        get_columns()

    else:
        st.warning("Please upload a file first.")

if __name__ == "__main__":
    done = st.button("Done with this analysis!")
    if done:
        file_path = 'plot.png'
        if os.path.exists(file_path):
            os.remove(file_path)
    main()