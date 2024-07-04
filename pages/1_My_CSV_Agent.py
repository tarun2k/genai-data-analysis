from langchain_openai import ChatOpenAI, OpenAI
from dotenv import load_dotenv
from langchain.agents import tool
import pandas as pd
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
import numpy as np
from scipy import stats

load_dotenv()

st.title("My CSV Chatbot!")
st.write("Helps you clean your CSV files!")
st.subheader('Upload the csv document', divider='rainbow')

uploaded_file = st.file_uploader("Upload the document", type='csv')

if uploaded_file is not None:
    if 'df' not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file)

else:
    st.warning("Please upload a file first.")
    st.stop()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are very powerful assistant."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

@tool
def display_df_head():
    """Returns the top 5 rows in DataFrame for the user to view."""
    df = st.session_state.df
    return df.head()

@tool
def df_info():
    """Returns a dictionary consisting of comprehensive information about the pandas DataFrame."""
    df = st.session_state.df
    information = {}

    information["rows"] = df.shape[0]
    information["columns"] = df.shape[1]

    information["data_types"] = df.dtypes.to_dict()
    information["missing_values"] = df.isnull().sum().to_dict()
    information["unique_values"] = df.nunique().to_dict()
    information["descriptive_statistics"] = df.describe().to_dict()

    return information    

@tool
def check_null():
    """Returns a dictionary of null values and respective columns in the pandas DataFrame."""
    df = st.session_state.df
    # return df.isnull().sum().sum()
    null_count_dict = {}
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            null_count_dict[col] = null_count
    return null_count_dict

@tool
def fix_null_median(column):
    """Returns DataFrame after replacing NaN with median value of the column."""
    try:
        df = st.session_state.df
        df[column] = df[column].fillna(df[column].median())
        df.to_csv("uncleaned_data.csv", index=False)
        st.session_state.df = df
        return df 
    except:
        return("Please check whether the column is numerical or please mention the column whose value you want to be fixed with the median!")

@tool
def fix_null_with_median_two_columns(target_column, column1, column2):
    """
    Fixes null values in a specified column using the median of rows where two other columns have the same values.
    Returns DataFrame with imputed values.
    """
    try:
        df = st.session_state.df

        for index, row in df[df[target_column].isnull()].iterrows():
            similar_rows = df[(df[column1] == row[column1]) & (df[column2] == row[column2])]
            median_value = similar_rows[target_column].median()
            if not pd.isna(median_value):
                df.at[index, target_column] = median_value

        return df
    except:
        return("Please check whether the column is numerical or please mention the column whose value you want to be fixed with the median!")

@tool
def fix_null_drop():
    """Returns DataFrame after dropping rows containing null values""" 
    df = st.session_state.df
    new_df = df.dropna()
    new_df.to_csv("uncleaned_data.csv", index=False)
    st.session_state.df = new_df
    return new_df

@tool
def check_data_anomalies():
    """Returns a dictionary with key as column name and value as different data types present in column name, 
    only for columns with more than one data type."""
    df = st.session_state.df
    mismatched_columns = {}
    for column in df.columns:
        column_types = df[column].apply(type).unique()
        if len(column_types) > 1:
            mismatched_columns[column] = column_types
    return mismatched_columns

@tool
def check_repeated_column_values():
    """Returns a list of column names where all the values are same."""
    df = st.session_state.df
    lst_cols = []
    for column in df.columns:
        if(len(df[column].unique())==1):
            lst_cols.append(column)
    return lst_cols

@tool
def fix_repeated_column_values():
    """
    Drops columns from a DataFrame where all rows have the same value.
    Returns the updated DataFrame.
    """
    df = st.session_state.df
    for column in df.columns:
        if len(df[column].unique()) == 1:
            df.drop(column, axis=1, inplace=True)

    st.session_state.df = df
    df.to_csv("uncleaned_data.csv", index=False)
    return df

@tool
def detect_outliers(method='iqr'):
    """Detects outliers in all numerical columns using IQR or Z-score method. 
    Returns a dictionary with each outlier row and the respective outlier columns."""

    df = st.session_state.df
    outliers_dict = {}

    for column in df.select_dtypes(include=[np.number]).columns:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outlier_condition = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))
        elif method == 'zscore':
            z_scores = stats.zscore(df[column].dropna())
            outlier_condition = df[column].isin(df[column][np.abs(z_scores) > 3])
        
        outliers = df[outlier_condition]
        for _, row in outliers.iterrows():
            row_key = tuple(row) 
            if row_key in outliers_dict:
                outliers_dict[row_key].append(column)
            else:
                outliers_dict[row_key] = [column]

    return outliers_dict

@tool
def fix_outliers(method='iqr'):
    """Returns updated df after replacing the outlier values with the median values of that column."""
    df = st.session_state.df    
    for column in df.select_dtypes(include=[np.number]).columns:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outlier_condition = ((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))
        elif method == 'zscore':
            z_scores = stats.zscore(df[column].dropna())
            outlier_condition = df[column].isin(df[column][np.abs(z_scores) > 3])
        
        median_column = df[column].median()
        df.loc[outlier_condition, column] = median_column

    df.to_csv("uncleaned_data.csv", index=False)
    st.session_state.df = df
    return df

@tool
def check_duplicate_rows():
    """
    Checks for duplicate rows in the DataFrame. 
    Returns number of duplicate rows present.
    """
    df = st.session_state.df
    number_duplicates = df.duplicated(keep='first').sum()
    return number_duplicates

@tool
def remove_duplicate_rows():
    """Removes duplicate rows in the DataFrame."""
    df = st.session_state.df
    df = df.drop_duplicates()
    df.to_csv("uncleaned_data.csv", index=False)
    st.session_state.df = df
    return df

@tool
def check_and_drop_high_correlation(threshold=0.8):
    """
    Checks for high correlation between numerical columns and drops one of the columns in highly correlated pairs.
    Returns a tuple of the updated DataFrame and list of columns that are dropped.
    """
    df = st.session_state.df
    numerical_df = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_df.corr()
    columns_to_drop = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                colname = corr_matrix.columns[i]
                columns_to_drop.add(colname)

    df.drop(columns=columns_to_drop, axis=1, inplace=True)

    df.to_csv("uncleaned_data.csv", index=False)
    st.session_state.df = df

    return df, list(columns_to_drop)



tools = [display_df_head, check_null, check_data_anomalies, check_repeated_column_values, fix_null_drop, fix_null_median, df_info, check_repeated_column_values,
 fix_repeated_column_values, remove_duplicate_rows, check_and_drop_high_correlation, detect_outliers, fix_outliers, fix_null_with_median_two_columns, 
 check_duplicate_rows]

llm_with_tools = llm.bind_tools(tools)

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are very powerful assistant, but bad at analyzing CSV files. You should suggest next steps to the user based on tools you have."),
    MessagesPlaceholder(variable_name=MEMORY_KEY),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_executor_m = AgentExecutor(agent=agent, tools=tools, memory = memory, verbose=True)

chat_history=[]

if "messages" not in st.session_state:
    st.session_state.messages =[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

input_text = st.chat_input()

if input_text:
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.markdown(input_text)

    with st.chat_message("assistant"):
        result = agent_executor_m.invoke({"input": input_text, "chat_history": chat_history})
        st.markdown(result['output'])
    st.session_state.messages.append({"role": "assistant", "content": result['output']})

if uploaded_file:
    df = st.session_state.df
    csv_df = df.to_csv(index=False).encode('utf-8')
    completed_cleaning = st.button("Click to finish and proceed to download")
    if (completed_cleaning and uploaded_file):
        downloaded = st.download_button(
            label="Download data as CSV",
            data= csv_df,
            file_name='cleaned_data.csv'
        )