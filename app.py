import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import joblib
import altair as alt
from PIL import Image

######################
# Stock Prices Project
######################

def display_stock_data(company):
    st.write(f"### Stock Prices for {company}")
    data = yf.download(company, start="2000-01-01", end="2024-01-01")
    st.line_chart(data[['Close', 'Volume']])

######################
# DNA Project
######################

def DNA_nucleotide_count(seq):
    d = dict([
        ('A', seq.count('A')),
        ('T', seq.count('T')),
        ('G', seq.count('G')),
        ('C', seq.count('C'))
    ])
    return d

def display_DNA_nucleotide_count(sequence):
    st.header('OUTPUT (DNA Nucleotide Count)')

    X = DNA_nucleotide_count(sequence)

    st.subheader('1. Print dictionary')
    X

    st.subheader('2. Print text')
    st.write('There are  ' + str(X['A']) + ' adenine (A)')
    st.write('There are  ' + str(X['T']) + ' thymine (T)')
    st.write('There are  ' + str(X['G']) + ' guanine (G)')
    st.write('There are  ' + str(X['C']) + ' cytosine (C)')

######################
# EDA Basketball Project
######################

def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)  # Deletes repeating headers in content
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    return playerstats

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

######################
# Penguin Prediction Project
######################

def penguin_prediction():
    st.write("""
    # Penguin Prediction App

    This app predicts the Palmer Penguin species!

    Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
    """)

    st.sidebar.header('User Input Features')

    st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
    """)

    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
            sex = st.sidebar.selectbox('Sex', ('male', 'female'))
            bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
            bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
            flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
            body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
            data = {'island': island,
                    'bill_length_mm': bill_length_mm,
                    'bill_depth_mm': bill_depth_mm,
                    'flipper_length_mm': flipper_length_mm,
                    'body_mass_g': body_mass_g,
                    'sex': sex}
            features = pd.DataFrame(data, index=[0])
            return features

        input_df = user_input_features()

    # Combines user input features with entire penguins dataset
    # This will be useful for the encoding phase
    penguins_raw = pd.read_csv('penguin/penguins_cleaned.csv')  # Change path here
    penguins = penguins_raw.drop(columns=['species'])
    df = pd.concat([input_df, penguins], axis=0)

    # Encoding of ordinal features
    # https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
    encode = ['sex', 'island']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        del df[col]
    df = df[:1]  # Selects only the first row (the user input data)

    # Displays the user input features
    st.subheader('User Input features')

    if uploaded_file is not None:
        st.write(df)
    else:
        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
        st.write(df)

    # Reads in saved classification model
    load_clf = joblib.load(open('penguin/penguins_clf.joblib', 'rb'))


    # Apply model to make predictions
    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)

    st.subheader('Prediction')
    penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
    st.write(penguins_species[prediction])

    st.subheader('Prediction Probability')
    st.write(prediction_proba)

######################
# Main App
######################

def main():
    st.title('Combined Web App CO21325')

    st.write("""

    
    This app combines multiple projects.
    
    ---
    """)

    selected_project = st.sidebar.selectbox("Select Project", ["Stock Prices", "DNA Project", "EDA Basketball", "Penguin Prediction"])

    if selected_project == "Stock Prices":
        st.write("""
        # Stock Viewer App
        
        This app shows the stock prices of a chosen company.
        """)
        
        selected_company = st.selectbox("Choose a company", ["AAPL", "GOOGL", "MSFT", "AMZN", "FB"])
        
        display_stock_data(selected_company)

    elif selected_project == "DNA Project":
        st.write("""
        # DNA Nucleotide Count Web App
        
        This app counts the nucleotide composition of query DNA!
        
        ---
        """)

        sequence_input = ">DNA Query 2\nGAACACGTGGAGGCAAACAGGAAGGTGAAGAAGAACTTATCCTATCAGGACGGAAGGTCCTGTGCTCGGG\nATCTTCCAGACGTCGCGACTCTAAATTGCCCCCTCTGAGGTCAAGGAACACAAGATGGTTTTGGAAATGC\nTGAACCCGATACATTATAACATCACCAGCATCGTGCCTGAAGCCATGCCTGCTGCCACCATGCCAGTCCT"
        sequence = st.text_area("Sequence input", sequence_input, height=250)
        sequence = sequence.splitlines()
        sequence = sequence[1:]
        sequence = ''.join(sequence)

        st.write("""---""")

        st.header('INPUT (DNA Query)')
        sequence

        display_DNA_nucleotide_count(sequence)
        X = DNA_nucleotide_count(sequence)
        
        ### 2. Print text
        st.subheader('2. Print text')
        st.write('There are  ' + str(X['A']) + ' adenine (A)')
        st.write('There are  ' + str(X['T']) + ' thymine (T)')
        st.write('There are  ' + str(X['G']) + ' guanine (G)')
        st.write('There are  ' + str(X['C']) + ' cytosine (C)')

        ### 3. Display DataFrame
        st.subheader('3. Display DataFrame')
        df = pd.DataFrame.from_dict(X, orient='index')
        df = df.rename({0: 'count'}, axis='columns')
        df.reset_index(inplace=True)
        df = df.rename(columns = {'index':'nucleotide'})
        st.write(df)

        ### 4. Display Bar Chart using Altair
        st.subheader('4. Display Bar chart')
        p = alt.Chart(df).mark_bar().encode(
            x='nucleotide',
            y='count'
        )
        p = p.properties(
            width=alt.Step(80)  # controls width of bar.
        )
        st.write(p)


    elif selected_project == "EDA Basketball":
        st.write("""
        # NBA Player Stats Explorer
        
        This app performs simple webscraping of NBA player stats data!
        * **Python libraries:** base64, pandas, streamlit
        * **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
        """)

        selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2020))))

        playerstats = load_data(selected_year)

        sorted_unique_team = sorted(playerstats.Tm.unique())
        selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

        unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
        selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

        df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

        st.header('Display Player Stats of Selected Team(s)')
        st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(
            df_selected_team.shape[1]) + ' columns.')
        st.dataframe(df_selected_team)

        st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)

        if st.button('Intercorrelation Heatmap'):
            st.header('Intercorrelation Matrix Heatmap')
            df_selected_team.to_csv('output.csv', index=False)
            df = pd.read_csv('output.csv')

            # Select only numeric columns
            numeric_cols = df.select_dtypes(include=np.number)

            corr = numeric_cols.corr()
            mask = np.zeros_like(corr)
            mask[np.triu_indices_from(mask)] = True
            with sns.axes_style("white"):
                f, ax = plt.subplots(figsize=(7, 5))
                ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
            st.pyplot()

    elif selected_project == "Penguin Prediction":
        penguin_prediction()

if __name__ == "__main__":
    main()
