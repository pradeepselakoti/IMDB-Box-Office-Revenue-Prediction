import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import streamlit as st

st.set_page_config(page_title="IMDb Box Office Revenue Prediction App", page_icon=":movie_camera:", layout="wide")

page_bg_image = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url(https://images.pexels.com/videos/3045163/free-video-3045163.jpg?auto=compress&cs=tinysrgb&dpr=1&w=500);
    background-size: cover;
    #opacity: 0.7;
}
</style>
"""
st.markdown(page_bg_image, unsafe_allow_html=True)


st.markdown('''
<style>
.my-div1 {
    display: inline-block;
    #border: 1px solid black;
    #-webkit-text-stroke: 0.5px black;
    #background-color: rgba(0, 0, 0, 0.75);
    font-size: 40px;
    font-weight: bold;
}

.my-div2 {
    display: inline-block;
    #border: 1px solid black;
    #-webkit-text-stroke: 0.5px black;
    color: white;
    #background-color: rgba(0, 0, 0, 0.75);
    font-size: 30px;
    font-weight: bold;
}

.my-div3 {
    display: inline-block;
    #border: 1px solid black;
    #-webkit-text-stroke: 0.5px black;
    color: white;
    font-size: 20px;
    font-weight: bold;
    #background-color: rgba(0, 0, 0, 0.75);
}

.my-div4 {
    display: inline-block;
    #border: 1px solid black;
    #-webkit-text-stroke: 0.5px black;
    color: white;
    font-size: 20px;
    font-weight: bold;
    background-color: rgba(0, 0, 0, 0.75);
}

</style>
''', unsafe_allow_html=True)

@st.cache_data()
def load_model():
    model = RandomForestRegressor(random_state=1)
    trainData = pd.read_csv('imdb_data_for_webapp.csv')
    features=['budget','popularity',"runtime","size_of_cast","number_of_spoken_languages","num_production_companies","genre_Count",'release_year']
    data = trainData[features]
    target = pd.read_csv('imdb_targeted_value_for_webapp.csv').values.ravel()
    model.fit(data, target)
    return model, features

model, features = load_model()

def predict(input_dict):
    input_df = pd.DataFrame(input_dict, index=[0])
    pred = model.predict(input_df[features])[0]
    return pred

def main():
    st.markdown("<div class='my-div1'>IMDb Box Office Revenue Prediction Tool</div>", unsafe_allow_html=True)
    st.write(""" ###### """ )
    st.markdown('<div class="my-div2">This app can predict the Movie&apos;s Revenue based on the metadata.</div>', unsafe_allow_html=True)
    st.write(""" ###### """ )
    st.markdown('<div class="my-div3">The dataset contains movie statistics of 5000 movies.</div>', unsafe_allow_html=True)
    st.write(""" ###### """)


    st.sidebar.header("User Input Features")
    selected_year = st.sidebar.selectbox("Select Release Year", options=range(2000, 2020))
    budget = st.sidebar.number_input("Budget in Dollars($)", min_value=10000, max_value=500000000, step=10000, value=1000000)
    popularity = st.sidebar.slider("Popularity", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    runtime = st.sidebar.slider("Runtime", min_value=1, max_value=300, step=1, value=90)
    size_of_cast = st.sidebar.slider("Size of Cast", min_value=1, max_value=25, step=1, value=5)
    num_spoken_languages = st.sidebar.slider("Number of Spoken Languages", min_value=1, max_value=10, step=1, value=1)
    num_production_companies = st.sidebar.slider("Number of Production Companies", min_value=1, max_value=10, step=1, value=1)
    genre_select = st.multiselect("Select Genre", options=['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy'])
    genre_count = len(genre_select)
    input_dict = {
        'budget': budget,
        'popularity': popularity,
        'runtime': runtime,
        'size_of_cast': size_of_cast,
        'number_of_spoken_languages': num_spoken_languages,
        'num_production_companies': num_production_companies,
        'genre_Count': genre_count,
        'release_year': selected_year
    }

    if st.sidebar.button("Predict Revenue"):
        output = predict(input_dict)
        st.success(f"The predicted Revenue for the movie is ${output:,.2f}")

if __name__ == "__main__":
    main()

st.write(""" ###### """)

st.markdown('''
<div class="my-div4">
    <h2>Team Members:</h2>
    <ul>
        <li>Mehul Basera (2161231)</li>
        <li>Adarsh Bisht (2161061)</li>
        <li>Mohit Datt Joshi (2161235)</li>
    </ul>
</div>
''', unsafe_allow_html=True)

st.write(""" ###### """)

st.write("""
##### * Dataset used for our Project * :- [Kaggle - TMDB 5000 Movie Dataset](https://www.kaggle.com/competitions/tmdb-box-office-prediction/data?select=test.csv)
""")
