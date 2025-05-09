## Streamlit application for Song Recommendation System

## 1. Import Libraries
import pandas as pd
import numpy as np
import os
import streamlit as st
import sklearn
import umap
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components

## 2. Set up Streamlit Application UI
st.set_page_config(
    page_title="Song Recommendation Application",
    page_icon="🎵",
    layout="wide",
)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("song recommendation app image.jpg", use_container_width=True)

# st.image("song recommendation app image.jpg", width= 500)
st.title("Discover Your Next Favorite Song!")

# Custom CSS to override the background color
##1DB954;  /* Green background */

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FF474C;  /* Red background */
    }
    </style>
    """,
    unsafe_allow_html=True
)



## 3. Import song csv file and edit file
songs = pd.read_csv('genres_v2.csv', dtype={19: str})
songs['duration_minutes'] = songs['duration_ms'] / 1000 / 60 #Convert to seconds, then minutes
songs = songs.drop(columns=['type', 'id', 'uri', 'track_href', 'analysis_url', 'Unnamed: 0', 'title', 'duration_ms'], axis=1)
songs = songs[songs['song_name'].notna()]
songs.drop_duplicates(subset=['song_name'], keep='last', inplace=True)
print(songs.head(2))


song_names_db = songs['song_name'].tolist()
song_names_db.sort()
selected_song = st.selectbox(
    "Search and select a song:",
    options=song_names_db
)

st.write("You entered:", selected_song)
st.write("🔍 **Finding your top 3 matches...**")
# loader = st.empty()
# emojis = ["🎵", "🎶", "🎼", "🎧", "🎷", "🎸", "🥁", "🎹"]



## 4. Use ML to find the best song recommendation to user's song choice
def song_recommendation_nn(df, song_input):
    df = df.reset_index(drop=True)
    X = df.drop(['genre', 'song_name'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mapper = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit(X_scaled)
    embedding = mapper.transform(X_scaled)
    umap_df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])

    # Add 'genre' column to UMAP DataFrame for color mapping
    umap_df['genre'] = df['genre']
    umap_df['song_name'] = df['song_name']
    # highlighted_song = umap_df[umap_df['song_name'] == song_input]

    #Find and Print nearest neighbors for each highlighted song
    nearest_n_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nearest_n_model.fit(embedding)

    song_index = umap_df.index[umap_df['song_name'] == song_input].tolist()[0]
    distances, indices = nearest_n_model.kneighbors([embedding[song_index]])
    nearest_neighbors = umap_df['song_name'].iloc[indices[0][:]].tolist()
    nearest_neighbors_genres = umap_df['genre'].iloc[indices[0][:]].tolist()
    print(f"Nearest neighbors for {song_input}: {nearest_neighbors[1:]}")
    print()


    cols = st.columns(3)  # create 3 columns

    for idx, song_name in enumerate(nearest_neighbors[1:][0:3]):
        with cols[idx]:
            st.markdown(
    f"""
    <div style='background-color:#fbeff2; padding:10px 20px; border-radius:8px; margin-bottom:10px;'>
        <h2 style='color:#FF474C; font-family:sans-serif; margin-bottom:0;'>
         {idx+1}
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)
            # st.markdown(f"## Recommendation {idx+1}")
            st.markdown(f"### 🎵 {song_name}")
    
    for idx, g_name in enumerate(nearest_neighbors_genres[1:][0:3]):
        with cols[idx]:
            st.markdown(f"### 🎼 Genre: {g_name}")
    
    # st.success("We hope these songs are to your liking! 🌟")

song_recommendation_nn(songs, selected_song)











#cd Documents
#cd Datasets
#streamlit run "Song Recommendation App.py"