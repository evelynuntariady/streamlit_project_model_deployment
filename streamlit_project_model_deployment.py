import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load all necesssary files
tfidf_matrix = joblib.load('tfidfmatrix.pkl')
indices = joblib.load('indices.pkl') 
df = joblib.load('netflix_title.pkl')

#Calculate the cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get the recommendations
def get_recommendations(title, df, indices, cosine_sim, num_recommend = 5):
    title = title.lower()
    idx = indices.get(title)
    if idx is None:
        return None

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda df: df[1], reverse=True)
    top_similar = sim_scores[1:num_recommend+1]

    movie_indices = [i[0] for i in top_similar]
    ret_df = pd.DataFrame(df.iloc[movie_indices])
    ret_df['score'] = [i[1] for i in top_similar]
    return ret_df.reset_index(drop=True)

def main():
    st.title('Project Model Deployment - Recommendation System')
    st.text('Caroline Ang - 2702208606 \n Evelyn Caristy Untariady - 2702209496 \n Laurel Evelina Widjaja - 2702213770')

    # Input dari user
    st.write('Masukkan judul film atau acara TV Netflix yang kamu suka:')
    input_title = st.text_input("Contoh: The Irishman")

    # Button
    if st.button('Get Recommendations'):
        recommendations = get_recommendations(input_title, df, indices, cosine_sim)
            
        if recommendations is None:
            st.error(f"Film '{input_title}' tidak ditemukan dalam database.")
        else:
            st.success('Here are some recommendations for you!')
            st.dataframe(recommendations)

if __name__ == '__main__':
    main()
