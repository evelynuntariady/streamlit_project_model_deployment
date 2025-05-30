import streamlit as st
import joblib
import scipy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_cosine_sim():
    tfidf = scipy.sparse.load_npz('tfidf_matrix.npz')
    return cosine_similarity(tfidf, tfidf)

@st.cache_resource
def load_indices():
    return joblib.load('indices.pkl')

@st.cache_data
def load_df():
    return joblib.load('netflix_title.pkl')

cosine_sim = load_cosine_sim()
indices = load_indices()
df = load_df()

def get_recommendations(title, df, indices, cosine_sim, num_recommend = 5):
    title = title.lower()
    idx = indices.get(title)
    if idx is None:
        return None

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_similar = sim_scores[1:num_recommend+1]

    movie_indices = [i[0] for i in top_similar]
    ret_df = pd.DataFrame(df.iloc[movie_indices])
    ret_df['score'] = [i[1] for i in top_similar]
    return ret_df.reset_index(drop=True)

def main():
    st.title('Project Model Deployment - Recommendation System')
    st.text('Caroline Ang - 2702208606 \n Evelyn Caristy Untariady - 2702209496 \n Laurel Evelina Widjaja - 2702213770')

    st.write('Masukkan judul film atau acara TV Netflix yang kamu suka:')
    input_title = st.text_input("Contoh: The Irishman")

    if st.button('Get Recommendations'):
        if input_title.strip() == "":
            st.warning("Masukkan judul terlebih dahulu!")
        else:
            recommendations = get_recommendations(input_title, df, indices, cosine_sim)
            if recommendations is None:
                st.error(f"Film '{input_title}' tidak ditemukan dalam database.")
            else:
                st.success('Here are some recommendations for you!')
                st.dataframe(recommendations)

if __name__ == '__main__':
    main()
