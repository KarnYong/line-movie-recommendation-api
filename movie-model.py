import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle

# Load Dataset
metadata = pd.read_csv('movies_metadata.csv', low_memory = False)
metadata = metadata[:20000]
metadata['overview'] = metadata['overview'].fillna('');

# Word Vectorize
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Create movie list and pandas series
movies = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

# Compute cosine (similarity)
cosine_overview = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_overview):
    idx = movies[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indics = [i[0] for i in sim_scores]
    recommend_text = 'Your Recommendations for the Movie ' + title + ' are:\n'
    for movie in movies[movie_indics].index:
        recommend_text += movie + '\n'
    recommend_text += '--------------'
    return recommend_text

text_recomend = get_recommendations('Die Hard')

print(text_recomend)
