import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle

# Load Dataset
metadata = pd.read_csv('movies_metadata.csv', low_memory = False)
metadata = metadata[:5000]
metadata['overview'] = metadata['overview'].fillna('');

# Word Vectorize
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Create movie list and pandas series
movies = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

# Compute cosine (similarity)
cosine_overview = linear_kernel(tfidf_matrix, tfidf_matrix)

# Dump movie list and cosine to pickle
pickle.dump((movies, cosine_overview), open('movies.p', 'wb'))