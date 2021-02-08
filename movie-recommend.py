import pickle

# Load movie list and cosine from pickle
movies, cosine_overview = pickle.load(open('movies.p', 'rb'))

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

text_recomend = get_recommendations('The Godfather')

print(text_recomend)
