from __future__ import unicode_literals

import datetime
import errno
import json
import os
import sys
import tempfile
from argparse import ArgumentParser
from dotenv import load_dotenv
import pickle

from flask import Flask, request, abort, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    LineBotApiError, InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    SourceUser, TemplateSendMessage, ImageCarouselTemplate, 
    ImageCarouselColumn, PostbackAction, PostbackEvent,)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1)

# reads the key-value pair from .env file and adds them to environment variable.
load_dotenv()

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)
if channel_secret is None or channel_access_token is None:
    print('Specify LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN as environment variables.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

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

@app.route("/", methods=['GET'])
def home():
    return "Movie Recommendation API"

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except LineBotApiError as e:
        print("Got exception from LINE Messaging API: %s\n" % e.message)
        for m in e.error.details:
            print("  %s: %s" % (m.property, m.message))
        print("\n")
    except InvalidSignatureError:
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    text = event.message.text

    if text == 'profile':
        if isinstance(event.source, SourceUser):
            profile = line_bot_api.get_profile(event.source.user_id)
            line_bot_api.reply_message(
                event.reply_token, [
                    TextSendMessage(text='Display name: ' + profile.display_name),
                    TextSendMessage(text='Status message: ' + str(profile.status_message))
                ]
            )
        else:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="Bot can't use profile API without user ID"))
    elif str(text).lower() == 'movies':
        image_carousel_template = ImageCarouselTemplate(columns=[
            ImageCarouselColumn(image_url='https://m.media-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@._V1_UY1200_CR87,0,630,1200_AL_.jpg',
                                action=PostbackAction(label='Select', data='Toy Story')),
            ImageCarouselColumn(image_url='https://m.media-amazon.com/images/M/MV5BM2MyNjYxNmUtYTAwNi00MTYxLWJmNWYtYzZlODY3ZTk3OTFlXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_UY1200_CR107,0,630,1200_AL_.jpg',
                                action=PostbackAction(label='Select', data='The Godfather')),
            ImageCarouselColumn(image_url='https://m.media-amazon.com/images/M/MV5BMTY4ODM0OTc2M15BMl5BanBnXkFtZTcwNzE0MTk3OA@@._V1_.jpg',
                                action=PostbackAction(label='Select', data='Die Hard')),
        ])
        template_message = TemplateSendMessage(
            alt_text='ImageCarousel alt text', template=image_carousel_template)
        line_bot_api.reply_message(event.reply_token, template_message)

@handler.add(PostbackEvent)
def handle_postback(event):
    movie = event.postback.data
    movie_recommend = get_recommendations(movie)
    print(movie_recommend)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=movie_recommend))

def get_recommendations(title, cosine_sim=cosine_overview):
    idx = movies[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indics = [i[0] for i in sim_scores]
    recommend_text = 'Your Recommendations for the Movie ' + title + ' are:\n'
    recommend_text += '--------------\n'
    for movie in movies[movie_indics].index:
        recommend_text += movie + '\n'
    recommend_text += '--------------'
    return recommend_text

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
