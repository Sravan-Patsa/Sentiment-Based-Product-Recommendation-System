import pandas as pd
import numpy as np
import pickle

lr_model = pickle.load(file = open("logistic_regression_model.pkl", 'rb'))

tfidf = pickle.load(file = open("tfidf.pkl", 'rb'))

recommender_model = pickle.load(file = open("user_based_recommender.pkl", 'rb'))

product_mapping = pickle.load(file = open("product_mapping.pkl", 'rb'))

df = pickle.load(file = open("df.pkl", 'rb'))

def getRecommendations(username):
    
    try:
        top20 = pd.DataFrame(recommender_model.loc[username]).reset_index()
    except KeyError:
        errorMessage = f'Incorrect username : "{username}"'
        return errorMessage, None

    top20.rename(columns = { top20.columns[1]: 'similarity_rating' }, inplace = True)
    top20 = top20.sort_values(by = 'similarity_rating', ascending = False)[0 : 20]

    top20 = pd.merge(top20, product_mapping, left_on = 'id', right_on = 'id', how = 'left')

    top20 = pd.merge(top20, df[['id', 'reviews_lemmatized']], left_on = 'id', right_on = 'id', how = 'left')

    top20_tfidf = tfidf.transform(top20['reviews_lemmatized'])

    top20['sentiment_pred'] = lr_model.predict(top20_tfidf)

    review_count = pd.DataFrame(top20.groupby('id')['sentiment_pred'].count()).reset_index()
    review_count.columns = ['id', 'no_of_reviews']

    pos_review_count = pd.DataFrame(top20.groupby('id')['sentiment_pred'].sum()).reset_index()
    pos_review_count.columns = ['id', 'no_of_pos_reviews']

    product_reviews = pd.merge(review_count, pos_review_count, left_on = 'id', right_on = 'id', how = 'left')

    product_reviews['prod_rating'] = round((product_reviews.no_of_pos_reviews/product_reviews.no_of_reviews) * 100, 2)
    
    product_reviews = product_reviews.sort_values(by = 'prod_rating', ascending = False)

    top5 = pd.merge(product_reviews, product_mapping, left_on = 'id', right_on = 'id', how = 'left')

    products = top5['name'][0:5].tolist()
    rating = top5['prod_rating'][0:5].tolist()

    return products, rating

