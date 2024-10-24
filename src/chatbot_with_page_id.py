import json
from transformers import pipeline
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests


USE_FACEBOOK_API = False # Set True to switch to using Facebook API
FB_ACCESS_TOKEN = 'facebook_access_token'
FB_PAGE_ID = 'facebook_page_id'

def load_reviews_from_json(file_path):
    with open(file_path, 'r') as file:
        reviews = json.load(file)
    return reviews

def load_reviews_from_facebook(page_id, access_token):
    url = f"https://graph.facebook.com/v12.0/{page_id}/ratings?access_token={access_token}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        reviews = []
        for review in data.get('data', []):
            reviews.append({
                "company": "Facebook",
                "review_text": review.get('review_text', ''),
                "rating": review.get('rating', 3),
                "created_time": review.get('created_time', '')
            })
        return reviews
    except requests.exceptions.RequestException as e:
        print(f"Error fetching reviews from Facebook: {e}")
        return []

'''
Used a Hugging Face Sentiment analysis model.
    -- uses a star rating method.
    -- highly accurate on facebook inputs.
    
Tested several other models.
    1. distilbert-base-uncased-finetuned-sst-2-english
        -- only "positive" and "negative"
    2. finiteautomata/bertweet-base-sentiment-analysis
        -- trained on twitter data, so fb data outputs are not accurate
'''
 
def analyze_sentiments(reviews):
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    label_mapping = {
        1: "NEGATIVE",
        2: "NEGATIVE",
        3: "NEUTRAL",
        4: "POSITIVE",
        5: "POSITIVE"
    }
    for review in reviews:
        sentiment = sentiment_pipeline(review['review_text'])[0]
        review['sentiment'] = label_mapping[int(sentiment['label'][0])]
        review['score'] = sentiment['score']
    return reviews

# Several pre-defined themes are used
def extract_themes(reviews):
    themes = []
    keywords = ["network", "service", "price", "quality", "customer", "support", "speed", "connection"]
    for review in reviews:
        review_text = review['review_text'].lower()
        for keyword in keywords:
            if re.search(rf'\b{keyword}\b', review_text):
                themes.append(keyword)
    return themes

def save_reviews_to_csv(reviews, output_file):
    df = pd.DataFrame(reviews)
    df.to_csv(output_file, index=False)
    print(f"Reviews saved to {output_file}")

def get_company_reviews(reviews, company_name):
    return [review for review in reviews if review['company'].lower() == company_name.lower()]

'''
TF-IDF and cosine similarity are used here to get the most representative review.

TF_IDF (Term Frequency - Inverse Document Frequency)
    TF - Measures the frequency of a word in a doc.
    IDF - Measures its importance by finding, how much this word has been used
    TF-IDF is the product of the above two values so that we can statistically determine the importance of a word.
    
Cosine Similarity
    Measures the similarity between two vectors. This is one of the first aproaches used in Face Recogntion also.

TF-IDF vectors are calculated first, the Cosine Similarity is used to calculate how much similar each review is to all others.
'''
def get_most_representative_review(reviews, theme):
    review_texts = [review['review_text'] for review in reviews]
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(review_texts)
    theme_vector = tfidf_vectorizer.transform([theme])
    similarities = cosine_similarity(theme_vector, tfidf_matrix).flatten()
    most_representative_index = np.argmax(similarities)
    return reviews[most_representative_index]

def analyze_company_reviews(company_name):
    if USE_FACEBOOK_API:
        reviews = load_reviews_from_facebook(FB_PAGE_ID, FB_ACCESS_TOKEN)
    else:
        reviews = load_reviews_from_json('reviews.json') # Using this for now because of the facebook ban.

    if not reviews:
        return "No reviews found. Please check the data source or try again later.", None, None

    company_reviews = get_company_reviews(reviews, company_name)
    if not company_reviews:
        return f"No reviews found for {company_name}.", None, None

    analyzed_reviews = analyze_sentiments(company_reviews)
    themes = extract_themes(analyzed_reviews)
    theme_counts = Counter(themes)
    top_themes = theme_counts.most_common(3)
    csv_file_path = 'analyzed_reviews.csv'
    save_reviews_to_csv(analyzed_reviews, csv_file_path)
    sentiment_counts = pd.DataFrame(analyzed_reviews)['sentiment'].value_counts()
    average_rating = pd.DataFrame(analyzed_reviews)['rating'].mean()
    
    # Chatbot response
    top_sentiments = []
    response_text = f"Analyzed the latest reviews for {company_name}...\n\nBased on {company_name}'s Facebook reviews, here are the top 3 sentiments:\n"
    for theme, count in top_themes:
        theme_reviews = [review for review in analyzed_reviews if theme in review['review_text'].lower()]
        if theme_reviews:
            representative_review = get_most_representative_review(theme_reviews, theme)
            example_review = representative_review['review_text']
            sentiment = representative_review['sentiment']
            confidence_scores = [review['score'] for review in theme_reviews]
            average_confidence = np.mean(confidence_scores)
            top_sentiments.append({
                "category": theme,
                "sentiment": sentiment.lower(),
                "confidence": average_confidence,
                "themes": [theme],
                "example_review": example_review
            })
            response_text += f"\n{len(top_sentiments)}. {theme.title()}\nSentiment: {sentiment.capitalize()} ({average_confidence * 100:.0f}% confidence)\nExample review: \"{example_review}\"\n"
    response_text += f"\nAverage rating: {'⭐' * int(round(average_rating))} ({average_rating:.1f}/5) from {len(analyzed_reviews)} reviews"
    output = {
        "company_name": company_name,
        "total_reviews": len(analyzed_reviews),
        "average_rating": average_rating,
        "top_sentiments": top_sentiments
    }
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=['red', 'blue', 'green'])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title(f'Sentiment Analysis for {company_name}')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('sentiment_plot.png')
    return response_text, csv_file_path, 'sentiment_plot.png'

# Gradio interface for chatbot interaction
def chatbot_interface(company_name):
    if company_name.lower() in ['dialog', 'mobitel', 'slt', 'elephant house']:
        response_text, csv_file_path, plot_path = analyze_company_reviews(company_name)
        return response_text, plot_path, csv_file_path
    else:
        return "Invalid company name. Please enter one of the following: Dialog, Mobitel, SLT, Elephant House.", None, None

if __name__ == "__main__":
    interface = gr.Interface(
        fn=chatbot_interface,
        inputs=gr.Textbox(label='Company Name'),
        outputs=[gr.Textbox(label='Chatbot Response', lines=10), gr.Image(label='Sentiment Analysis Graph'), gr.File(label='Download Analyzed Reviews CSV')],
        title="Facebook Reviews Sentiment Analyzer Chatbot",
        description="Enter the company name (Dialog, Mobitel, SLT, Elephant House) to analyze reviews and view sentiment analysis.",
        live=True
    )
    interface.launch()
