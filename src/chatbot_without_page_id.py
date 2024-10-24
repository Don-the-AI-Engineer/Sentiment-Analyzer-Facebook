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

USE_FACEBOOK_API = False
FB_ACCESS_TOKEN = 'your_facebook_access_token_here'

def load_reviews_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            reviews = json.load(file)
        if not reviews:
            raise ValueError("No reviews found in the JSON file.")
        return reviews
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading reviews from JSON: {e}")
        return []

def get_facebook_page_id(company_name, access_token):
    search_url = f"https://graph.facebook.com/v12.0/search"
    params = {
        "q": company_name,
        "type": "page",
        "access_token": access_token
    }
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            return data['data'][0]['id']
        else:
            print(f"No Facebook page found for {company_name}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error searching for Facebook page ID: {e}")
        return None

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
        if not reviews:
            raise ValueError("No reviews found from Facebook API.")
        return reviews
    except requests.exceptions.RequestException as e:
        print(f"Error fetching reviews from Facebook: {e}")
        return []
    except ValueError as e:
        print(e)
        return []

def analyze_sentiments(reviews):
    try:
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
    except Exception as e:
        print(f"Error analyzing sentiments: {e}")
        return []

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
    try:
        df = pd.DataFrame(reviews)
        df.to_csv(output_file, index=False)
        print(f"Reviews saved to {output_file}")
    except Exception as e:
        print(f"Error saving reviews to CSV: {e}")

def get_most_representative_review(reviews, theme):
    try:
        review_texts = [review['review_text'] for review in reviews]
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(review_texts)
        theme_vector = tfidf_vectorizer.transform([theme])
        similarities = cosine_similarity(theme_vector, tfidf_matrix).flatten()
        most_representative_index = np.argmax(similarities)
        return reviews[most_representative_index]
    except Exception as e:
        print(f"Error selecting most representative review: {e}")
        return reviews[0] if reviews else None

def analyze_company_reviews(company_name):
    if USE_FACEBOOK_API:
        page_id = get_facebook_page_id(company_name, FB_ACCESS_TOKEN)
        if page_id:
            reviews = load_reviews_from_facebook(page_id, FB_ACCESS_TOKEN)
        else:
            return f"No Facebook page found for {company_name}.", None, None
    else:
        reviews = load_reviews_from_json('reviews.json')

    if not reviews:
        return "No reviews found. Please check the data source or try again later.", None, None

    company_reviews = [review for review in reviews if review['company'].lower() == company_name.lower()]
    if not company_reviews:
        return f"No reviews found for {company_name}.", None, None

    analyzed_reviews = analyze_sentiments(company_reviews)
    if not analyzed_reviews:
        return "Error analyzing sentiments. Please try again later.", None, None

    themes = extract_themes(analyzed_reviews)
    theme_counts = Counter(themes)
    top_themes = theme_counts.most_common(3)

    csv_file_path = 'analyzed_reviews.csv'
    save_reviews_to_csv(analyzed_reviews, csv_file_path)

    sentiment_counts = pd.DataFrame(analyzed_reviews)['sentiment'].value_counts()
    average_rating = pd.DataFrame(analyzed_reviews)['rating'].mean()

    top_sentiments = []
    response_text = f"Analyzed the latest reviews for {company_name}...\n\nBased on {company_name}'s Facebook reviews, here are the top 3 sentiments:\n"
    for theme, count in top_themes:
        theme_reviews = [review for review in analyzed_reviews if theme in review['review_text'].lower()]
        if theme_reviews:
            representative_review = get_most_representative_review(theme_reviews, theme)
            if representative_review:
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

def chatbot_interface(company_name):
    response_text, csv_file_path, plot_path = analyze_company_reviews(company_name)
    return response_text, plot_path, csv_file_path

if __name__ == "__main__":
    interface = gr.Interface(
        fn=chatbot_interface,
        inputs=gr.Textbox(label='Company Name'),
        outputs=[gr.Textbox(label='Chatbot Response', lines=10), gr.Image(label='Sentiment Analysis Graph'), gr.File(label='Download Analyzed Reviews CSV')],
        title="Facebook Reviews Sentiment Analyzer Chatbot",
        description="Enter the company name to analyze reviews and view sentiment analysis.",
        live=True
    )

    interface.launch()
