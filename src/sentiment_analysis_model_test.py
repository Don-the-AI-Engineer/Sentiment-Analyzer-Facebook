import json
from transformers import pipeline
import pandas as pd

def load_reviews(file_path):
    with open(file_path, 'r') as file:
        reviews = json.load(file)
    return reviews

def analyze_sentiments(reviews):
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    # Mappinig the outputs to Negative, Neutral and Positive based on this models 5, star based outputs
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

def save_reviews_to_csv(reviews, output_file):
    df = pd.DataFrame(reviews)
    df.to_csv(output_file, index=False)
    print(f"Reviews saved to {output_file}")

if __name__ == "__main__":
    reviews = load_reviews('reviews.json')

    analyzed_reviews = analyze_sentiments(reviews)

    save_reviews_to_csv(analyzed_reviews, 'analyzed_reviews.csv')
