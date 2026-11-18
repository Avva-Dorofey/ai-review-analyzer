
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


# Russian positive reviews
ru_positive = [
    "Отличный товар, очень доволен покупкой",
    "Прекрасное качество, рекомендую всем",
    "Быстрая доставка и отличный сервис",
    "Мне очень понравился этот продукт",
    "Качественный товар, буду покупать еще",
    "Все работает идеально, спасибо",
    "Покупкой полностью доволен",
    "Очень хорошее качество за свою цену",
    "Продавец быстро ответил на вопросы",
    "Товар превзошел все мои ожидания",
    "Отличное обслуживание и приятный персонал",
    "Все прекрасно, обязательно закажу снова"
]

# Russian negative reviews
ru_negative = [
    "Ужасный товар, деньги потрачены зря",
    "Очень плохое качество, не рекомендую",
    "Доставка была долгой и неприятной",
    "Мне совершенно не понравилась покупка",
    "Товар сломался уже на следующий день",
    "Обслуживание было ужасным",
    "Качество не соответствует цене",
    "Продавец не отвечает на сообщения",
    "Покупкой очень разочарован",
    "Получил поврежденный товар",
    "Это худший продукт, который я покупал",
    "Больше никогда здесь не закажу"
]

# English positive reviews
en_positive = [
    "Excellent product, I am very happy with it",
    "Great quality, highly recommended",
    "Fast delivery and excellent customer service",
    "I really love this product",
    "Very good quality for the price",
    "Everything works perfectly",
    "I am completely satisfied with my purchase",
    "Amazing experience, thank you",
    "The seller was very helpful",
    "This product exceeded my expectations",
    "Very friendly staff and great service",
    "I will definitely buy again"
]

# English negative reviews
en_negative = [
    "Terrible product, a complete waste of money",
    "Very poor quality, not recommended",
    "Delivery was extremely slow",
    "I really dislike this product",
    "The product broke after one day",
    "Customer service was awful",
    "The quality does not match the price",
    "The seller never replied to my messages",
    "I am very disappointed with this purchase",
    "The item arrived damaged",
    "This is the worst product I have ever bought",
    "I will never order from here again"
]

# Combine all reviews and labels
reviews = ru_positive + ru_negative + en_positive + en_negative

sentiments = (
    ["positive"] * len(ru_positive)
    + ["negative"] * len(ru_negative)
    + ["positive"] * len(en_positive)
    + ["negative"] * len(en_negative)
)

data = {
    "review": reviews,
    "sentiment": sentiments
}

df = pd.DataFrame(data)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["review"],
    df["sentiment"],
    test_size=0.25,
    random_state=42,
    stratify=df["sentiment"]
)

# Create the machine learning pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print(f"Total reviews: {len(df)}")
print(f"Training reviews: {len(X_train)}")
print(f"Testing reviews: {len(X_test)}")
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# Save the trained model
with open("sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
