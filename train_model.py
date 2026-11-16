
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

data = {
    "review": [
        "Отличный товар, очень доволен покупкой",
        "Прекрасное качество, рекомендую всем",
        "Быстрая доставка и отличный сервис",
        "Мне очень понравилось, все супер",
        "Качественный продукт, буду покупать еще",
        "Ужасный товар, деньги потрачены зря",
        "Очень плохое качество, не рекомендую",
        "Доставка была ужасной и долгой",
        "Я разочарован этой покупкой",
        "Товар сломался через один день",
        "Все отлично, спасибо продавцу",
        "Худшая покупка в моей жизни",
        "Прекрасный продукт, все работает",
        "Очень медленная доставка",
        "Качество просто отвратительное",
        "Товар хороший, но доставка плохая",
        "Мне понравилось обслуживание",
        "Не доволен качеством товара",
        "Все работает идеально",
        "Ужасный сервис и плохое отношение"
    ],
    "sentiment": [
        "positive", "positive", "positive", "positive", "positive",
        "negative", "negative", "negative", "negative", "negative",
        "positive", "negative", "positive", "negative", "negative",
        "negative", "positive", "negative", "positive", "negative"
    ]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df["review"],
    df["sentiment"],
    test_size=0.25,
    random_state=42,
    stratify=df["sentiment"]
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print(f"Accuracy: {accuracy:.2%}")
print(classification_report(y_test, predictions))

with open("sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
