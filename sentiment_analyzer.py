
import re

positive_words = [
    "отлично", "хорошо", "прекрасно", "замечательно",
    "быстро", "качественно", "нравится", "доволен",
    "супер", "люблю", "great", "good", "excellent",
    "amazing", "love", "perfect"
]

negative_words = [
    "плохо", "ужасно", "медленно", "долго",
    "некачественно", "разочарован", "проблема",
    "сломался", "недоволен", "отвратительно",
    "bad", "terrible", "slow", "hate", "poor",
    "awful"
]

def analyze_sentiment(text):
    words = re.findall(r"[а-яёa-z]+", text.lower())

    positive = sum(word in positive_words for word in words)
    negative = sum(word in negative_words for word in words)

    if positive > negative:
        sentiment = "Положительный"
    elif negative > positive:
        sentiment = "Отрицательный"
    else:
        sentiment = "Нейтральный"

    return {
        "sentiment": sentiment,
        "positive_words": positive,
        "negative_words": negative
    }


if __name__ == "__main__":
    print("AI Review Analyzer")
    print("-----------------")

    review = input("Введите отзыв: ")
    result = analyze_sentiment(review)

    print("\nРезультат анализа:")
    print("Тональность:", result["sentiment"])
    print("Положительные слова:", result["positive_words"])
    print("Отрицательные слова:", result["negative_words"])
