
import streamlit as st
import pickle

st.set_page_config(
    page_title="AI Review Analyzer",
    page_icon="🤖",
    layout="centered"
)

@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as file:
        return pickle.load(file)


model = load_model()

st.title("🤖 AI Review Analyzer")

st.write(
    "Analyze customer reviews using Machine Learning "
    "and Natural Language Processing."
)

st.markdown("---")

review = st.text_area(
    "Enter your customer review:",
    placeholder="Write your review here...",
    height=150
)

if st.button("Analyze Review", type="primary"):
    if review.strip():
        prediction = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]
        confidence = max(probabilities)

        st.subheader("Analysis Results")

        if prediction == "positive":
            st.success("Sentiment: POSITIVE")
        else:
            st.error("Sentiment: NEGATIVE")

        st.metric(
            "Model Confidence",
            f"{confidence:.2%}"
        )

        st.caption(
            "This is an educational ML prototype. "
            "Predictions are not guaranteed to be correct."
        )
    else:
        st.warning("Please enter a review first.")

st.markdown("---")
st.caption("AI Review Analyzer | Machine Learning Project")
