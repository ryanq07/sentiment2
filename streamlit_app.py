import streamlit as st
from transformers import pipeline

# Load pre-trained sentiment-analysis pipeline from Hugging Face's transformers
sentiment_analyzer = pipeline("sentiment-analysis")

# Function to analyze sentiment
def analyze_sentiment(conversation):
    return sentiment_analyzer(conversation)

# Function to provide advice based on sentiment
def provide_advice(sentiment_label):
    if sentiment_label == "POSITIVE":
        return "Great job! Keep the positive tone, and continue to express enthusiasm or encouragement."
    elif sentiment_label == "NEGATIVE":
        return "Consider rephrasing to sound more constructive. Try using empathetic or encouraging language."
    elif sentiment_label == "NEUTRAL":
        return "Neutral tone is fine, but if possible, engage more by showing empathy or positivity to build rapport."
    else:
        return "Monitor the tone and try to keep responses supportive and constructive."

# Streamlit app layout
def main():
    st.title("Conversation Sentiment & Tone Analyzer with Improvement Advice")

    # Text input box for user to paste conversations
    conversation_thread = st.text_area("Paste conversation thread here:")

    # When the user clicks the "Analyze" button
    if st.button("Analyze"):
        if conversation_thread:
            # Analyze sentiments of the conversation
            sentiments = analyze_sentiment(conversation_thread)

            # Show results and provide advice
            for i, sentiment in enumerate(sentiments):
                label = sentiment['label']
                score = sentiment['score']
                
                st.write(f"**Sentence {i+1}:** {label} (Confidence: {score:.2f})")
                
                # Provide advice based on sentiment
                advice = provide_advice(label)
                st.write(f"**Advice:** {advice}")
                st.write("---")
        else:
            st.write("Please input a conversation thread.")

if __name__ == "__main__":
    main()
