import streamlit as st
from transformers import pipeline

# Load pre-trained sentiment-analysis pipeline from Hugging Face's transformers
sentiment_analyzer = pipeline("sentiment-analysis")

# Function to analyze sentiment
def analyze_sentiment(conversation):
    analysis_results = sentiment_analyzer(conversation)
    return analysis_results

# Streamlit app layout
def main():
    st.title("Conversation Sentiment & Tone Analyzer")

    # Text input box for user to paste conversations
    conversation_thread = st.text_area("Paste conversation thread here:")

    # When the user clicks the "Analyze" button
    if st.button("Analyze"):
        if conversation_thread:
            # Analyze sentiments of the conversation
            sentiments = analyze_sentiment(conversation_thread)
            
            # Show results
            for i, sentiment in enumerate(sentiments):
                st.write(f"Sentence {i+1}: {sentiment['label']} (Confidence: {sentiment['score']:.2f})")
        else:
            st.write("Please input a conversation thread.")

if __name__ == "__main__":
    main()
