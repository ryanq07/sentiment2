import streamlit as st
import pandas as pd
from transformers import pipeline
from textblob import TextBlob

# Load the sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Function to analyze conversation and infer sentiment/tone
def analyze_conversation(conversation):
    # Analyze each message in the conversation thread
    analysis_results = []
    for idx, message in enumerate(conversation):
        sentiment_result = sentiment_analyzer(message)[0]  # using transformers pipeline for sentiment
        blob = TextBlob(message)  # using TextBlob for subjectivity
        analysis_results.append({
            "Message": message,
            "Sentiment": sentiment_result["label"],
            "Confidence": sentiment_result["score"],
            "Subjectivity": blob.sentiment.subjectivity
        })
    return pd.DataFrame(analysis_results)

# Function to provide suggestions for improvement
def suggest_improvements(analysis_df):
    suggestions = []
    for _, row in analysis_df.iterrows():
        if row["Sentiment"] == "NEGATIVE" or row["Subjectivity"] > 0.5:
            suggestions.append(f"Consider rephrasing: '{row['Message']}' for a more positive and objective tone.")
        elif row["Sentiment"] == "POSITIVE" and row["Confidence"] < 0.7:
            suggestions.append(f"Enhance confidence in positive tone in message: '{row['Message']}'.")
        else:
            suggestions.append("Message tone appears balanced.")
    return suggestions

# Streamlit application layout
def main():
    st.title("Conversation Tone Analyzer")
    st.write("Upload a conversation thread file (CSV with 'message' column) to analyze tone and get suggestions.")

    uploaded_file = st.file_uploader("Upload Conversation File", type=["csv", "txt"])

    if uploaded_file is not None:
        # Load conversation data
        conversation_df = pd.read_csv(uploaded_file)
        if 'message' not in conversation_df.columns:
            st.error("The CSV file must have a 'message' column.")
            return
        
        st.write("Conversation Data:")
        st.write(conversation_df)

        # Analyze tone/sentiment
        st.write("Analysis Results:")
        analysis_df = analyze_conversation(conversation_df['message'].tolist())
        st.write(analysis_df)

        # Generate suggestions
        st.write("Improvement Suggestions:")
        suggestions = suggest_improvements(analysis_df)
        for suggestion in suggestions:
            st.write(suggestion)

elif page == "About Us":
        st.title("About Us")
        st.write("""
        **Conversation Tone Analyzer** is an advanced tool designed to help users analyze and improve the tone of their conversations.
        By leveraging natural language processing and sentiment analysis, our application provides insights into the emotional 
        tone of each message in a conversation and suggests ways to foster constructive, positive communication.

        **Our Mission**: Empower users to communicate more effectively, promoting clarity and positivity in every conversation.

        **Contact Us**: For more information or feedback, please reach out at support@toneanalyzer.com.
        """)

if __name__ == "__main__":
    main()
