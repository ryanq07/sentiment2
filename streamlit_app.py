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

# Main function for Streamlit app
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Tone Analyzer", "About Us", "Methodology"])

    if page == "Tone Analyzer":
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

    elif page == "Methodology":
        st.title("Methodology")
        st.write("""
        **Methodology**:
        
        Our tool uses state-of-the-art natural language processing (NLP) techniques to analyze conversation threads.
        
        1. **Sentiment Analysis**: We utilize a pre-trained model from Hugging Face's `transformers` library to classify the sentiment 
           of each message as positive or negative. The model provides a confidence score for each classification.
        
        2. **Subjectivity Analysis**: Using TextBlob, we calculate the subjectivity of each message. High subjectivity indicates that 
           the message is more opinionated, whereas low subjectivity suggests a more objective tone.

        3. **Improvement Suggestions**: Based on the sentiment and subjectivity scores:
           - **Negative Messages**: For messages classified as negative, we suggest ways to rephrase for a more constructive tone.
           - **Highly Subjective Messages**: For messages with high subjectivity, we recommend approaches for a more balanced and 
             objective tone.
           - **Positive Messages with Low Confidence**: For positive messages with lower confidence scores, we suggest enhancing the 
             tone to ensure clarity and strength in positive communication.
        
        These combined approaches allow our tool to provide nuanced insights into conversation tone, enabling more effective communication.
        """)

if __name__ == "__main__":
    main()
