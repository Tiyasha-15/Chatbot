import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from keras.models import load_model
import webbrowser
import datetime

# Load the chatbot data
def load_chatbot_data():
    lemmatizer = WordNetLemmatizer()
    intents = json.loads(open(r"app/intents.json").read())
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    model = load_model("chatbot_model.h5")
    return lemmatizer, intents, words, classes, model

# Clean up the sentence
def clean_up_sentence(sentence, lemmatizer):
    return [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)]

# Convert sentence to bag of words
def bag_of_words(sentence, words, lemmatizer):
    sentence_words = clean_up_sentence(sentence, lemmatizer)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

# Predict the class of the sentence
def predict_class(sentence, model, words, classes, lemmatizer):
    bow = bag_of_words(sentence, words, lemmatizer)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Get the response from the chatbot
def get_response(intents_list, intents):
    if not intents_list:
        return "I'm not sure how to respond to that. Can you please rephrase your question?"
    tag = intents_list[0]["intent"]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm sorry, I don't have a specific response for that. Can you try asking something else?"

# Function to handle the chatbot response logic
def get_bot_response(user_message, model, intents, words, classes, lemmatizer):
    if user_message.lower() in ["exit", "quit", "bye"]:
        return "Goodbye! Have a good day!"
    elif user_message.lower().startswith("search"):
        query = user_message[7:]
        webbrowser.open(f"https://www.google.com/search?q={query}")
        return f"I have opened a web search for '{query}'."
    elif user_message.lower() == "time":
        return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."
    else:
        intents_list = predict_class(user_message, model, words, classes, lemmatizer)
        return get_response(intents_list, intents)

# Streamlit GUI components
def main():
    st.title("Chat-Bot")
    st.markdown("Welcome to the Chat-Bot! Type your message below and press Enter or click Send.")

    # Initialize the chatbot
    lemmatizer, intents_data, words, classes, model = load_chatbot_data()

    # Conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    def send_message():
        user_message = st.session_state.user_input
        if user_message:
            st.session_state.conversation_history.append(f"You: {user_message}")
            bot_response = get_bot_response(user_message, model, intents_data, words, classes, lemmatizer)
            st.session_state.conversation_history.append(f"Bot: {bot_response}")
            st.session_state.user_input = ""

    # Display chat history
    for message in st.session_state.conversation_history:
        st.write(message)

    # Input box
    st.text_input("Type your message here...", key="user_input", on_change=send_message)

    # Buttons for clearing chat, saving chat, and showing help
    if st.button("Clear Chat"):
        st.session_state.conversation_history = []

    if st.button("Save Chat"):
        filename = f"Chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w") as f:
            for message in st.session_state.conversation_history:
                f.write(message + "\n")
        st.success(f"Chat history has been saved to {filename}")

    if st.button("Help"):
        st.info("""
        Welcome to the Chat-Bot!

        Special Commands:
        - Type "exit", "quit", or "bye" to end the conversation.
        - Type "search <query>" to open a web search.
        - Type "time" to get the current time.

        Features:
        - Clear Chat: Clears the current conversation.
        - Save Chat: Saves the conversation history to a file.
        - Help: Shows this help message.

        Enjoy Chatting!
        """)

if __name__ == "__main__":
    main()
