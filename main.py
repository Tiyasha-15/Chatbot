import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load the intents JSON file
intents = json.loads(open(r"Chatbot/app/intents.json").read())

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

# Tokenize and process the patterns in intents
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        wordlist = nltk.word_tokenize(pattern)
        words.extend(wordlist)
        documents.append((wordlist, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize and filter out ignored letters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]

# Sort and remove duplicates
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes to pickle files
pickle.dump(words, open("Chatbot/words.pkl", "wb"))
pickle.dump(classes, open("Chatbot/classes.pkl", "wb"))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create training data in bag-of-words format
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

# Shuffle the training data
random.shuffle(training)
training = np.array(training)

# Split the data into input (X) and output (Y)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation="softmax"))

# Compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train the model
hist = model.fit(np.array(trainX), np.array(trainY), epochs=250, batch_size=5, verbose=1)

# Save the model
model.save("Chatbot/chatbot_model.h5", hist)

print("Done")
