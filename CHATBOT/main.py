import nltk 
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import json
import pickle

# Download required NLTK data (only needs to run once)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load the intents file
with open("intents.json") as file:
    data = json.load(file)

print(data)

# Initialize lists to store our data
words = []
labels = []
docs_x = []
docs_y = []

# Process each intent in our data
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize each pattern into words
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
    
    # Add the tag to our labels if it's not already there
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Stem all words and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

print(f"Number of unique words: {len(words)}")
print(f"Number of labels: {len(labels)}")

# Create training data
training = []
output = []

# Create empty output template
out_empty = [0 for _ in range(len(labels))]

# Create bag of words for each pattern
for x, doc in enumerate(docs_x):
    bag = []
    
    # Stem the words from the current pattern
    wrds = [stemmer.stem(w.lower()) for w in doc]
    
    # Create bag of words array
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    
    # Create output array (1 for current tag, 0 for others)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    
    training.append(bag)
    output.append(output_row)

# Convert to numpy arrays
training = np.array(training)
output = np.array(output)

print(f"Training data shape: {training.shape}")
print(f"Output data shape: {output.shape}")

# Build the neural network using Keras (replaces tflearn)
model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(len(training[0]),), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(output[0]), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training the model...")
# Train the model
model.fit(training, output, epochs=1000, batch_size=8, verbose=1)

# Save the model and training data
model.save("chatbot_model.keras")
with open("chatbot_data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

print("Model trained and saved successfully!")

# Function to create bag of words for user input
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return np.array(bag)

# Function to chat with the bot
def chat():
    print("Start talking with the bot! (type 'quit' to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        # Get prediction from the model
        prediction = model.predict(np.array([bag_of_words(inp, words)]), verbose=0)[0]
        results_index = np.argmax(prediction)
        tag = labels[results_index]
        
        # Only respond if confidence is high enough
        if prediction[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    print(f"Bot: {random.choice(responses)}")
                    break
        else:
            print("Bot: I didn't understand that. Could you please rephrase?")

# Start the chat
if __name__ == "__main__":
    chat()