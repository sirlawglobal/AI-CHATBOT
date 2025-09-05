from flask import Flask, render_template, request, jsonify
import random
import numpy as np
import nltk
import pickle
import tensorflow as tf
import json
from nltk.stem.lancaster import LancasterStemmer

# Initialize Flask app
app = Flask(__name__)

# Load model and training data
model = tf.keras.models.load_model("chatbot_model.keras")

with open("chatbot_data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

with open("intents.json") as file:
    intents = json.load(file)

stemmer = LancasterStemmer()

# Convert input text into bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    if not user_message:
        return jsonify({"response": "I didn't catch that. Please try again."})

    # Predict intent
    prediction = model.predict(np.array([bag_of_words(user_message, words)]), verbose=0)[0]
    results_index = np.argmax(prediction)
    tag = labels[results_index]

    # If confidence is high enough, choose a response
    if prediction[results_index] > 0.7:
        for tg in intents["intents"]:
            if tg["tag"] == tag:
                response = random.choice(tg["responses"])
                return jsonify({"response": response})

    return jsonify({"response": "I'm not sure I understood. Could you rephrase?"})

# Run the app
# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT env var
    app.run(host="0.0.0.0", port=port, debug=True)
