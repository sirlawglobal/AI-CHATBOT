# AI Chatbot Live Coding Roadmap
## Step-by-Step Guide for Teaching 100 Students



---

## 🎬 Phase 1: Setup & Environment (30 minutes)

### 1.1 Welcome & Introduction (10 minutes)
```python
# Show final working chatbot first!
python main.py
# Demo conversation to build excitement
```

**Talk Track:**
- "Today we're building this conversational AI from scratch"
- "By the end, you'll understand how ChatGPT-style bots work"
- "This same technique powers customer service bots, virtual assistants"

### 1.2 Environment Setup (15 minutes)
```bash
# Everyone runs together:
mkdir my-chatbot
cd my-chatbot
pip install nltk tensorflow numpy

# Test installation:
python -c "import nltk, tensorflow; print('All good!')"
```

**Teaching Tips:**
- Have TAs help with installation issues
- Provide pre-setup environments if possible
- Share screen showing each command

### 1.3 Project Structure Overview (5 minutes)
```bash
touch main.py
touch intents.json
```

**Explain the plan:**
- "We'll build 2 files that create magic together"
- "intents.json = the bot's knowledge"
- "main.py = the bot's brain"

---

## 🧠 Phase 2: Building the Brain (2.5 hours)

### 2.1 Create the Knowledge Base (20 minutes)
**File: `intents.json`**

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey"],
      "responses": ["Hello!", "Hi there!"]
    }
  ]
}
```

**Live Coding Approach:**
1. Start with just greeting intent
2. Explain each part as you type
3. Have students add their own patterns
4. **Interactive moment**: "Everyone add a greeting from your language!"

**Add more intents together:**
```json
{
  "tag": "goodbye", 
  "patterns": ["Bye", "See you", "Goodbye"],
  "responses": ["Goodbye!", "See you later!"]
},
{
  "tag": "help",
  "patterns": ["Help", "I need help", "Can you help me"],
  "responses": ["How can I help you?", "I'm here to help!"]
}
```

### 2.2 Import Libraries & Setup (15 minutes)
**File: `main.py`**

```python
# Code along - explain each import
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import pickle
import random

# Download NLTK data - explain why we need it
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
```

**Teaching moment:**
- "NLTK = Natural Language Toolkit"
- "Stemmer reduces words to roots: running → run"
- "TensorFlow = Google's AI library"

### 2.3 Load and Process Data (25 minutes)

```python
# Load our knowledge base
with open("intents.json") as file:
    data = json.load(file)

print(data)  # Show what we loaded

# Initialize storage
words = []
labels = []
docs_x = []  # All patterns
docs_y = []  # Corresponding labels

stemmer = LancasterStemmer()
```

**Process the intents:**
```python
# Code together - explain each step
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # Break sentence into words
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
    
    # Collect all unique tags
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# Clean up words - stem and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

print(f"Words: {len(words)}, Labels: {len(labels)}")
```

**Interactive Check:**
- "Pause - everyone run this and see your word count"
- "Add more patterns to intents.json and watch numbers change!"

### 2.4 Create Training Data (30 minutes)

**Concept Explanation (10 minutes):**
```python
# Show the concept with simple example
sample_pattern = ["Hello", "there"]
sample_words = ["hello", "hi", "there", "goodbye"]

# Bag of words = [1, 0, 1, 0] 
# 1 means word exists, 0 means it doesn't
```

**Build actual training data (20 minutes):**
```python
training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    
    # Stem words from current pattern
    wrds = [stemmer.stem(w.lower()) for w in doc]
    
    # Create bag of words
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    
    # Create output (one-hot encoding)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    
    training.append(bag)
    output.append(output_row)

# Convert to numpy arrays
training = np.array(training)
output = np.array(output)

print(f"Training shape: {training.shape}")
print(f"Output shape: {output.shape}")
```

**Teaching Break:**
- "Let's understand what we just created"
- Show example: `print(training[0])` and explain
- "This is how computers understand text - as numbers!"

### 2.5 Build Neural Network (20 minutes)

**Start simple - explain the architecture:**
```python
# Build the brain of our chatbot
model = keras.Sequential([
    # Input layer - size of our vocabulary
    keras.layers.Dense(128, input_shape=(len(training[0]),), activation='relu'),
    
    # Dropout prevents overfitting (like studying too hard for one test)
    keras.layers.Dropout(0.5),
    
    # Hidden layer
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    
    # Output layer - number of possible intents
    keras.layers.Dense(len(output[0]), activation='softmax')
])

# Configure how the model learns
model.compile(
    optimizer='adam',  # How to adjust weights
    loss='categorical_crossentropy',  # How to measure mistakes
    metrics=['accuracy']  # What to track
)

print("Model architecture created!")
model.summary()  # Show the structure
```

**Teaching moment:**
- Draw the network on whiteboard/screen
- "128, 64 neurons - these are like brain cells"
- "ReLU = if negative, make it 0"
- "Softmax = convert to probabilities"

### 2.6 Train the Model (15 minutes)

```python
print("Training the AI brain...")
print("This will take a few minutes - like teaching a child language!")

# Train the model
history = model.fit(
    training, 
    output, 
    epochs=1000,  # How many times to see all data
    batch_size=8,  # How many examples at once
    verbose=1
)

# Save everything
model.save("chatbot_model.keras")
with open("chatbot_data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

print("🎉 Model trained and saved!")
```

**While training runs:**
- Explain epochs, batches, loss
- Show the accuracy improving
- "Like a student getting better at tests!"

### 2.7 Create Chat Functions (25 minutes)

**Bag of words function:**
```python
def bag_of_words(s, words):
    """Convert user input to same format as training data"""
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return np.array(bag)

# Test it!
test_input = "Hello there"
test_bag = bag_of_words(test_input, words)
print(f"'{test_input}' becomes: {test_bag[:10]}...")  # Show first 10
```

**Main chat function:**
```python
def chat():
    """Main chatbot conversation loop"""
    print("🤖 Chatbot ready! Type 'quit' to exit")
    
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
            
        # Get prediction
        prediction = model.predict(
            np.array([bag_of_words(inp, words)]), 
            verbose=0
        )[0]
        
        results_index = np.argmax(prediction)
        tag = labels[results_index]
        confidence = prediction[results_index]
        
        print(f"Debug - Intent: {tag}, Confidence: {confidence:.2f}")
        
        # Only respond if confident enough
        if confidence > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    print(f"🤖: {random.choice(responses)}")
                    break
        else:
            print("🤖: I'm not sure I understand. Can you rephrase?")

# Start chatting!
if __name__ == "__main__":
    chat()
```

---

## 🧪 Phase 3: Testing & Customization (45 minutes)

### 3.1 First Test Run (15 minutes)
```bash
python main.py
```

**Interactive Testing:**
- Everyone test basic greetings
- Try edge cases: "HELLO!", "hiii", "good morning"
- Show confidence scores
- Celebrate when it works!

### 3.2 Add Personal Touch (20 minutes)
**Challenge everyone:**
1. Add an intent about yourself
2. Add a hobby/interest intent  
3. Add local language greetings

**Example additions to intents.json:**
```json
{
  "tag": "about_me",
  "patterns": ["Tell me about yourself", "Who are you", "What do you do"],
  "responses": ["I'm an AI assistant created in this workshop!", "I'm learning to help humans!"]
},
{
  "tag": "weather",
  "patterns": ["How's the weather", "Is it sunny", "Weather today"],
  "responses": ["I can't check weather yet, but it's always sunny in AI land! ☀️"]
}
```

### 3.3 Advanced Experiments (10 minutes)
**For faster learners:**
```python
# Add confidence tuning
# Show prediction probabilities for all intents
def detailed_predict(user_input):
    prediction = model.predict(np.array([bag_of_words(user_input, words)]), verbose=0)[0]
    
    for i, label in enumerate(labels):
        print(f"{label}: {prediction[i]:.3f}")
    
    return labels[np.argmax(prediction)]

# Test with: detailed_predict("Hello there")
```

---

## ❓ Phase 4: Q&A & Next Steps (15 minutes)

### 4.1 Common Questions & Answers
**Q: "How do I make it smarter?"**
A: More training data, better intents, larger neural network

**Q: "Can it remember previous messages?"**
A: Not yet! That's called context - advanced topic

**Q: "How is this different from ChatGPT?"**
A: Same concept, but ChatGPT uses transformers and MUCH more data

### 4.2 Next Steps Roadmap
1. **Week 1**: Add 10 more intents, experiment with responses
2. **Week 2**: Deploy to web using Flask
3. **Month 1**: Add context/memory
4. **Month 2**: Connect to APIs (weather, news)
5. **Month 3**: Try transformer models (BERT, GPT)

### 4.3 Resources for Continued Learning
- Practice repository: `github.com/[your-username]/chatbot-workshop`
- Discord/Slack for questions
- Next workshop: "Deploying AI to the Web"

---

## 🎯 Teaching Success Metrics
- [ ] Everyone has a working chatbot
- [ ] Students can add new intents
- [ ] Students understand the basic ML pipeline
- [ ] At least 80% can explain bag-of-words concept
- [ ] Students are excited to continue learning

---

## 🚨 Troubleshooting Guide

### Quick Fixes:
```bash
# If NLTK fails
python -c "import nltk; nltk.download('all')"

# If TensorFlow issues
pip uninstall tensorflow
pip install tensorflow==2.15.0

# If JSON errors
# Use online JSON validator: jsonlint.com

# If model doesn't train
# Reduce epochs to 100 for faster testing
```

### Emergency Backup Plan:
- Have pre-trained model files ready
- Provide working code repository
- Use Colab notebooks as backup environment

---

## 📝 Post-Workshop Follow-up

### Send to Students:
1. Complete code repository
2. Tutorial markdown (the first file we created)
3. List of additional challenges
4. Community Discord/Slack invite
5. Survey for feedback

### Instructor Notes:
- Collect common questions for FAQ
- Note which concepts need more explanation
- Identify fast/slow learners for future workshops
- Update materials based on feedback

---

*This roadmap provides a structured approach to teaching AI development that keeps students engaged while building practical skills. Adjust timing based on your audience's experience level.*