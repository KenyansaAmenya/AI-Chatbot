import os
import json
import random

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


for pkg in ('punkt', 'punkt_tab', 'wordnet', 'omw-1.4'):
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg.startswith('punkt') else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, quiet=True)

class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}

        self.functions_mappings = function_mappings or {}

        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        return [lemmatizer.lemmatize(w.lower()) for w in words]

    def bag_of_words(self, words):
        return [1 if w in words else 0 for w in self.vocabulary]

    def parse_intents(self):
        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(
                f"Could not find {self.intents_path}. "
                "Make sure it is in the same folder as this script."
            )

        with open(self.intents_path, 'r') as f:
            intents_data = json.load(f)

        for intent in intents_data['intents']:
            if intent['tag'] not in self.intents:
                self.intents.append(intent['tag'])
                self.intents_responses[intent['tag']] = intent['responses']

            for pattern in intent['patterns']:
                pattern_words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, intent['tag']))

        self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags, indices = [], []
        for words, tag in self.documents:
            bags.append(self.bag_of_words(words))
            indices.append(self.intents.index(tag))

        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=batch_size,
            shuffle=True
        )

        self.model = ChatbotModel(self.X.shape[1], len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump(
                {'input_size': self.X.shape[1], 'output_size': len(self.intents)},
                f
            )

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dims = json.load(f)

        self.model = ChatbotModel(dims['input_size'], dims['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)
            predicted_idx = torch.argmax(predictions, dim=1).item()
            predicted_intent = self.intents[predicted_idx]

        if predicted_intent in self.functions_mappings:
            self.functions_mappings[predicted_intent]()

        responses = self.intents_responses.get(predicted_intent)
        return random.choice(responses) if responses else None


def get_stocks():
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    picks = random.sample(stocks, 3)
    print(f"   [function] Picked stocks: {picks}")
    return picks

def main():
    intents_path = 'intents.json'
    model_path = 'chatbot_model.pth'
    dims_path = 'dimensions.json'

    assistant = ChatbotAssistant(
        intents_path,
        function_mappings={'stocks': get_stocks}
    )

    if os.path.exists(model_path) and os.path.exists(dims_path):
        print("Found trained model. Loading...")
        assistant.parse_intents()
        assistant.load_model(model_path, dims_path)
    else:
        print("No trained model found. Training from scratch...")
        assistant.parse_intents()
        assistant.prepare_data()
        assistant.train_model(batch_size=8, lr=0.001, epochs=100)
        assistant.save_model(model_path, dims_path)
        print("Training complete. Model saved.")

   
    print("\n" + "=" * 50)
    print(" Chatbot is ready! Type 'quit' to exit.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        if user_input.lower() in ('quit', 'exit', 'bye'):
            print("Bot: Goodbye!")
            break

        if not user_input:
            continue

        response = assistant.process_message(user_input)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()