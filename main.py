import os
import json 
import random

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.fucntional as f
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

nltk.download('punkt_tab')

class ChatbotModel(nn.Module):

    def __init__(self, input_size, output_size):
        Super(ChatbotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.linear(64, out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))    
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = []

        self.functions_mappings = function_mappings

        self.X = None
        self.y = None
    
    @staticmethod
    def tokenize_and_lemmatize(text):  
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        


Chatbot = ChatbotAssisrtant('intents.json')
chatbot.tokenize_and_lemmatize('Hello world how are you, i am programming in Python today')          