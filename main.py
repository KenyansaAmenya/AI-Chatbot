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
        