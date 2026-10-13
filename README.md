# AI-Chatbot

A lightweight, intent-based chatbot that runs in the terminal. It uses a small feed-forward neural network built with PyTorch to classify a user's message, then selects a reply from the matching intent in `intents.json`.

The project includes example intents for greetings, goodbyes, thanks, and a stock-selection demonstration.

## Features

- Command-line conversation loop with `quit`, `exit`, or `bye` commands to stop.
- Text preprocessing with NLTK tokenization and WordNet lemmatization.
- Bag-of-words message representation based on the patterns in `intents.json`.
- A three-layer PyTorch classifier with ReLU activations and dropout.
- Random selection from the response list associated with a predicted intent.
- Optional intent callbacks, demonstrated by the `stocks` intent.
- Save and load support for the trained model weights and input/output dimensions.

## Requirements

- Python 3
- PyTorch
- NumPy
- NLTK

The script attempts to download the NLTK resources `punkt`, `punkt_tab`, `wordnet`, and `omw-1.4` at startup if they are not already available. This first run may need an internet connection.

## Getting started

Clone the repository and enter its directory:

```bash
git clone https://github.com/KenyansaAmenya/AI-Chatbot.git
cd AI-Chatbot
```

Install the Python dependencies:

```bash
python -m pip install nltk numpy torch
```

Run the chatbot:

```bash
python main.py
```

When the model files are present, the program loads them. If either `chatbot_model.pth` or `dimensions.json` is missing, it trains a model from `intents.json` for 100 epochs, saves the resulting files, and starts the chat loop. Training progress and loss are printed in the terminal.

Example:

```text
==================================================
 Chatbot is ready! Type 'quit' to exit.
==================================================

You: Hello
Bot: Hi there!
You: quit
Bot: Goodbye!
```

Replies are selected randomly from the matching intent's response list, so the exact response can vary between runs.

## How it works

1. **Load intent data:** `ChatbotAssistant.parse_intents()` reads the intent tags, example patterns, and responses from `intents.json`.
2. **Preprocess patterns:** Each pattern is tokenized with NLTK and lemmatized to lowercase. The unique tokens form a sorted vocabulary.
3. **Build training examples:** Each pattern becomes a binary bag-of-words vector. The target label is the index of its intent tag.
4. **Train the classifier:** A feed-forward network maps the vocabulary vector through hidden layers of 128 and 64 units to one score per intent. Training uses cross-entropy loss and the Adam optimizer.
5. **Classify a message:** Incoming text is converted to the same bag-of-words representation. The model's highest-scoring output determines the predicted intent.
6. **Return a response:** The chatbot chooses one response from that intent. If a callback is registered for the intent, it runs as well.

This is a small pattern-matching classifier, not a generative language model. It can only choose among the intent classes and responses supplied by the project.

## Project files

| File | Purpose |
| --- | --- |
| `main.py` | Contains the PyTorch model, intent parsing, preprocessing, training and inference logic, the stock callback, and the terminal interface. |
| `intents.json` | Defines intent tags, example user patterns, and possible responses. |
| `chatbot_model.pth` | Saved PyTorch state dictionary used to load the trained classifier. |
| `dimensions.json` | Stores the model's input and output layer sizes. |

The current `dimensions.json` records an input size of 28 and an output size of 4. These values correspond to the vocabulary and intent count used when the checked-in model was built.

## Customizing the chatbot

Edit `intents.json` to add or change example patterns and responses. Its top-level structure is an `intents` array. Each intent entry has:

- `tag`: A unique label for the intent.
- `patterns`: Example messages that should map to the intent.
- `responses`: Possible replies the chatbot can return.

For example:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Good morning"],
      "responses": ["Hello!", "Hi there!"]
    }
  ]
}
```

After changing patterns or intent tags, retrain the model so the vocabulary and output classes match the updated data. The current startup logic trains only when one or both model files are absent. To force a fresh training run, remove `chatbot_model.pth` and `dimensions.json` before starting the script. Keep the model files in sync with `intents.json`; otherwise, the model's output indices or dimensions may no longer represent the loaded intent data.

### Adding an intent callback

Pass a mapping from intent tags to Python functions when creating `ChatbotAssistant`. The repository demonstrates this in `main()`:

```python
assistant = ChatbotAssistant(
    "intents.json",
    function_mappings={"stocks": get_stocks}
)
```

The callback runs when its intent is predicted. The included `get_stocks()` function randomly picks three symbols from a hard-coded list and prints them to the terminal. The chatbot then returns a regular response from the `stocks` intent. It does not fetch live market data or provide financial analysis.

## Limitations

- The model uses a binary bag-of-words representation, so it does not preserve word order or conversational context.
- Intent predictions are limited to patterns and classes in `intents.json`; unfamiliar messages are still assigned to one of the known intents.
- The current training code does not hold out a validation set or report accuracy.
- The stock example uses a fixed symbol list and random selection; it is a callback demonstration, not a recommendation service.
- NLTK resource downloads happen from the script at startup when resources are missing.

## License

No license file is currently included in the repository. Unless a license is added, standard copyright applies and reuse permissions are not explicitly granted.
