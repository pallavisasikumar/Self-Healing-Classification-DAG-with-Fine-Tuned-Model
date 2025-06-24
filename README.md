# 🤖 Self-Healing Sentiment Classifier

This is a command-line (CLI) project that predicts if a sentence is positive or negative using a fine-tuned AI model. If the AI isn't confident enough, it will ask the user for help or use a backup model.


What This Project Does

- It reads a sentence from the user
- It predicts if the sentence is Positive or Negative
- If the AI is not confident, it:
  - Asks the user to confirm or correct it
  - Or uses a second backup model to suggest a label
- It shows prediction confidence and logs everything
- At the end, it shows how confident the model was and how many times fallback was used


## Instructions for Running Fine-Tuning

## 🛠️ Step-by-Step Setup

### 1. Open Your Project Folder in VS Code

Make sure your folder contains:
- `fine_tune.py`
- `dag_nodes.py`
- `cli_interface.py`

### 2. Create a Virtual Environment

This keeps all your libraries separate:

python -m venv venv
venv\Scripts\activate

            "venv stands for virtual environment. It’s a way to create an isolated Python environment just for your project.Different projects need different libraries (and versions).Without venv, they’d conflict with each other."

## Install Required Libraries
pip install transformers==4.40.1 torch datasets accelerate rich sentence-transformers

# Run the training script
python fine_tune.py
   ## What This Script Does:
        Downloads the SST-2 sentiment dataset from Hugging Face (glue/sst2)

        Tokenizes the sentences and fine-tunes a DistilBERT model

        Trains for 2 epochs on a sample of 2000 examples

        Saves the trained model and tokenizer to: /trained_model/

## How to Launch and Interact with the LangGraph DAG
This project uses a LangGraph-style Directed Acyclic Graph (DAG), where each step is like a node in a logical flow.
The flow consists of:

InferenceNode – makes a sentiment prediction

ConfidenceCheckNode – checks if the model is confident enough

FallbackNode – asks the user or uses a backup model if confidence is too low

#  Run the CLI script:
 python cli_interface.py

# How interaction works:
    Once you launch the script, you'll see:
    🤖 Self-Healing Sentiment Classifier (Type 'exit' to quit)

    This triggers the LangGraph DAG flow:

    First, the model predicts the sentiment and confidence score.

    If confidence is below 70%, it triggers a fallback:

    You will be asked if the prediction was correct.

    You can correct it, or ask the backup model for help.

## 💬  CLI Flow Explanations

The project runs entirely through a command-line interface (CLI) that allows you to type a sentence, and the system walks it through a decision process using three smart steps — like a human would.

Here’s how it works behind the scenes for every input:

🔁 Step-by-Step CLI Flow
✅ 1. You type a sentence

Enter a sentence: The movie was painfully slow and boring.

2. InferenceNode runs
The AI model (DistilBERT) predicts the sentiment: Positive or Negative

It also returns a confidence score between 0 and 1

[InferenceNode] Prediction: Positive | Confidence: 54%

 ConfidenceCheckNode runs
It checks if confidence is above a threshold (default is 70%)

If confidence is high → the result is accepted

If confidence is low → fallback is triggered
[ConfidenceCheckNode] Confidence too low. Triggering fallback...

FallbackNode runs (only if needed)
Asks the user: “Do you agree with this label?”

You can say yes or no

If yes → the label is accepted

If no → you're asked to correct it

Optionally, a backup model can suggest a better label

You can type exit to quit
After you exit, the CLI shows a summary of all predictions, confidence levels, and how many times fallback was triggered.

Example:

📊 Session Summary

╭────────────────────────────╮
│   Confidence Per Input     │
├────────────┬───────────────┤
│ 1          │ 54%           │
│ 2          │ 98%           │
╰────────────┴───────────────╯

Fallbacks triggered: 1 times out of 2 inputs.
Fallback rate: 50.0%
This CLI design follows a LangGraph-style DAG — every sentence flows through a series of intelligent nodes and fallback logic.


## 🔽 Download Trained Model

Due to GitHub’s file size limit (100 MB), the trained model is not included in the repository.

📦 Download the model here:  
👉 [Download trained_model.zip](https://drive.google.com/file/d/1YN5JIot-86gnWmqfN2pwaIzxz_t3F0O3/view?usp=sharing)
