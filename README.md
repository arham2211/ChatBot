# ChatBot

This project implements a chatbot that uses FAISS for efficient similarity search and OpenAI embeddings to understand and process natural language queries.

## Introduction

The chatbot is designed to provide intelligent responses to user queries by leveraging powerful embeddings from OpenAI and fast similarity search from FAISS. This combination allows the bot to handle large datasets and provide relevant responses in real-time.

## Features

- **Natural Language Understanding**: Utilizes OpenAI embeddings to comprehend user queries.
- **Efficient Similarity Search**: Employs FAISS to quickly find the most relevant responses from a large dataset.
- **Scalable**: Capable of handling and querying large amounts of data efficiently.
- **Easy to Use**: Simple setup and usage instructions to get the chatbot up and running.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/chatbot-faiss-openai.git
   cd chatbot-faiss-openai
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API key**:
   Obtain your API key from [OpenAI](https://openai.com/), then set it as an environment variable:
   ```bash
   export OPENAI_API_KEY='your-openai-api-key'
   ```



