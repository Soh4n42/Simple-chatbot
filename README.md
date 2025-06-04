
# 🧠 FastAPI Chatbot using LangChain & Groq

This is a simple FastAPI application that integrates LangChain with the Groq API to provide a Generative AI chatbot. It uses a `.env` file to securely manage your Groq API key and serves a single `/chat` endpoint to interact with the model.

---

## 🚀 Features

- 🔒 Secure API key loading via `.env`
- ⚡ Powered by [LangChain](https://python.langchain.com) and [Groq](https://groq.com/)
- 🧠 Uses `Gemma2-9b-IT` as the large language model
- 📡 Lightweight FastAPI server with a single `/chat` POST endpoint
- 🔁 Structured response using LangChain's output parser

---

## 📁 Project Structure
├── file.py # FastAPI application
├── .env # Contains your GROQ_API_KEY (not tracked by Git)
├── .gitignore # Ensures .env, pycache, etc. are ignored
└── README.md # This file

## 🛠️ Installation

1. **Clone the repository:**
2. **Setup Virtual Environment.**
3. **Install Dependencies.**
4. **Setup Environment Variables**
5. **Run the Server.**

**Accessing the bot goto /docs in your local host and to /chat and try to play with it if you want a simple frontend you can add**


