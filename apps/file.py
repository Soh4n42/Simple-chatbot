import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

# Get API key from .env
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # type: ignore

# Initialize FastAPI app
app = FastAPI()

# Define the input model
class Message(BaseModel):
    msg: str

# Set up the LLM
llm_model = ChatGroq(model='Gemma2-9b-IT', temperature=0.6)

# Define the prompt
prompts = ChatPromptTemplate.from_messages([
    ("system", "you are an AI assistant expert in generative AI"),
    ("user", "{input}")
])

# Define the route
@app.post("/chat")
async def input_function(msg: Message):
    formatted_prompt = prompts.format_messages(input=msg.msg)
    response = llm_model.invoke(formatted_prompt)
    parser = StrOutputParser()
    final_response = parser.invoke(response)
    return {"response": final_response}
