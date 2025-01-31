import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import google.generativeai as genai
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
from note_engine import notes_engine
from llama_index.llms.gemini import Gemini

api_key = load_dotenv(find_dotenv())
print(api_key)
GOOGLE_API_KEY =os.getenv('GOOGLE_API_KEY')
print(GOOGLE_API_KEY)


tools = [
    notes_engine
]

llm = Gemini(model="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY)
#response = llm.complete("Write a poem about a magic wand? in notes")
#print(response)

context="Write a poem about a magic wand in notes file"
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)
agent.query(context)
