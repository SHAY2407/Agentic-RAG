from llama_index.llms.google_genai import GoogleGenAI
import google.generativeai as genai
from llama_index.experimental.query_engine.pandas import PandasQueryEngine
import os
from dotenv import load_dotenv
import pandas as pd
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from prompts import new_prompt, instruction_str, context
from pdf import circular_engine

# Load .env and get API key
load_dotenv()
api_key = os.getenv("API_KEY")  # this is your Gemini key
genai.configure(api_key=api_key)
# Use Gemini as LLM
llm = GoogleGenAI(model="gemini-2.0-flash", api_key=api_key)

# Read your CSV
population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

# Use the Gemini LLM explicitly
population_query_engine = PandasQueryEngine(df=population_df, llm=llm, verbose=True)

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine, 
        metadata=ToolMetadata(
            name="population_data",
            description="this gives information at the world population and demographics",
        ),
    ),
    QueryEngineTool(
        query_engine=circular_engine, 
        metadata=ToolMetadata(
            name="circular_data",
            description="this gives detailed information about the circular",
        ),
    ),
]

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

while (prompt:=input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)