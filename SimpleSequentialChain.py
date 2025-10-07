'''
When to use:

You want to feed one LLM’s output directly into another LLM (no variable mapping needed).
It assumes:

Each chain takes a single input string

Each chain outputs a string
'''

from langchain.cahins import LLMChain, SimpleSequentialChain
from langchain_core.prompts import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize model
llm = OpenAI(temperature=0.7)

# Chain 1 → Generate an idea
prompt1 = PromptTemplate.from_template("Give me an interesting startup idea about {topic}.")
chain1 = LLMChain(llm=llm, prompt=prompt1)

# Chain 2 → Write a tweet about that idea
prompt2 = PromptTemplate.from_template("Write a catchy tweet promoting this startup idea:\n{idea}")
chain2 = LLMChain(llm=llm, prompt=prompt2)

# Combine them
overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

# Run it
result = overall_chain.run("artificial intelligence")
print(result)


'''
What happens:

First LLM generates a startup idea based on "artificial intelligence".

Its output becomes {idea} for the second chain.

The final output is a tweet.

✅ Output example:

🚀 New AI startup: "MindSync" – AI-powered brainwave translator helping paralyzed
'''
