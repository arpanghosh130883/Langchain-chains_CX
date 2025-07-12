from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

'''
LangChain’s built-in Runnable pipeline operator |, which is syntactic sugar for chaining together components.

This syntax supports automatic graph generation, parallel execution, structured tracing, and reusability.

| works only with components that implement LangChain’s Runnable interface (PromptTemplate, ChatOpenAI, StrOutputParser, etc.).

'''

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)

chain.get_graph().print_ascii()


#### Wanted both the outputs #####


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Define Prompt Templates
prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5-pointer summary from the following text:\n{text}',
    input_variables=['text']
)

# Initialize Model and Parser
model = ChatOpenAI()
parser = StrOutputParser()

# Step 1: Generate detailed report
report_chain = prompt1 | model | parser

# Step 2: Generate summary from detailed report
summary_chain = prompt2 | model | parser

# Invoke the first chain
detailed_report = report_chain.invoke({'topic': 'Unemployment in India'})

# Invoke the second chain using detailed report as input
summary = summary_chain.invoke({'text': detailed_report})

# Print both outputs
print("===== Detailed Report =====\n")
print(detailed_report)

print("\n===== 5-Point Summary =====\n")
print(summary)

# Optional: Visualize graph of the first chain
report_chain.get_graph().print_ascii()

