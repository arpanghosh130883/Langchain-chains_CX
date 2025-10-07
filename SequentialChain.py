'''
SequentialChain — More Flexible, Supports Multiple Variables
✅ When to use:

You want to pass multiple named variables between chains (more structured workflow).
'''

from langchain.chains import LLMChain, SequentialChain

from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)

# Step 1 → Generate a company name
prompt1 = PromptTemplate(
    input_variables=["product"],
    template="Suggest a creative company name for a product that does {product}."
)
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="company_name")

# Step 2 → Write a tagline using company name
prompt2 = PromptTemplate(
    input_variables=["company_name"],
    template="Write a short and catchy tagline for the company {company_name}."
)
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="tagline")

# Step 3 → Write a short marketing paragraph
prompt3 = PromptTemplate(
    input_variables=["company_name", "tagline"],
    template="Write a short paragraph describing {company_name} with the tagline: {tagline}."
)
chain3 = LLMChain(llm=llm, prompt=prompt3, output_key="description")

# Combine all
overall_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["product"],
    output_variables=["company_name", "tagline", "description"],
    verbose=True
)

# Run
result = overall_chain({"product": "AI-driven nutrition tracking"})
print(result)

'''
What happens:

Chain 1 → generates company_name.

Chain 2 → uses that name to create tagline.

Chain 3 → uses both company_name and tagline to write a description.

✅ Output example:

{
  "company_name": "NutriAI",
  "tagline": "Smart eating made simple.",
  "description": "NutriAI is revolutionizing how you track your diet with intelligent, personalized nutrition insights. Smart eating made simple."
}
'''
