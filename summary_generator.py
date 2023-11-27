from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_summary(text_chunk, llm):
    # Defining the template to generate summary
    template = """
    Write a concise summary of the text, return your responses with 5 lines that cover the key points of the text.
    ```{text}```
    SUMMARY:
    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    summary = llm_chain.run(text_chunk)
    return summary