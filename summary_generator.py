from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def generate_summary(text_chunk, llm):
    # Defining the template to generate summary
    template = """
    Viết một bản tóm tắt ngắn gọn của văn bản, trả lời câu trả lời của bạn bằng 5 dòng bao gồm các điểm chính của văn bản.
    ```{text}```
    Tóm tắt:
    """
    prompt = PromptTemplate(template=template, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    summary = llm_chain.run(text_chunk)
    return summary