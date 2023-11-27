from tqdm import tqdm
from text_processing import clean_text
from summary_generator import generate_summary
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_summaries(df, text_splitter, llm):
    df["summary"] = ""

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating Summaries"):
        wonder_city = row["wonder_city"]
        text_chunk = row["cleaned_information"]
        chunks = text_splitter.split_text(text_chunk)
        chunk_summaries = []

        for chunk in chunks:
            summary = generate_summary(chunk, llm)
            chunk_summaries.append(summary)

        combined_summary = "\n".join(chunk_summaries)
        df.at[index, "summary"] = combined_summary

    return df
