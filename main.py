import logging
import click
import torch
import transformers
import os
import re
import shutil
import subprocess
import requests
from pathlib import Path
from huggingface_hub import hf_hub_download
from rouge import Rouge
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline, LongformerTokenizer
import pandas as pd
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from termcolor import colored
from langchain.text_splitter import RecursiveCharacterTextSplitter
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

####
from module_checker import check_and_print_module_versions
from load_model import load_model
from text_processing import fetch_and_save_wiki_text, clean_text, count_tokens, process_wonders_data
from summary_generator import generate_summary
from summary_processing import process_summaries
from result_display import display_result


def main():
    # # Configurations
    # DEVICE_TYPE = "cuda" 
    # #if torch.cuda.is_available() else "cpu"
    # SHOW_SOURCES = True
    # logging.info(f"Running on: {DEVICE_TYPE}")
    # logging.info(f"Display Source Documents set to: {SHOW_SOURCES}")

    # # Model configurations
    # model_id = "vinai/PhoGPT-7B5-Instruct"
    # model_basename = "llama-2-7b-chat.ggmlv3.q4_0.bin"
    # tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=50, length_function=len)
    # LLM = load_model(device_type=DEVICE_TYPE, model_id=model_id, model_basename=None)
    
    wonders_df = process_wonders_data()
    print(wonders_df.head())
    # processed_df = process_summaries(wonders_df, text_splitter, LLM)
    # selected_columns = processed_df[["wonder_city", "summary"]]
    # for index, row in selected_columns.iterrows():
    #     wonder_city = row["wonder_city"]
    #     summary = row["summary"]
    #     display_result(wonder_city, summary)

if __name__ == "__main__":
    main()
