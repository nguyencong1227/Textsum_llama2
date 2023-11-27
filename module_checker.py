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
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from termcolor import colored
from langchain.text_splitter import RecursiveCharacterTextSplitter
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

GREEN = '\033[92m'
END_COLOR = '\033[0m'

def check_and_print_module_versions():
    try:
        imported_modules = [
            ("logging", logging),
            ("click", click),
            ("torch", torch),
            ("transformers", transformers),
            ("os", os),
            ("re", re),
            ("shutil", shutil),
            ("subprocess", subprocess),
            ("requests", requests),
            ("pathlib", Path),
            ("auto_gptq", AutoGPTQForCausalLM),
            ("huggingface_hub", hf_hub_download),
            ("huggingface_instruct_embeddings", HuggingFaceInstructEmbeddings),
            ("langchain_pipeline", HuggingFacePipeline),
            ("llama_cpp", LlamaCpp),
            ("prompt_template", PromptTemplate),
            ("llm_chain", LLMChain),
            ("transformers_auto_tokenizer", AutoTokenizer),
            ("transformers_auto_model", AutoModelForCausalLM),
            ("transformers_generation_config", GenerationConfig),
            ("transformers_llm_model", LlamaForCausalLM),
            ("transformers_llm_tokenizer", LlamaTokenizer),
            ("transformers_longformer_tokenizer", LongformerTokenizer),
            ("transformers_pipeline", pipeline),
            ("rouge", Rouge),
            ("text_splitter", RecursiveCharacterTextSplitter),
            ("tqdm", tqdm),
            ("termcolor_colored", colored),
        ]

        print("Module(s) Imported:")
        for module_name, module in imported_modules:
            if module:
                print(f" - {module_name}")

                version = getattr(module, "__version__", None)
                if version:
                    print(f"   Version: {GREEN}{version}{END_COLOR}")
    except ImportError as e:
        print(f"Failed to import a module: {e}")