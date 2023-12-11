import logging
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, pipeline, LlamaTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from tqdm import tqdm

def load_model(device_type, model_id, model_basename=None):
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            max_ctx_size = 4096
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1000
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
            return LlamaCpp(**kwargs)
        else:
            logging.info("Using AutoGPTQForCausalLM for quantized models")

            if ".safetensors" in model_basename:
                # Remove the ".safetensors" ending if present
                model_basename = model_basename.replace(".safetensors", "")

            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            logging.info("Tokenizer loaded")

            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                model_basename=model_basename,
                use_safetensors=True,
                trust_remote_code=True,
                device="cuda:0",
                use_triton=False,
                quantize_config=None,
            )
    elif (
        device_type.lower() == "cuda"
    ):
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    generation_config = GenerationConfig.from_pretrained(model_id)

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=1,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm
