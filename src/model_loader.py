# ------- Sử dụng Singleton Pattern hoặc caching của Streamlit để không load lại model gây tốn RAM ---------

import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from src.config import Config

class ModelManager:
    @staticmethod
    def load_embeddings():
        """Load model embedding tiếng Việt """
        return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_ID)

    @staticmethod
    def load_llm():
        """Load Quantized LLM (Vicuna) """
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=Config.TORCH_DTYPE
        ) 

        model = AutoModelForCausalLM.from_pretrained(
            Config.LLM_MODEL_ID,
            quantization_config=nf4_config,
            low_cpu_mem_usage=True,
            device_map="auto" 
        )
        
        tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL_ID)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            temperature=Config.TEMPERATURE,
            repetition_penalty=1.1, # Tránh lặp từ
        ) 

        return HuggingFacePipeline(pipeline=pipe)