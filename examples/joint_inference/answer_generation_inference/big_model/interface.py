import logging
import os
from abc import ABC, abstractmethod
from typing import List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI

LOG = logging.getLogger(__name__)

def replace_prefix(model_path, prefix, new_prefix):
    if model_path.startswith(prefix):
        model_path = model_path[len(prefix):]
        if model_path.startswith("/"):
            model_path = model_path[1:]
        return new_prefix + model_path
    return model_path

# Format model path based on model load mode
def format_model_path(model_path):
    model_load_mode = os.environ.get("MODEL_LOAD_MODE", "file")
    prefix = os.environ.get("DATA_PATH_PREFIX", "/downloads")
    if model_load_mode == "hf":
        model_path = replace_prefix(model_path, prefix, "")
    elif model_load_mode == "http":
        model_path = replace_prefix(model_path, prefix, "http://")
    elif model_load_mode == "https":
        model_path = replace_prefix(model_path, prefix, "https://")
    return model_path

class BaseLLM(ABC):
    
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', os.environ.get('MODEL_NAME', 'deepseek-chat'))

    @abstractmethod
    def load(self, model_url: str = "") -> None:
        pass
    
    @abstractmethod
    def predict(self, data: Any, **kwargs) -> List[str]:
        pass


class HuggingFaceLLM(BaseLLM):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
    
    def load(self, model_url: str = "") -> None:
        model_path = model_url if model_url else os.environ.get("MODEL_URL", self.model_name)
        model_path = format_model_path(model_path)  
        
        LOG.info(f"Loading HuggingFace model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        LOG.info("HuggingFace model loaded successfully.")
    
    def predict(self, data: Any, **kwargs) -> List[str]:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model and tokenizer must be loaded before prediction")
        
        text = data[0] if isinstance(data, (list, tuple)) else data
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [result]


class APIBasedLLM(BaseLLM):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_url = None
        self.api_key = kwargs.get('api_key', os.environ.get('API_KEY', ''))
        self.client = None
    
    def load(self, model_url: str = "") -> None:
        model_path = model_url if model_url else os.environ.get("MODEL_URL", self.model_name)
        model_path = format_model_path(model_path)
        
        self.api_url = model_path if model_path.startswith(('http://', 'https://')) else f"http://{model_path}"
        
        base_url = self.api_url.rstrip('/')
        if not base_url.endswith('/v1'):
            base_url = f"{base_url}/v1"
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        
        LOG.info(f"API mode: Using endpoint {base_url} with model {self.model_name}")
    
    def predict(self, data: Any, **kwargs) -> List[str]:
        if self.client is None:
            raise RuntimeError("Client must be initialized before prediction")
        
        text = data[0] if isinstance(data, (list, tuple)) else data
        
        messages = [{"role": "user", "content": text}]
        
        completion_params = {
            "model": self.model_name,
            "messages": messages
        }
        
        generation_params = ['max_tokens', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty']
        for param in generation_params:
            if param in kwargs:
                completion_params[param] = kwargs[param]
        
        try:
            completion = self.client.chat.completions.create(**completion_params)
            generated_text = completion.choices[0].message.content
            return [generated_text]
                
        except Exception as e:
            LOG.error(f"API call failed: {e}")
            return [f"Error: {str(e)}"]


class Estimator:
    
    def __init__(self, **kwargs):
        self.load_mode = os.environ.get("MODEL_LOAD_MODE", "file")
        
        if self.load_mode in ("http", "https"):
            self._llm = APIBasedLLM(**kwargs)
            LOG.info("Using API-based LLM implementation")
        elif self.load_mode == "hf":
            self._llm = HuggingFaceLLM(**kwargs)
            LOG.info("Using HuggingFace LLM implementation")
        else:
            raise ValueError(f"Unsupported MODEL_LOAD_MODE: {self.load_mode}")
    
    def load(self, model_url: str = "") -> None:
        return self._llm.load(model_url)
    
    def predict(self, data: Any, **kwargs) -> List[str]:
        return self._llm.predict(data, **kwargs) 