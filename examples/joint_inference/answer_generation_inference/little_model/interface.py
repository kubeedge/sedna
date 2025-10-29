import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

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

class Estimator:
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name', 'distilgpt2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.model_load_mode = os.environ.get("MODEL_LOAD_MODE", "file")

    def load(self, model_path=""):
        if not model_path:
            model_path = os.environ.get("MODEL_URL")

        model_path = format_model_path(model_path)

        LOG.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        LOG.info("Model loaded successfully.")

    def predict(self, data, **kwargs):
        text = data[0] if isinstance(data, (list, tuple)) else data
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        max_new_tokens = int(kwargs.get("max_new_tokens", 50))
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens
            )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result 
