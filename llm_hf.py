from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
import torch

# Configuration
model_id = "microsoft/phi-2"
device = "cpu"  # or "auto" if you have some GPU

# 4-bit quantization for CPU efficiency
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32
)

# Load tokenizer and quantized model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    quantization_config=quantization_config,
    torch_dtype=torch.float32,
    trust_remote_code=True
)

# Phi-2 requires this specific prompt format
def format_prompt(message):
    return f"Instruct: {message}\nOutput:"

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    format_prompt=format_prompt  # Custom formatting for Phi-2
)

# Wrap with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Test it
prompt = "Explain quantum computing in simple terms."
print(llm.invoke(format_prompt(prompt)))