from transformers import AutoTokenizer,AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large")