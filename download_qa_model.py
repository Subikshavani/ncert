from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-small"

# Download the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Save locally
tokenizer.save_pretrained("./models/flan-t5-small")
model.save_pretrained("./models/flan-t5-small")
print("T5 QA model saved locally!")
