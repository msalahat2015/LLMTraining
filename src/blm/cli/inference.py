from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login

# Authenticate with Hugging Face
login("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")  # Replace with your Hugging Face token

# Define the chat template
SYSTEMPROMPT = (
    "You are an AI assistant, trained to compare sentences and detect common sense. "
    "Based on the provided instructions, you will be given two sentences, each starting with an index (0 and 1). "
    "Your task is to determine which sentence reflects common sense or if both are equally plausible. "
    "You should focus on the meaning and logical coherence of the sentences, and ensure that the comparison is fair, objective, and neutral, "
    "without any bias towards specific perspectives or individuals."
)

formatted_instructions = (
    "Given the following two sentences, each sentence starts with an index:\n"
    "0 ميغان رابينو تمتلك قوى خارقة تجعلها تلعب كرة القدم على القمر وتحقق أهدافًا من خلال جاذبيته المنخفضة.\n"
    "1 ميغان رابينو هي لاعبة كرة قدم أمريكية بارزة، قادت منتخب الولايات المتحدة للفوز بكأس العالم للسيدات 2019 وحصلت على جائزة أفضل لاعبة.\n\n"
    "Which sentence has common sense, 0 or 1?"
)

# Input data for the chat
input_data = [
    {"content": SYSTEMPROMPT, "role": "system"},
    {"content": formatted_instructions, "role": "user"},
    {"content": "", "role": "assistant"}  # Assistant's response will be generated
]

# Initialize the tokenizer and model
model_path = "/content/drive/MyDrive/LLMTraining-main/merged_output_folder"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
#model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)

# Assuming the tokenizer has an apply_chat_template method (which is not a default method in Hugging Face)
# In case you need to manually apply the template and then tokenize
tokenized_chat = tokenizer.apply_chat_template(input_data, tokenize=True, add_generation_prompt=True, return_tensors="pt")

# Create the text generation pipeline using the model from the local directory
generator = pipeline(task="text-generation", model=model_path, device=0)

# Generate the response using the pipeline with generation_token set to True
response = generator(input_data, max_length=500, num_return_sequences=1, return_full_text=True)

# Output the result
print(response)