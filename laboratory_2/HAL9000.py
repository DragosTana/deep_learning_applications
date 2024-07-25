from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.generation_config.pad_token_id = tokenizer.pad_token_id

model.to(device)

def generate_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(device)
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    attention_mask = attention_mask.to(input_ids.device)    

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7, 
        pad_token_id=tokenizer.eos_token_id)
        
    output = output.to("cpu")
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("Chat with GPT-2 (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    response = generate_response(user_input)
    print("GPT-2:", response)

print("Chat ended.")