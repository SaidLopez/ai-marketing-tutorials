from transformers import AutoModelForCausalLM, AutoTokenizer  

model_name = "facebook/opt-1.3b"  # or another small model  
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForCausalLM.from_pretrained(model_name,
                                            device_map = 'auto')

print("============First Prompt=============")                                        
prompt = "Once upon a time, my friend had"  
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=25)  
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("============Second Prompt=============")   
prompt2 = "The relativity theory"
inputs = tokenizer(prompt2, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)  
print(tokenizer.decode(outputs[0], skip_special_tokens=True))