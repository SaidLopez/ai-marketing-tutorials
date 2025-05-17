import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class QwenChatbot:
    """
    A chatbot class using a Qwen model from Hugging Face Transformers.
    It maintains a conversation history and generates responses.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Initializes the tokenizer and model.

        Args:
            model_name (str): The name of the Qwen model to load from Hugging Face.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # device_map="auto" intelligently places the model on available devices (GPU, CPU, multi-GPU).
        # torch_dtype="auto" enables automatic mixed precision if supported, for faster inference.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Handles device placement (GPU/CPU)
            torch_dtype="auto"  # For automatic mixed precision (e.g., bfloat16 or float16)
        )
        self.history: list[dict[str, str]] = [] # Type hint for history

    def generate_response(self, user_input: str, max_new_tokens: int = 1024) -> str:
        """
        Generates a response from the model based on the user input and conversation history.

        Args:
            user_input (str): The user's current message.
            max_new_tokens (int): The maximum number of new tokens to generate.
        Returns:
            str: The model's generated response.
        """
        # Prepare messages for the current turn, including history
        messages = self.history + [{"role": "user", "content": user_input}]

        # Apply the chat template to format the input for the model
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,        # We'll tokenize next
            add_generation_prompt=True # Ensures the prompt ends correctly for generation
        )

        # Tokenize the prompt and move tensors to the model's device
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        # Generate response
        # The model.generate output includes the input_ids, so we slice them off.
        # [0] because generate can return multiple sequences if num_return_sequences > 1
        generated_ids_with_prompt = self.model.generate(
            **inputs, # Pass tokenized input_ids and attention_mask
            max_new_tokens=max_new_tokens
        )

        # Slice to get only the newly generated token_ids (excluding the prompt)
        response_ids = generated_ids_with_prompt[0][inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history with the new user input and assistant response
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response_text})

        return response_text

    def clear_history(self):
        """Clears the conversation history."""
        self.history = []

# Example Usage
if __name__ == "__main__":
    chatbot = QwenChatbot()

    user_input_1 = "How many r's in strawberries?"
    print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
    print("----------------------")


    user_input_2 = "Then, how many r's in blueberries? "
    print(f"User: {user_input_2}")
    response_2 = chatbot.generate_response(user_input_2)
    print(f"Bot: {response_2}") 
    print("----------------------")

    while True:
        user_input = input("User: ")
        if user_input.lower().strip() == "exit":
            break
        if not user_input.strip():
            print("Bot: Please tell me something!")
            continue
        
        print("Bot thinking ...")
        response = chatbot.generate_response(user_input)
        print(f"Bot: {response}")

