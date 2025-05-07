from ollama import chat
from ollama import ChatResponse
import json

SYSTEM_PROMPT_1 = """
You are a helpful assistant that helps with the continuous improvement of product-related messages.
You will be given a series of messages, and your task is to create a report highlighting how to improve the product.
"""

SYSTEM_PROMPT_2 = """
You are a helpful assistant that helps with the continuous improvement of product-related messages.
You will be given a series of messages and your task is to highlight what the product is doing well.
We want to use this information to improve the product and improve our marketing efforts.
"""

SYSTEM_PROMPT_3 = """
You are a helpful assistant that helps with the continuous improvement of product-related messages.
You will be given a series of messages with 'comments', 'pros' and 'cons'.
Create a report highlighting how to improve the product and what the product is doing well.
We want to use this information to improve the product and improve our marketing efforts.
"""

def analyse_negative_responses(messages):
    # Example of a chat with a model
    response: ChatResponse = chat(model='gemma3:latest', messages=[
        {
            'role': 'system',
            'content': SYSTEM_PROMPT_1,
        },
        {
            'role': 'user',
            'content': f'Analyse the following messages and provide a report on how to improve the product: \n\n{messages}\n\n',
        },
    ])
    

    return response.message.content

def analyse_positive_responses(messages):
    # Example of a chat with a model
    response: ChatResponse = chat(model='gemma3:latest', messages=[
        {
            'role': 'system',
            'content': SYSTEM_PROMPT_2,
        },
        {
            'role': 'user',
            'content': f'Analyse the following messages and provide a report on how to improve the product and our marketing: \n\n{messages}\n\n',
        },
        
    ])
    
    return response.message.content

def analyse_all(messages):
    # Example of a chat with a model
    response: ChatResponse = chat(model='gemma3:latest', messages=[
        {
            'role': 'system',
            'content': SYSTEM_PROMPT_3,
        },
        {
            'role': 'user',
            'content': f'Analyse the following messages and provide a report on how to improve the product and our marketing: \n\n{messages}\n\n',
        },
        
    ])
    
    return response.message.content


if __name__ == "__main__":
    # Load the data
    with open("output.json", "r") as f:
        data = json.load(f)
        f.close()

    # Extract the messages
    cons_messages = []
    pros_messages = []
    for item in data:
        cons_messages.append(item['cons'])
        pros_messages.append(item['pros'])

    # Analyse the messages
    negative_report = analyse_negative_responses(cons_messages)
    positive_report = analyse_positive_responses(pros_messages)
    analyse_all_report = analyse_all(data)

    # Print the reports
    print(negative_report)
    print(positive_report)
    print(analyse_all_report)
