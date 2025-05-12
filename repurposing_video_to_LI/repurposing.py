import json
from ollama import chat
from ollama import ChatResponse

def load_transcript():
    # Load the transcript from the JSON file
    with open("transcript.json", "r", encoding = 'utf-8') as json_file:
        raw = json.load(json_file)
    # Convert the JSON string to a Python dictionary
    transcript = json.loads(raw)
    
    text_only = ""
    for snippet in transcript:
        text_only += snippet['text'] + " "
    
    
    return text_only

def repurpose_content_to_LI(transcript, number_of_snippets = 2):
    
    response: ChatResponse = chat(model='qwen3:30b-a3b', messages=[
    {
        'role': 'user',
        'content': f"""You are a helpful assistant that repurposes content for social media.
        Your job is to create {number_of_snippets} posts for social media, specifically LinkedIn, from the text provided.
        The type of posts I want to create are like the following:  
        "5 things I learned from this podcast episode"
        "3 things I realised I am doing wrong"
        "I wish I had known this before"

        Remember to quote the text from the summary to make it more engaging.

        Here is the text:\n\n{transcript}
        """
    },
    ])

    return response.message.content

def get_summary_from_chunk(transcript):
    
    response: ChatResponse = chat(model='llama3.1:8B', messages=[
    {
        'role': 'system',
        'content': f"""You are a helpful assistant that sumarrises content from podcasts.
        Your job is to create an extensive summary from the text provided, ensuring as much context as possible is preserved.
        The summary is designed to facilitate the creation of social media posts, specifically LinkedIn, from the text provided.

        Don't disappoint me, I want to see the best post from the text.

        Here is the text:\n\n{transcript}
        """
    },
    {
        'role': 'user',
        'content': f"""Summarise the text provided, ensuring as much context as possible is preserved.

        Here is the text:\n\n{transcript}
        """
    },
    ])

    return response.message.content

def divide_and_conquer_and_repurpose():
    transcript = load_transcript()
    # Divide the transcript into smaller chunks
    chunk_size = 10000
    chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
    # Process each chunk separately
    summary_snippets = ""
    for i, chunk in enumerate(chunks):
        # Call the function to extract text from each chunk
        print(f"Extracting text from chunk...{i+1}/{len(chunks)}")
        summary_snippets += get_summary_from_chunk(chunk)
    
    # Repurpose the content for LinkedIn
    print("Repurposing content for LinkedIn...")
    repurposed_content = repurpose_content_to_LI(summary_snippets)
    with open("repurposed.txt", "w", encoding = 'utf-8') as f:
        f.write(repurposed_content)
        f.close()

if __name__ == "__main__":
    divide_and_conquer_and_repurpose()
