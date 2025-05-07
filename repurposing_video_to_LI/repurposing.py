import json
from ollama import chat
from ollama import ChatResponse

def extract_text_from_transcript(transcript, number_of_snippets = 5):
    
    response: ChatResponse = chat(model='llama3.1:8B', messages=[
    {
        'role': 'system',
        'content': f"""You are a helpful assistant that extracts text from a transcript for content creation.
        Your job is to extract {number_of_snippets} snippets from the transcript and provide them in a list format.
        Your job is to summarise these snippets in a way that is easy to understand and engaging for the reader.
        """
    },
    {
        'role': 'user',
        'content': f"""Extract {number_of_snippets} snippets from the following transcript:\n\n{transcript}
        """
    },
    ])

    return response.message.content

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

def get_content_repurposed():

    transcript = load_transcript()

    snippets = extract_text_from_transcript(transcript)
    print(snippets)

if __name__ == "__main__":
    get_content_repurposed()