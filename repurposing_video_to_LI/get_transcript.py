from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter
import json




def get_transcript_from_yt():
    # For reference only, this is a podcast
    full_url = "https://www.youtube.com/watch?v=D67eWcX2XYQ&ab_channel=TheDiaryOfACEO"

    # Now we fetch the video ID
    video_id = "D67eWcX2XYQ"  
    ytt_api = YouTubeTranscriptApi()
    # Fetch the transcript
    fetched_transcript = ytt_api.fetch(video_id)
    formatter = JSONFormatter()

    #Get a snapshot of the transcript
    for snippet in fetched_transcript[0:5]:
        print(snippet.text) 
    
    json_formatted = formatter.format_transcript(fetched_transcript)
    # Save the transcript to a JSON file
    with open("transcript.json", "w", encoding = 'utf-8') as json_file:
        json.dump(json_formatted, json_file, ensure_ascii=False, indent=4)
    print("Transcript saved to transcript.json")




if __name__ == "__main__":
    load_transcript()
