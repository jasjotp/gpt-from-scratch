"""
load_data.py: This file loads a .txt file of compiled transcripts of Jensen Huang (NVIDIA CEO) on the Future of AI. The transcripts are from: 

    - The Future of AI with NVIDIA Founder and CEO Jensen Huang: https://www.youtube.com/watch?v=q54RnCUwDuY
    - Jensen Huang GTC 2025 Keynote: https://www.youtube.com/watch?v=_waPvOwL9Z8 
    - NVIDIA CEO Jensen Huang's Vision for the Future: https://www.youtube.com/watch?v=7ARBJQn6QkM
    - NVIDIA CEO Jensen Huang Keynote at COMPUTEX 2025: https://www.youtube.com/watch?v=TLzna9__DnI
    - NVIDIA CEO Jensen Huang Live GTC Paris Keynote at VivaTech 2025: https://www.youtube.com/watch?v=X9cHONwKkn4
    - AI and The Next Computing Platforms With Jensen Huang and Mark Zuckerberg: https://www.youtube.com/watch?v=w-cmMcMZoZ4&embeds_referring_euri=https%3A%2F%2Fwww.bing.com%2F&embeds_referring_origin=https%3A%2F%2Fwww.bing.com&source_ve_path=Mjg2NjY
    - NVIDIA CEO Jensen Huang Keynote at CES 2025: https://www.youtube.com/watch?v=k82RwXqZHY8&embeds_referring_euri=https%3A%2F%2Fwww.bing.com%2F&embeds_referring_origin=https%3A%2F%2Fwww.bing.com&source_ve_path=Mjg2NjY

This GPT-like model is implemented from scratch for learning purposes.

The source code and concepts are based on Andrej Karpathy's video:
"Let's build GPT: from scratch, in code, spelled out" from: https://www.youtube.com/watch?v=kCc8FmEb1nY

This project is meant to help me understand how LLMs work under the hood. The text corpus I used is a compilation of Jensen Huang's interviews from the YouTube Transcript API. 

- Large languague models take in input (a prompt) and completes the rest of the sequence 
- GPT is built from the Transformer = Landmark paper that proposed the transformer architecutre: Attention is all you Need
- GPT is short for Generatively Pretrained Transformer 
- LLMs like GPT are built on the Transformer Architecture 

"""
from youtube_transcript_api import YouTubeTranscriptApi

# initialize the YouTube transacript API 
ytt_api = YouTubeTranscriptApi()

# store video IDs for all videos you want to predict on 
video_ids = ['q54RnCUwDuY', # The Future of AI with NVIDIA Founder and CEO Jensen Huang: https://www.youtube.com/watch?v=q54RnCUwDuY
             '_waPvOwL9Z8', # Jensen Huang GTC 2025 Keynote: https://www.youtube.com/watch?v=_waPvOwL9Z8 
             '7ARBJQn6QkM', # NVIDIA CEO Jensen Huang's Vision for the Future: https://www.youtube.com/watch?v=7ARBJQn6QkM
             'TLzna9__DnI', # NVIDIA CEO Jensen Huang Keynote at COMPUTEX 2025
             'X9cHONwKkn4',  # NVIDIA CEO Jensen Huang Live GTC Paris Keynote at VivaTech 2025
             'w-cmMcMZoZ4', # AI and The Next Computing Platforms
             'k82RwXqZHY8'# NVIDIA CEO Jensen Huang Keynote at CES 2025
        ]

# fetch all transcripts from all 3 of the above videos and combine them 
combined_text = ''

for video_id in video_ids: 
    transcript = ytt_api.fetch(video_id)
    video_text = '\n'.join([snippet.text for snippet in transcript])
    combined_text += video_text + '\n\n' # add spacing between the transcripts

# save the transcripts of all the videos to one file
with open('jensen_huang.txt', 'w', encoding = 'utf-8') as f:
    f.write(combined_text)
