import replicate
import re
import os
from dotenv import load_dotenv


load_dotenv()

def create_prompt(new_act, examples):
    prompt = "Aşağıdaki Spotify API eylemlerini, verilen örnekleri kullanarak API çağrılarına dönüştürün.\n\n"
    for example in examples:
        prompt += f"Action: **{example['action']}**\nInstruction:\n{example['instruction']}\nAPI Call:\n```bash\n{example['api_call']}\n```\n\n"
    prompt += f"**Now, convert this action:**\n\nAction: **{new_act}**\nAPI Call:\n"
    return prompt

def generate_api_call(playlist_id, access_token):
    examples = [
        {
            "action": f"ID'si {playlist_id} olan Playlist'i getir.",
            "instruction": "Kullanıcı belirli bir playlisti getirmek istiyor. Aşağıdaki API çağrısını kullanarak bu aksiyonu gerçekleştir.",
            "api_call": f"curl --request GET --url https://api.spotify.com/v1/playlists/{playlist_id} --header 'Authorization: Bearer {access_token}'"
        }
    ]
    
    prompt = create_prompt(f"ID'si {playlist_id} olan bir playlisti getir.", examples)

    system_prompt = "Spotify API eylemlerine karşılık gelen API çağrılarını uygun şekilde oluşturan bir AI asistanısınız. Her eylem için tutarlı API çağrıları kullan. API çağrısının sağlanan eylemi doğru şekilde yansıttığından emin ol."

    AI_Response = replicate.run(
        "meta/llama-2-70b-chat",
        input={
            "top_k": 0,
            "top_p": 1,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.5,
            "system_prompt": system_prompt,
            "length_penalty": 1,
            "max_new_tokens": 500,
            "min_new_tokens": -1,
            "prompt_template": "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]",
            "presence_penalty": 0,
            "log_performance_metrics": False
        },
    )

    full_response = "".join(AI_Response)
    
    # "curl" komutundan URL ve headers'ı ayıklıyorum
    url = re.search(r"--url (https?://[^\s]+)", full_response)
    auth_header = re.search(r"--header 'Authorization: Bearer ([^']+)'", full_response)
    
    url = url.group(1) if url else None
    auth_token = auth_header.group(1) if auth_header else None
    
    return url, {"Authorization": f"Bearer {auth_token}"}
