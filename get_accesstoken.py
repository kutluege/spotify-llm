import requests
import os
from dotenv import load_dotenv


#client_id = os.getenv("CLIENT_ID")  # Replace with your Client ID in .env
#client_secret = os.getenv("CLIENT_SECRET")  # Replace with your Client Secret in .env
#redirect_uri = os.getenv("REDIRECT_URI")  # Replace with your Redirect URI in .env

client_id = "f29ee281b7664d65ad23ecc58965e61d"
client_secret ="55f57bd1a7c14c3ba3e05657f9f19294"
redirect_uri = "https://spotify-llm.streamlit.app/"

code = "AQBs9VDIIx1YjTPAuDghLnwWKuiBLukTclPbCfqJVK3H-VwcZ1jic3_Gg564mAbwEH7oITHantqFj8v5QCKCA5MX3obOrG10PlYg8PwoKJec2BGLd3PQbFwmhuUzUGZzea-GM1y8PWvAu6xVJ9pENo6K7Pl-eO5UdXL4yML1uu5PvBVP3NU_PqxkPDhgqGED7XfB4aBPlnSWj-hD7w"

def get_access_token(code):
    url = "https://accounts.spotify.com/api/token"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "client_secret": client_secret,
    }

    response = requests.post(url, data=data)

    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data["access_token"]

        # Access token'ı access_token.txt dosyasına kaydediyoruz
        with open("access_token.txt", "w") as f:
            f.write(access_token)
        print("Access token başarıyla kaydedildi.")
    else:
        print("Access token alınamadı:", response.json())

get_access_token(code)
