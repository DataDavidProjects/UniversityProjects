import time
import requests
CLIENT_ID = "32b4285219474e48a926eb7892e0fd81"
CLIENT_SECRET =  "f2ea9f18c9514186a25d943a3d19739d"
AUTH_URL = 'https://accounts.spotify.com/api/token'

# POST
auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
})
# convert the response to JSON
auth_response_data = auth_response.json()
# save the access token
access_token = auth_response_data['access_token']
headers = {
    'Authorization': 'Bearer {token}'.format(token=access_token)
}

import pandas as pd
import spotipy
sp = spotipy.Spotify()
from spotipy.oauth2 import SpotifyClientCredentials

cid = CLIENT_ID # check up
secret = CLIENT_SECRET  # check up
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace=False


df = pd.read_csv("Data/regional-us-daily-latest.csv",skiprows = 1)



urls = df["URL"].values.tolist()
ids = [ i.split('track/')[1] for i in  urls ]

main_audio_df = pd.DataFrame()
lista_blocks = [i for i in range(0, df.shape[0])[::100]]
for i in lista_blocks:
        j = i + 100
        df_block = pd.DataFrame(sp.audio_features(ids[i:j]))
        main_audio_df = pd.concat([main_audio_df, df_block], 0,ignore_index=True)
        time.sleep(0.9)
        print(i,j)

total_df = pd.concat([df, main_audio_df],1)


total_df.to_csv("Data/regional-us-daily-latest-audio.csv")