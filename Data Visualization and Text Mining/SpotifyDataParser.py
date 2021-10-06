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

# playlist_id = "5FN6Ego7eLX6zHuCMovIR2"
# arg = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
# r = requests.get(arg, headers=headers)
# print(r.json())
#_____ Spotipy Interaction

import pandas as pd
import spotipy
sp = spotipy.Spotify()
from spotipy.oauth2 import SpotifyClientCredentials

cid = CLIENT_ID # check up
secret = CLIENT_SECRET  # check up
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace=False


# Get tracks in playlist
playlist = sp.user_playlist("Top 50 Global", "5FN6Ego7eLX6zHuCMovIR2")
# get response of call
songs = playlist["tracks"]["items"]

# Get track ids of songs and additional informations
ids = []
artists_ids = []
for i in range(len(songs)):
    # Get track id and append it to the list
    ids.append(songs[i]["track"]["id"]) #5PjdY0CKGZdEuoNab3yDmX

    # Get response of artists
    artist_response  = songs[i]["track"]["artists"]
    # get ids that are in track and collect them in this list
    artist_list_track =[]
    for j in range(len(artist_response)):
        artist_list_track.append(artist_response[j]["id"]) #['4gzpq5DPGxSnKTe4SA8HAU', '3Nrfpe0tUJi4K4DXYWgMUX']

    # append to artists_ids  as a list of ids for each artist [['4gzpq5DPGxSnKTe4SA8HAU', '3Nrfpe0tUJi4K4DXYWgMUX']]
    artists_ids.append(artist_list_track)

# Get Audio Analysis
features = sp.audio_features(ids)
df = pd.DataFrame(features)

# Other metrics
df["popularity"] = [ songs[i]["track"]["popularity"] for i in range(len(songs))]
df['artist_ids'] = artists_ids

# save to csv
df.to_csv("Data/CustomSpotify.csv", index=False)

dftot = pd.read_csv("Data Visualization and Text Mining/Data/Spotify_Songs_Scrapped.csv")
df = dftot.loc[dftot.country_code == 'it',:]

ids = [ url.split('track/')[1] for url in df.song_url.values.tolist() ]
main_audio_df = pd.DataFrame()
lista_blocks = [i for i in range(0, df.shape[0])[::100]]
for i in lista_blocks:
        j = i + 100
        df_block = pd.DataFrame(sp.audio_features(ids[i:j]))
        main_audio_df = pd.concat([main_audio_df, df_block], 0,ignore_index=True)
        time.sleep(0.9)
        print(i,j)