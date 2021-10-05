import pandas as pd
import spotipy
sp = spotipy.Spotify()
from spotipy.oauth2 import SpotifyClientCredentials

cid ="32b4285219474e48a926eb7892e0fd81"
secret = "f2ea9f18c9514186a25d943a3d19739d"
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace=False


playlist = sp.user_playlist("Top 50 Global", "5FN6Ego7eLX6zHuCMovIR2")

popularity = []

songs = playlist["tracks"]["items"]
ids = []
for i in range(len(songs)):
    ids.append(songs[i]["track"]["id"])

features = sp.audio_features(ids)
df = pd.DataFrame(features)


