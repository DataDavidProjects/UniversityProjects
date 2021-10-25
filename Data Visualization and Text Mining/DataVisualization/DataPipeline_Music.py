
from Keys import *
# SPOTIFY
""" 
This pipeline use Spotify API , Spotipy and Genius API
Create a dataset about music with different metrics information and also the lyrics.
"""


#####################################
import time
import requests

CLIENT_ID = spotify_client_id
CLIENT_SECRET =  spotify_client_secret
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
import numpy as np
import spotipy
sp = spotipy.Spotify()
from spotipy.oauth2 import SpotifyClientCredentials

cid = CLIENT_ID # check up
secret = CLIENT_SECRET  # check up
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace=False


import sys
from bs4 import BeautifulSoup
from datetime import date, timedelta


def top200_chart(region):

    url = f"https://spotifycharts.com/regional/{region}/daily/latest"
    print(region , url)
    debug = {'verbose': sys.stderr}
    user_agent = {'User-agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=user_agent)
    time.sleep(1)
    soup = BeautifulSoup(response.text, "html.parser")
    songs = soup.find("table", {"class": "chart-table"})
    final = []
    for tr in songs.find("tbody").findAll("tr"):
            artist = tr.find("td", {"class": "chart-table-track"}).find("span").text
            artist = artist.replace("by ", "").strip()

            streams = tr.find("td", {"class": "chart-table-streams"}).text.replace(',',"")

            title = tr.find("td", {"class": "chart-table-track"}).find("strong").text

            songid = tr.find("td", {"class": "chart-table-image"}).find("a").get("href")
            songid = songid.split("track/")[1]

            url_date = url.split("daily/")[1]

            final.append([title, artist, songid, streams])

    data = pd.DataFrame(final, columns = ["title", "artist", "songid" , "streams"])
    data["region"] = [region for i in range(len(data))]

    # get audio analysis
    ids = data["songid"].values.tolist()
    main_audio_df = pd.DataFrame()
    for id in ids:
        analysis = sp.audio_features(id)
        if analysis == [None]:
            audio_columns = ['acousticness', 'analysis_url', 'danceability',
                             'duration_ms', 'energy', 'id',
                             'instrumentalness', 'key', 'liveness',
                             'loudness', 'mode', 'speechiness',
                             'tempo', 'time_signature', 'track_href',
                             'type', 'uri', 'valence']
            print(f"No audio analysis for region: {region} at id : {id}")
            df_block = pd.DataFrame(np.zeros((1, 18)), columns=audio_columns)
        else:
            df_block = pd.DataFrame(sp.audio_features(id))
        main_audio_df = pd.concat([main_audio_df, df_block], axis="rows", ignore_index=True)
        time.sleep(0.0001)

    total_df = pd.concat([data, main_audio_df], axis="columns")

    return total_df[['title', 'artist', 'songid', 'streams', 'region', 'danceability',
                     'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                     'instrumentalness', 'liveness', 'valence', 'tempo',
                     'duration_ms', 'time_signature']]

token = genius_lyrics_client_token # local security in local file keys

from lyricsgenius import Genius
genius = Genius(token)

def get_lyrics(title, artist):
    try:
        song = genius.search_song(title = title , artist  = artist )
        lyrics = song.lyrics
    except:
        print(f"Cant find lyrics for {title}")
        lyrics = np.nan

    return lyrics

df = top200_chart("it")

df["lyrics"] = df[["title","artist"]].apply(lambda x:get_lyrics(title =x["title"],
                                                                artist = x["artist"]),
                                            axis = 1 )