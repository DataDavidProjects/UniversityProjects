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

region_list = ['global',
 'us',
 'gb',
 'ae',
 'ar',
 'at',
 'au',
 'be',
 'bg',
 'bo',
 'br',
 'ca',
 'ch',
 'cl',
 'co',
 'cr',
 #'cy'# doesnt exist
 'cz',
 'de',
 'dk',
 'do',
 'ec',
 'ee',
 'eg',
 'es',
 'fi',
 'fr',
 'gr',
 'gt',
 'hk',
 'hn',
 'hu',
 'id',
 'ie',
 'il',
 'in',
 'is',
 'it',
 'jp',
 'kr',
 'lt',
 'lu',
 'lv',
 'ma',
 'mx',
 'my',
 'ni',
 'nl',
 'no',
 'nz',
 'pa',
 'pe',
 'ph',
 'pl',
 'pt',
 'py',
 'ro',
 'ru',
 'sa',
 'se',
 'sg',
 'sk',
 'sv',
 'th',
 'tr',
 'tw',
 'ua',
 'uy',
 'vn',
 'za']
global_df = pd.concat( [ top200_chart(r)  for r in region_list ], axis = 0 )

global_df.to_csv("C:/Users/david/Desktop/UniversityProjects/Data Visualization and Text Mining/Data/top200_all.csv",
                 index=False)


