import time
import requests
from Keys import *


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
nation_dict = dict({
 'af': 'AFG',
 'al': 'ALB',
 'dz': 'DZA',
 'ad': 'AND',
 'ao': 'AGO',
 'ai': 'AIA',
 'aq': 'ATA',
 'ag': 'ATG',
 'sa': 'SAU',
 'ar': 'ARG',
 'am': 'ARM',
 'aw': 'ABW',
 'au': 'AUS',
 'at': 'AUT',
 'az': 'AZE',
 'bs': 'BHS',
 'bh': 'BHR',
 'bd': 'BGD',
 'bb': 'BRB',
 'be': 'BEL',
 'bz': 'BLZ',
 'bj': 'BEN',
 'bm': 'BMU',
 'bt': 'BTN',
 'by': 'BLR',
 'mm': 'MMR',
 'bo': 'BOL',
 'ba': 'BIH',
 'bw': 'BWA',
 'br': 'BRA',
 'bn': 'BRN',
 'bg': 'BGR',
 'bf': 'BFA',
 'bi': 'BDI',
 'kh': 'KHM',
 'cm': 'CMR',
 'ca': 'CAN',
 'cv': 'CPV',
 'td': 'TCD',
 'cl': 'CHL',
 'cn': 'CHN',
 'cy': 'CYP',
 'va': 'VAT',
 'co': 'COL',
 'km': 'COM',
 'kp': 'PRK',
 'kr': 'KOR',
 'ci': 'CIV',
 'cr': 'CRI',
 'hr': 'HRV',
 'cu': 'CUB',
 'cw': 'CUW',
 'dk': 'DNK',
 'dm': 'DMA',
 'ec': 'ECU',
 'eg': 'EGY',
 'sv': 'SLV',
 'ae': 'ARE',
 'er': 'ERI',
 'ee': 'EST',
 'et': 'ETH',
 'fj': 'FJI',
 'ph': 'PHL',
 'fi': 'FIN',
 'fr': 'FRA',
 'ga': 'GAB',
 'gm': 'GMB',
 'ge': 'GEO',
 'gs': 'SGS',
 'de': 'DEU',
 'gh': 'GHA',
 'jm': 'JAM',
 'jp': 'JPN',
 'gi': 'GIB',
 'dj': 'DJI',
 'jo': 'JOR',
 'gr': 'GRC',
 'gd': 'GRD',
 'gl': 'GRL',
 'gp': 'GLP',
 'gu': 'GUM',
 'gt': 'GTM',
 'gg': 'GGY',
 'gn': 'GIN',
 'gw': 'GNB',
 'gq': 'GNQ',
 'gy': 'GUY',
 'gf': 'GUF',
 'ht': 'HTI',
 'hn': 'HND',
 'hk': 'HKG',
 'in': 'IND',
 'id': 'IDN',
 'ir': 'IRN',
 'iq': 'IRQ',
 'ie': 'IRL',
 'is': 'ISL',
 'bv': 'BVT',
 'im': 'IMN',
 'cx': 'CXR',
 'nf': 'NFK',
 'ax': 'ALA',
 'bq': 'BES',
 'ky': 'CYM',
 'cc': 'CCK',
 'ck': 'COK',
 'fo': 'FRO',
 'fk': 'FLK',
 'hm': 'HMD',
 'mp': 'MNP',
 'mh': 'MHL',
 'um': 'UMI',
 'pn': 'PCN',
 'sb': 'SLB',
 'vg': 'VGB',
 'vi': 'VIR',
 'il': 'ISR',
 'it': 'ITA',
 'je': 'JEY',
 'kz': 'KAZ',
 'ke': 'KEN',
 'kg': 'KGZ',
 'ki': 'KIR',
 'kw': 'KWT',
 'la': 'LAO',
 'ls': 'LSO',
 'lv': 'LVA',
 'lb': 'LBN',
 'lr': 'LBR',
 'ly': 'LBY',
 'li': 'LIE',
 'lt': 'LTU',
 'lu': 'LUX',
 'mo': 'MAC',
 'mk': 'MKD',
 'mg': 'MDG',
 'mw': 'MWI',
 'my': 'MYS',
 'mv': 'MDV',
 'ml': 'MLI',
 'mt': 'MLT',
 'ma': 'MAR',
 'mq': 'MTQ',
 'mr': 'MRT',
 'mu': 'MUS',
 'yt': 'MYT',
 'mx': 'MEX',
 'fm': 'FSM',
 'md': 'MDA',
 'mn': 'MNG',
 'me': 'MNE',
 'ms': 'MSR',
 'mz': 'MOZ',
 'na': 'NAM',
 'nr': 'NRU',
 'np': 'NPL',
 'ni': 'NIC',
 'ne': 'NER',
 'ng': 'NGA',
 'nu': 'NIU',
 'no': 'NOR',
 'nc': 'NCL',
 'nz': 'NZL',
 'om': 'OMN',
 'nl': 'NLD',
 'pk': 'PAK',
 'pw': 'PLW',
 'ps': 'PSE',
 'pa': 'PAN',
 'pg': 'PNG',
 'py': 'PRY',
 'pe': 'PER',
 'pf': 'PYF',
 'pl': 'POL',
 'pr': 'PRI',
 'pt': 'PRT',
 'mc': 'MCO',
 'qa': 'QAT',
 'gb': 'GBR',
 'cd': 'COD',
 'cz': 'CZE',
 'cf': 'CAF',
 'cg': 'COG',
 'do': 'DOM',
 're': 'REU',
 'ro': 'ROU',
 'rw': 'RWA',
 'ru': 'RUS',
 'eh': 'ESH',
 'kn': 'KNA',
 'lc': 'LCA',
 'sh': 'SHN',
 'vc': 'VCT',
 'bl': 'BLM',
 'mf': 'MAF',
 'pm': 'SPM',
 'ws': 'WSM',
 'as': 'ASM',
 'sm': 'SMR',
 'st': 'STP',
 'sn': 'SEN',
 'rs': 'SRB',
 'sc': 'SYC',
 'sl': 'SLE',
 'sg': 'SGP',
 'sx': 'SXM',
 'sy': 'SYR',
 'sk': 'SVK',
 'si': 'SVN',
 'so': 'SOM',
 'es': 'ESP',
 'lk': 'LKA',
 'us': 'USA',
 'za': 'ZAF',
 'sd': 'SDN',
 'ss': 'SSD',
 'sr': 'SUR',
 'sj': 'SJM',
 'se': 'SWE',
 'ch': 'CHE',
 'sz': 'SWZ',
 'tw': 'TWN',
 'tj': 'TJK',
 'tz': 'TZA',
 'tf': 'ATF',
 'io': 'IOT',
 'th': 'THA',
 'tl': 'TLS',
 'tg': 'TGO',
 'tk': 'TKL',
 'to': 'TON',
 'tt': 'TTO',
 'tn': 'TUN',
 'tr': 'TUR',
 'tm': 'TKM',
 'tc': 'TCA',
 'tv': 'TUV',
 'ua': 'UKR',
 'ug': 'UGA',
 'hu': 'HUN',
 'uy': 'URY',
 'uz': 'UZB',
 'vu': 'VUT',
 've': 'VEN',
 'vn': 'VNM',
 'wf': 'WLF',
 'ye': 'YEM',
 'zm': 'ZMB',
 'zw': 'ZWE'})
id_dict = dict({
 'af': 4,
 'al': 8,
 'dz': 12,
 'ao': 24,
 'aq': 10,
 'ar': 32,
 'am': 51,
 'au': 36,
 'at': 40,
 'az': 31,
 'bs': 44,
 'bd': 50,
 'by': 112,
 'be': 56,
 'bz': 84,
 'bj': 204,
 'bt': 64,
 'bo': 68,
 'ba': 70,
 'bw': 72,
 'br': 76,
 'bn': 96,
 'bg': 100,
 'bf': 854,
 'bi': 108,
 'kh': 116,
 'cm': 120,
 'ca': 124,
 'cf': 140,
 'td': 148,
 'cl': 152,
 'cn': 156,
 'co': 170,
 'cg': 178,
 'cd': 180,
 'cr': 188,
 "nan": 807,
 'hr': 191,
 'cu': 192,
 'cy': 196,
 'cz': 203,
 'dk': 208,
 'dj': 262,
 'do': 214,
 'ec': 218,
 'eg': 818,
 'sv': 222,
 'gq': 226,
 'er': 232,
 'ee': 233,
 'et': 231,
 'fk': 238,
 'fj': 242,
 'fi': 246,
 'fr': 250,
 'tf': 260,
 'ga': 266,
 'gm': 270,
 'ge': 268,
 'de': 276,
 'gh': 288,
 'gr': 300,
 'gl': 304,
 'gt': 320,
 'gn': 324,
 'gw': 624,
 'gy': 328,
 'ht': 332,
 'hn': 340,
 'hu': 348,
 'is': 352,
 'in': 356,
 'id': 360,
 'ir': 364,
 'iq': 368,
 'ie': 372,
 'il': 376,
 'it': 380,
 'jm': 388,
 'jp': 392,
 'jo': 400,
 'kz': 398,
 'ke': 404,
 'kp': 408,
 'kr': 410,
 'kw': 414,
 'kg': 417,
 'la': 418,
 'lv': 428,
 'lb': 422,
 'ls': 426,
 'lr': 430,
 'ly': 434,
 'lt': 440,
 'lu': 442,
 'mg': 450,
 'mw': 454,
 'my': 458,
 'ml': 466,
 'mr': 478,
 'mx': 484,
 'md': 498,
 'mn': 496,
 'me': 499,
 'ma': 504,
 'mz': 508,
 'mm': 104,
 'na': 516,
 'np': 524,
 'nl': 528,
 'nc': 540,
 'nz': 554,
 'ni': 558,
 'ne': 562,
 'ng': 566,
 'no': 578,
 'om': 512,
 'pk': 586,
 'pa': 591,
 'pg': 598,
 'py': 600,
 'pe': 604,
 'ph': 608,
 'pl': 616,
 'pt': 620,
 'pr': 630,
 'qa': 634,
 'ro': 642,
 'ru': 643,
 'rw': 646,
 'sa': 682,
 'sn': 686,
 'rs': 688,
 'sl': 694,
 'sk': 703,
 'si': 705,
 'sb': 90,
 'so': 706,
 'za': 710,
 'ss': 728,
 'es': 724,
 'lk': 144,
 'sd': 729,
 'sr': 740,
 'sz': 748,
 'se': 752,
 'ch': 756,
 'sy': 760,
 'tw': 158,
 'tj': 762,
 'tz': 834,
 'th': 764,
 'tl': 626,
 'tg': 768,
 'tt': 780,
 'tn': 788,
 'tr': 792,
 'tm': 795,
 'ug': 800,
 'ua': 804,
 'ae': 784,
 'gb': 826,
 'us': 840,
 'uy': 858,
 'uz': 860,
 'vu': 548,
 've': 862,
 'vn': 704,
 'eh': 732,
 'ye': 887,
 'zm': 894,
 'zw': 716})
inv_map_id = {v: k for k, v in id_dict.items()}

# Create Dataset

def create_dataset(path_file_name = "C:/Users/david/Desktop/UniversityProjects/Data Visualization and Text Mining/Data/top200_all.csv"):
 global_df = pd.concat( [ top200_chart(r)  for r in region_list ], axis = 0 )
 global_df.to_csv(path_file_name,
                 index=False)
 return global_df





# Debugging and creating Datasets
# g = pd.read_csv("C:/Users/david/Desktop/UniversityProjects/Data Visualization and Text Mining/Data/top200_all.csv")
# g = g.loc[g.region != "global"]
# g["id"] = g["region"].map(id_dict) # or id_dict
# g.to_csv("C:/Users/david/Desktop/UniversityProjects/Data Visualization and Text Mining/Data/top200_Noglobal.csv",
#                  index=False)
# means = g.groupby('id').mean()[['streams','danceability','energy','speechiness','time_signature','valence','duration_ms']]
# means =means.reset_index()
# means["region"] = means["id"].map(inv_map_id)
# means["region"] = means["region"].str.upper()
# means["id"] = means["id"].astype(int)
# means.to_csv("C:/Users/david/Desktop/UniversityProjects/Data Visualization and Text Mining/Data/top200_means.csv",
#                  index=False)