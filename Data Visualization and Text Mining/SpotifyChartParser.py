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

from bs4 import BeautifulSoup
import pandas as pd
import requests
from time import sleep
from datetime import date, timedelta

# create empty arrays for data we're collecting
dates = []
url_list = []
final = []

# map site
url = "https://spotifycharts.com/regional/it/daily/latest/"
start_date = today = date.today()
end_date = today = date.today()

delta = end_date - start_date

for i in range(delta.days + 1):
    day = start_date + timedelta(days=i)
    day_string = day.strftime("%Y-%m-%d")
    dates.append(day_string)


def add_url():
    for date in dates:
        c_string = url + date
        url_list.append(c_string)

time.sleep(4)
add_url()


# function for going through each row in each url and finding relevant song info

def song_scrape(x):

    pg = x
    for tr in songs.find("tbody").findAll("tr"):
        artist = tr.find("td", {"class": "chart-table-track"}).find("span").text
        artist = artist.replace("by ", "").strip()

        title = tr.find("td", {"class": "chart-table-track"}).find("strong").text

        songid = tr.find("td", {"class": "chart-table-image"}).find("a").get("href")
        songid = songid.split("track/")[1]

        url_date = x.split("daily/")[1]

        final.append([title, artist, songid, url_date])


# loop through urls to create array of all of our song info

for u in url_list:
    read_pg = requests.get(u)
    sleep(5)
    soup = BeautifulSoup(read_pg.text, "html.parser")
    songs = soup.find("table", {"class": "chart-table"})
    song_scrape(u)

# convert to data frame with pandas for easier data manipulation

final_df = pd.DataFrame(final, columns=["Title", "Artist", "Song ID", "Chart Date"])

final_df.to_csv("Data/DayUS.csv", index=False)