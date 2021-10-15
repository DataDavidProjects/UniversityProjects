from Keys import *
token = genius_lyrics_client_token # local security in local file keys

from lyricsgenius import Genius
genius = Genius(token)

def get_lyrics(title, artist):
    song = genius.search_song(title = title , artist  = artist )
    lyrics = song.lyrics
    return lyrics
