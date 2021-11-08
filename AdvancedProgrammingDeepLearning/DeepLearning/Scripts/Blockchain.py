# Place your own Etherscan API key here
from Keys import ETHERSCAN_API_KEY
import json
import time
import requests
import pandas as pd
import networkx as nx

ADDRESS = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
def get_last_block():
    return int(json.loads(requests.get(
        f"https://api.etherscan.io/api?module=block&action=getblocknobytime&timestamp={round(time.time())}&closest=before&apikey={ETHERSCAN_API_KEY}"
    ).text)["result"])

def get_last_txs(since=5):
    return json.loads(requests.get(
        f"https://api.etherscan.io/api?module=account&action=txlist&address={ADDRESS}&startblock={get_last_block() - since}&sort=asc&apikey={ETHERSCAN_API_KEY}"
    ).text)["result"]

txs = get_last_txs()


df = pd.DataFrame(txs)
G = nx.from_pandas_edgelist(source='from' , target='to' , df = df)


