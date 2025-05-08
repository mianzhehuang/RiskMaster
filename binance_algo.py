# -*- coding: utf-8 -*-
"""
Created on Thu May 20 14:35:38 2021

@author: huang
"""

import os
from binance.client import Client

api_key = "SocRWXpL1GtmtLiZa6di4rzbYpbwmyjJOPMW95AFloqlSQdtHS2jcHuilmsBo2sb"
api_secret = "uhbf4GL4VVXkZTzDASDdcldkc10uvvbSacemkKmUeTZDKN7Tkig4TkxXry6qh3R4"

client = Client(api_key, api_secret)
client.API_URL = 'https://testnet.binance.vision/api'
# get balances for all assets & some account information
print(client.get_account())