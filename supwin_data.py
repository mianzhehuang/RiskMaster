# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 21:59:35 2018

@author: huang
"""

import pandas as pd
import pyodbc

def load_data():
	connection_string = "Driver={ODBC Driver 13 for SQL Server};Server=supwinygt.database.windows.net;Port=1433;Database=Supwin;UID=supwinadmin;PWD=Adminaccount!;"
	conn = pyodbc.connect(connection_string)
	query = "SELECT * FROM [dbo].[t_yahoo_option_price]"
	df = pd.read_sql(query, conn)
	return df



data = load_data()
ticker = data[data['contract_symbol']=='SPY180518P00262000']
print(ticker) 
'''
if __name__ == '__main__':
	data = load_data()
   ticker = data[data['contract_symbol']=='SPY180518P00262000'] 
	a = 1
'''  

   