# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 21:33:20 2018

@author: huang
"""

import pyodbc 
cnxn = pyodbc.connect('Driver={SQL Server};'
                      'Server=supwinygt.database.windows.net,1433;'
                      'Database=Supwin;'
                      'Trusted_Connection=yes;'
                      'user=supwinadmin'
                      'password=Adminaccount!')

pyodbc.connect('DRIVER={SQL Server};SERVER=yoursqlAzureServer.database.windows.net,1433', user='yourName@yoursqlAzureServer', password='Password', database='DBName')

cursor = cnxn.cursor()
cursor.execute('SELECT * FROM db_name.Table')