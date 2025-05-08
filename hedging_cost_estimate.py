# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 12:45:26 2022

@author: huang
"""

import pyodbc
import pandas as pd 

cnxn = pyodbc.connect("DRIVER={ODBC Driver 17 for SQL Server};"
                      "Trusted_Connection=No;"
                      "SERVER=47.89.186.236;"
                      "DATABASE=Mainnet;"
                      "UID=sa;"
                      "PWD=vsBNEp8dDNpI4pnF;")


####################################################
# get the trade info ###############################
####################################################
trade_sql = "select * from trades where as_of_date>='2022-03-01'" #where as_of_date = '2022-03-18'"
trade = pd.read_sql(trade_sql, cnxn)

trade_future = trade[trade["symbol"]=="ES"]
trade1 = trade_future[["as_of_date", "action", "quantity","sec_type",
                       "symbol", "trading_class", "multiplier", "price", 
                       "commission","trade_time", "sub_strategy"]] 

trade1 = trade1.sort_values(by="trade_time", ascending=True)
trade1.reset_index(inplace=True)
#################################validate############################
trade1 = trade1.drop([207,208,209,210,211])
#####################################################################

trade1["quantity_direction"] = trade1.apply(lambda x: x["quantity"] if x["action"]=="BOT" else -1 * x["quantity"],axis=1) 
trade1["hedge_value"] = -1 * trade1["quantity_direction"] * trade1["price"]*trade1["multiplier"] -  trade1["commission"]
trade1_sum = trade1[["as_of_date","quantity", "quantity_direction","hedge_value"]].groupby("as_of_date").sum()


#trade1_validate_sum= trade1_validate[["as_of_date","quantity", "quantity_direction","hedge_value"]].groupby("as_of_date").sum()

#trade1_validate_sum = trade1_validate_sum[trade1_validate_sum['quantity_direction']!=0]
#remain_position = pd.DataFrame()
trade2 = trade1[["as_of_date", "trade_time", "quantity", "quantity_direction", "hedge_value"]]
trade3 = trade2.loc[trade2.index.repeat(trade2.quantity)]
trade3["quantity_direction"] = trade3["quantity_direction"]/abs(trade3["quantity_direction"])
trade3["hedge_value"] = trade3["hedge_value"] / trade3["quantity"]

remain_position = pd.DataFrame()
remain_position_values = []
for date in trade1_sum.index:
    trade_temp = trade3[trade3["as_of_date"]==date]    
    trade_temp.reset_index(inplace=True)
    position_number = {}
    count = 0 
    pos_num = trade1_sum["quantity_direction"][date]
    if pos_num != 0:
        for i in trade_temp.index:
            count = count + trade_temp["quantity_direction"][i] 
            print([count, trade_temp["hedge_value"][i]])
            if count !=0 and trade_temp["quantity_direction"][i] == pos_num/abs(pos_num):
                position_number[count] = trade_temp[trade_temp.index==i]
        #new_position = {k:position_number[k] for k in range(1,pos_num+1) if k in position_number}
        if pos_num >0: 
            rangelist = range(1,pos_num+1)
        else:
            rangelist = range(pos_num, 0)
        for  k in rangelist:
            remain_position =pd.concat([remain_position, position_number[k]])
    #new_position = {k: position_number[k] for k in range(1:5) if k in position_number}        
    #new_position_number = {k: position_number[k] for k in range(0:1) }

        
remain_position_sum  = remain_position[["as_of_date", "quantity_direction", "hedge_value"]].groupby("as_of_date").sum()
remain_position_sum["remain_hedge_value"] = remain_position_sum["hedge_value"]
trade1_sum = pd.concat([trade1_sum, remain_position_sum[["remain_hedge_value"]]], axis=1)
trade1_sum["remain_hedge_value"] = trade1_sum["remain_hedge_value"].fillna(0)
trade1_sum["hedge_cost"] = trade1_sum["hedge_value"] - trade1_sum["remain_hedge_value"]
trade1_sum["hedge_cost_per_trade"] = trade1_sum["hedge_cost"] / (trade1_sum["quantity"] - abs(trade1_sum["quantity_direction"])) * 2

##############mannually fix##################
#trade_valiate = trade1[207:212]
#trade_validate_sum = trade_valiate[["as_of_date","quantity", "quantity_direction","hedge_value"]].groupby("as_of_date").sum()
#remain_position_sum["hedge_cost"] = remain_position_sum.apply(lambda x: x["hedge_value"] if pd.isna(x["hedge_cost"]) else x["hedge_cost"], axis=1)

 