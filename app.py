import pandas as pd # Pandas used for read file
import yfinance as yf # Yahoo Finance used for fetching stock historical data
import numpy as np # Numpy used for log calculation
from datetime import datetime, timedelta
import math 
from scipy.stats import norm #Used for getting z values of the confidence level
from pathlib import Path #Used for getting file paths
from scipy.stats import genpareto
import xml.etree.ElementTree as ET
import glob
import os
from scipy.stats import norm
from scipy.optimize import brentq
from growwapi import GrowwAPI
from datetime import date
import streamlit as st

#===================================================================================
#Defining paths for input and output
input_path = "./Input" # Input path used for picking up the input files

folder = Path(input_path)
#====================================================================================
#Grow access keys
user_api_key = os.getenv("GROWW_API_KEY")
user_secret = os.getenv("GROWW_SECRET")

access_token = GrowwAPI.get_access_token(api_key = user_api_key, secret = user_secret) 
groww = GrowwAPI(access_token)
#======================================================================================
#Reading previous day Daily Volatility Report
previous_volatility_file_path = next(folder.glob("FOVOLT*"), None)
if not previous_volatility_file_path: 
    raise FileNotFoundError(fr"No previous volatility file present in the folder location "+ input_path)

previous_volatility_file = pd.read_csv(previous_volatility_file_path)

#======================================================================================
#Reading exposure file
exposure_file_path = next(folder.glob("ael*"), None)
if not previous_volatility_file_path: 
    raise FileNotFoundError(fr"No exposure file present in the folder location "+ input_path)

exposure_file = pd.read_csv(exposure_file_path)
#======================================================================================
#Reading market lot file
market_lot_file_path = next(folder.glob("fo_m*"), None)
if not previous_volatility_file_path: 
    raise FileNotFoundError(fr"No market lot file present in the folder location "+ input_path)

market_lot_file = pd.read_csv(market_lot_file_path)
#=======================================================================================
#Getting the instruments and expiry dates from Groww
@st.cache_data
def load_instruments():
    return groww.get_all_instruments()

instruments_df = load_instruments()
#========================================================================================
#Getting the list of all the instruments
underlying={}

underlying_list = instruments_df[
    (instruments_df['instrument_type'] == "FUT") & 
    (~instruments_df['underlying_symbol'].str.endswith("TEST", na=False))
].groupby('underlying_symbol')['underlying_symbol'].unique().to_dict()

#========================================================================================
#Getting the expiry date for the instrument
expiry_list={}

expiry_list = instruments_df[instruments_df['instrument_type'] == "FUT"].groupby('underlying_symbol')['expiry_date'].unique().to_dict()
#=========================================================================================

risk_free_rate_mibor=0.0625
#==========================================================================================



#==========================================================================================
def run_risk_analysis(symbol, exp_date, scenario, u_specified_price, buy_sell):
    scenario1 = "Current SPAN"
    scenario2 = "What-if Analysis" 
    min_date=1000000000   
    for j in range(len(instruments_df)):
        if symbol==instruments_df.loc[j,"underlying_symbol"] and pd.notna(instruments_df.loc[j,"expiry_date"]):
            exp_date=instruments_df.loc[j,"expiry_date"]
            exp_date=int(exp_date.replace("-",""))
            min_date=int(min_date)
            if exp_date<min_date:
                min_date=exp_date
                exchange=instruments_df.loc[j,"exchange"]

            min_date=str(min_date)
            year= str(min_date[2:4])   
            month = str(min_date[4:6]) 
            if month=="01":
                mon="JAN"
            elif month=="02":
                mon="FEB"
            elif month=="03":
                mon="MAR"
            elif month=="04":
                mon="APR"
            elif month=="05":
                mon="MAY"
            elif month=="06":
                mon="JUN"
            elif month=="07":
                mon="JUL"
            elif month=="08":
                mon="AUG"
            elif month=="09":
                mon="SEP"
            elif month=="10":
                mon="OCT"
            elif month=="11":
                mon="NOV"
            elif month=="12":
                mon="DEC"
            
            ltp_symbol=exchange+"_"+symbol+year+mon+"FUT"
        
    ltp_response = groww.get_ltp(
    segment=groww.SEGMENT_FNO,
    exchange_trading_symbols=ltp_symbol
    )
    ltp=ltp_response[ltp_symbol]

    if symbol!="SX40" and symbol!="SENSEX50":
        if symbol=="BANKEX":
            tic="BSE-BANK.BO"
        elif symbol=="BANKNIFTY":
            tic="^NSEBANK"
        elif symbol=="FINNIFTY":
            tic="NIFTY_FIN_SERVICE.NS"
        elif symbol=="MIDCPNIFTY":
            tic="NIFTY_MID_SELECT.NS"
        elif symbol=="NIFTY":
            tic="^NSEI"
        elif symbol=="NIFTYNXT50":
            tic="^NSMIDCP"
        elif symbol=="SENSEX":
            tic="^BSESN"
        #elif symbol=="SENSEX50":
            #symbol="SNSX50.BO"
        else:
            tic=symbol +".NS"
        ticker = yf.Ticker(tic)
        latest_price = ticker.history(period="5d",auto_adjust=True)["Close"].iloc[-1]


    if scenario==scenario1:
        for i in range(len(previous_volatility_file)):
            if symbol==previous_volatility_file.loc[i," Symbol"]:
                previous_close=previous_volatility_file.loc[i," Underlying Close Price (A)"]
                log_return=math.log(float(latest_price)/float(previous_close))
                previous_underlying_volatility=previous_volatility_file.loc[i," Current Day Underlying Daily Volatility (E) = Sqrt (0.995*D*D + 0.005* C*C)"]
                current_underlying_volatility=math.sqrt((0.995*previous_underlying_volatility*previous_underlying_volatility)+(0.005*log_return*log_return))
                underlying_annual_volatility=current_underlying_volatility*math.sqrt(365)
                current_futures_closing_price=float(ltp)
                previous_futures_closing_price=previous_volatility_file.loc[i," Futures Close Price (G)"]
                futures_log_return=math.log(float(current_futures_closing_price)/float(previous_futures_closing_price))
                previous_futures_volatility=previous_volatility_file.loc[i," Current Day Futures Daily Volatility (K) = Sqrt (0.995*J*J + 0.005* I*I)"]
                current_futures_volatility=math.sqrt((0.995*float(previous_futures_volatility)*float(previous_futures_volatility))+(0.005*float(futures_log_return)*float(futures_log_return)))
                futures_annual_volatility=current_futures_volatility*math.sqrt(365)
                applicable_daily_volatility=np.maximum(float(current_underlying_volatility),float(current_futures_volatility))
                applicable_annual_volatility=np.maximum(float(underlying_annual_volatility),float(futures_annual_volatility))

        price_scan_range_multiplier=6*applicable_daily_volatility*math.sqrt(2)    
        volatility_scan_range=applicable_annual_volatility*0.25
        

    elif scenario==scenario2:
        for i in range(len(previous_volatility_file)):
            if symbol==previous_volatility_file.loc[i," Symbol"]:
                previous_close=previous_volatility_file.loc[i," Underlying Close Price (A)"]
                log_return=math.log(float(u_specified_price)/float(previous_close))
                previous_underlying_volatility=previous_volatility_file.loc[i," Current Day Underlying Daily Volatility (E) = Sqrt (0.995*D*D + 0.005* C*C)"]
                current_underlying_volatility=math.sqrt((0.995*previous_underlying_volatility*previous_underlying_volatility)+(0.005*log_return*log_return))
                underlying_annual_volatility=current_underlying_volatility*math.sqrt(365)
                previous_futures_closing_price=previous_volatility_file.loc[i," Futures Close Price (G)"]
                current_futures_closing_price=(previous_futures_closing_price*u_specified_price)/previous_close
                futures_log_return=math.log(float(current_futures_closing_price)/float(previous_futures_closing_price))
                previous_futures_volatility=previous_volatility_file.loc[i," Current Day Futures Daily Volatility (K) = Sqrt (0.995*J*J + 0.005* I*I)"]
                current_futures_volatility=math.sqrt((0.995*float(previous_futures_volatility)*float(previous_futures_volatility))+(0.005*float(futures_log_return)*float(futures_log_return)))
                futures_annual_volatility=current_futures_volatility*math.sqrt(365)
                applicable_daily_volatility=np.maximum(float(current_underlying_volatility),float(current_futures_volatility))
                applicable_annual_volatility=np.maximum(float(underlying_annual_volatility),float(futures_annual_volatility))

        price_scan_range_multiplier=6*applicable_daily_volatility*math.sqrt(2)    
        volatility_scan_range=applicable_annual_volatility*0.25
        ltp=current_futures_closing_price
        
        

    zz=0
    for i in range(len(market_lot_file)):
        if market_lot_file.loc[i,"SYMBOL    "].replace(" ","")==symbol:
            lot_size=float(market_lot_file.iloc[i,2])
            zz=i

    if zz<5:
        if price_scan_range_multiplier<0.093:
            price_scan_range_multiplier=0.093
        if volatility_scan_range<0.04:
                volatility_scan_range=0.04
    elif zz>5:
        if price_scan_range_multiplier<0.142:
            price_scan_range_multiplier=0.142
        if volatility_scan_range<0.1:
                volatility_scan_range=0.01

    price_scan=ltp*price_scan_range_multiplier

    risk_array=[]

    risk_array1=ltp*2*risk_free_rate_mibor/365
    risk_array.append(risk_array1)
    risk_array2=ltp*2*risk_free_rate_mibor/365
    risk_array.append(risk_array2)
    risk_array3=-1*(((price_scan/3)*math.exp(risk_free_rate_mibor*2/365))-risk_array1)
    risk_array.append(risk_array3)
    risk_array4=-1*(((price_scan/3)*math.exp(risk_free_rate_mibor*2/365))-risk_array1)
    risk_array.append(risk_array4)
    risk_array5=1*(((price_scan/3)*math.exp(risk_free_rate_mibor*2/365))+risk_array1)
    risk_array.append(risk_array5)
    risk_array6=1*(((price_scan/3)*math.exp(risk_free_rate_mibor*2/365))+risk_array1)
    risk_array.append(risk_array6)
    risk_array7=-1*(((2*price_scan/3)*math.exp(risk_free_rate_mibor*2/365))-risk_array1)
    risk_array.append(risk_array7)
    risk_array8=-1*(((2*price_scan/3)*math.exp(risk_free_rate_mibor*2/365))-risk_array1)
    risk_array.append(risk_array8)
    risk_array9=1*(((2*price_scan/3)*math.exp(risk_free_rate_mibor*2/365))+risk_array1)
    risk_array.append(risk_array9)
    risk_array10=1*(((2*price_scan/3)*math.exp(risk_free_rate_mibor*2/365))+risk_array1)
    risk_array.append(risk_array10)
    risk_array11=-1*(((3*price_scan/3)*math.exp(risk_free_rate_mibor*2/365))-risk_array1)
    risk_array.append(risk_array11)
    risk_array12=-1*(((3*price_scan/3)*math.exp(risk_free_rate_mibor*2/365))-risk_array1)
    risk_array.append(risk_array12)
    risk_array13=1*(((3*price_scan/3)*math.exp(risk_free_rate_mibor*2/365))+risk_array1)
    risk_array.append(risk_array13)
    risk_array14=1*(((3*price_scan/3)*math.exp(risk_free_rate_mibor*2/365))+risk_array1)
    risk_array.append(risk_array14)
    risk_array15=-1*((0.35*2*price_scan)-0.35*risk_array1)
    risk_array.append(risk_array15)
    risk_array16=+1*((0.35*2*price_scan)+0.35*risk_array1)
    risk_array.append(risk_array16)

    if buy_sell=="Buy":
        initial_var=max(risk_array)*lot_size
    elif buy_sell=="Sell":
        initial_var=min(risk_array)*lot_size*-1

    elm_rate=ltp*lot_size*0.02
    for k in range(len(exposure_file)):
        if exposure_file.loc[k,"Symbol"]==symbol and exposure_file.loc[k,"Instrument Type"]=="OTH":
            elm_rate=ltp*lot_size*exposure_file.loc[k,"Total applicable ELM%"]/100

    total_margin=(elm_rate+initial_var)
    return {
        "Initial VAR": round(initial_var,2), 
        "ELM Margin": round(elm_rate,2),
        "Total Margin": round(total_margin,2)
    }

#===============================================================
#initail_margin,elm_margin,total_margin=run_risk_analysis("NIFTY", "2026-05-26", "What-if Analysis", 26000, "Buy")


#scenario1 = "Current SPAN"
#scenario2 = "What-if Analysis" 




st.title("📊 Risk Scenario Analyzer")

# --- 1. Select Symbol ---
symbols = list(underlying_list.keys())
symbol = st.selectbox("Select Underlying", symbols)

# --- 2. Select Expiry ---
expiry_options = expiry_list.get(symbol, [])
expiry = st.selectbox("Select Expiry Date", expiry_options)

# --- 3. Scenario Selection ---
scenario = st.radio(
    "Select Scenario",
    ["Current SPAN", "What-if Analysis"]
)

# --- 4. Conditional Input ---
u_specified_price = None
if scenario == "What-if Analysis":
    u_specified_price = st.number_input(
        "Enter Expected Underlying Price",
        min_value=0.0,
        step=0.1
    )

# --- 5. Buy/Sell Selection ---
buy_sell = st.selectbox("Position Type", ["Buy", "Sell"])

# --- Run Button ---
if st.button("Run Analysis"):

    st.write("### Selected Inputs")
    st.write(f"Symbol: {symbol}")
    st.write(f"Expiry: {expiry}")
    st.write(f"Scenario: {scenario}")

    if scenario == "What-if Analysis":
        st.write(f"User Price: {u_specified_price}")

    # ---------------- CORE LOGIC CALL ----------------
    # You should wrap your large script into a function like:

    result = run_risk_analysis(
        symbol,
        expiry,
        scenario,
        u_specified_price,
        buy_sell
    )
    
    result_df = pd.DataFrame({
    "Metric": list(result.keys()),
    "Value": list(result.values())
    })

    st.subheader("📊 Results")
    st.dataframe(result_df)

    # --- Output ---
    #st.write("### Results")
    #st.json(result)
        



#python -m streamlit run "Risk Array Calculator.py"


