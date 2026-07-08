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

import requests

session = requests.Session()

http_headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nseindia.com/option-chain",
    "Accept": "application/json",
}

# Get cookies once
session.get("https://www.nseindia.com", headers=http_headers)


def get_option_chain(symbol, expiry, instrument_type):
    url = "https://www.nseindia.com/api/option-chain-v3"

    params = {
        "type": instrument_type,
        "symbol": symbol,
        "expiry": expiry,
    }

    r = session.get(url, headers=http_headers, params=params)
    r.raise_for_status()
    data = r.json()
    #records = data["records"]
    rows = []

    for item in data["records"]["data"]:

        ce = item.get("CE", {})
        pe = item.get("PE", {})

        rows.append({
            "Strike": item["strikePrice"],
            "Expiry": item["expiryDates"],

            "CE_LTP": ce.get("lastPrice"),
            "CE_OI": ce.get("openInterest"),
            "CE_ChangeOI": ce.get("changeinOpenInterest"),
            "CE_IV": ce.get("impliedVolatility"),
            "CE_Volume": ce.get("totalTradedVolume"),

            "PE_LTP": pe.get("lastPrice"),
            "PE_OI": pe.get("openInterest"),
            "PE_ChangeOI": pe.get("changeinOpenInterest"),
            "PE_IV": pe.get("impliedVolatility"),
            "PE_Volume": pe.get("totalTradedVolume"),

            "Underlying": ce.get("underlying") or pe.get("underlying"),
            "UnderlyingValue": ce.get("underlyingValue") or pe.get("underlyingValue")
        })

    df = pd.DataFrame(rows)

    return df


#=========================================================================================

#Getting today's date
today = date.today()
today=str(today)
today=int(today.replace("-",""))

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
instruments_df = pd.read_csv("instruments.csv")
#========================================================================================
#Getting the list of all the instruments
underlying_list={}

underlying_list = instruments_df[
    (instruments_df['instrument_type'] == "FUT") & 
    (~instruments_df['underlying_symbol'].str.endswith("TEST", na=False))
].groupby('underlying_symbol')['underlying_symbol'].unique().to_dict()

#========================================================================================
#Getting the expiry date for the instrument
expiry_list={}

expiry_list = instruments_df[instruments_df['instrument_type'] == "FUT"].groupby('underlying_symbol')['expiry_date'].unique().to_dict()
#=========================================================================================

risk_free_rate_mibor=0.0649
#==========================================================================================
call_df=instruments_df[instruments_df["instrument_type"] == "CE"]
strikes_dict_c = (
    call_df.groupby(["underlying_symbol", "expiry_date"])["strike_price"]
    .unique()
    .apply(lambda x: sorted(list(x)))
    .to_dict()
)

put_df=instruments_df[instruments_df["instrument_type"] == "PE"]
strikes_dict_p = (
    put_df.groupby(["underlying_symbol", "expiry_date"])["strike_price"]
    .unique()
    .apply(lambda x: sorted(list(x)))
    .to_dict()
)
#=========================================================================================
def month_year_calculator(date):

    year=str(date[2:4])
    month=str(date[4:6])
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
    return year,mon
#=========================================================================================
#Metron model to calculate the price of option
def merton_price(S, K, T, r, q, sigma, option_type='C'):
#"""Calculates price using Spot and Dividend Yield."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'C':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'P':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

#==========================================================================================
def calculate_iv_merton(target_price, S, K, T, r, q, option_type='C'):
        """Finds IV using Spot and Dividend."""
        func = lambda sigma: merton_price(S, K, T, r, q, sigma, option_type) - target_price
        try:
            if func(0.0001) * func(5.0) > 0:
                return 0.0            
            return brentq(func, 0.00000001, 5.0)
        except ValueError:
            return 0.0
#======================================================================================
def calculate_price_with_dividend(S, K, T, r, q, sigma, option_type='C'):
    # Standard Black-Scholes adjusted for Dividend Yield (q)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'C':
        return (S * np.exp(-q * T) * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type == 'P':
        return (K * np.exp(-r * T) * norm.cdf(-d2)) - (S * np.exp(-q * T) * norm.cdf(-d1))
#=======================================================================================
#Calculating Composite Delta
def calculate_composite_delta(F, K, T, r, sigma, psr_value, option_type='c'):
    """
    F: Current Futures Price
    K: Strike Price
    T: Time to Expiration (years)
    r: MIBOR (decimal)
    sigma: Implied Volatility (decimal)
    psr_value: Price Scan Range in absolute terms (e.g., F * 0.10)
    """
    
    # 1. Define the 7 Price Points and their Weights
    # Weights: [Unchanged, +/- 33% PSR, +/- 67% PSR, +/- 100% PSR]
    points = [0, 0.33, -0.33, 0.67, -0.67, 1.0, -1.0]
    weights = [0.270, 0.217, 0.217, 0.111, 0.111, 0.037, 0.037] # Standard NSE/SPAN Weights
    
    composite_delta = 0.0
    exp_rt = math.exp(-r * T)

    # 2. Loop through each point to calculate individual deltas
    for i in range(len(points)):
        # Shift the underlying price
        F_shifted = F + (points[i] * psr_value)
        
        # Calculate d1 at this shifted price
        d1 = (math.log(F_shifted / K) + (0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        
        # Calculate Delta for this scenario
        if option_type.lower() == 'c':
            delta_i = exp_rt * norm.cdf(d1)
        elif option_type.lower() == 'p':
            delta_i = exp_rt * (norm.cdf(d1) - 1)
            
        # Add the weighted delta to the total
        composite_delta += delta_i * weights[i]
        
    return composite_delta
#==========================================================================================
def simulate_option_with_exact_curve_slope(sim_spot, current_spot, strike, t, r, dividend, curve_fit, option_type='C'):
    """
    Works for both 'C' (Calls) and 'P' (Puts).
    curve_fit: array of 3 coefficients [a, b, c] from np.polyfit(strikes, ivs, 2)
    """
    # 1. Unpack the curve coefficients
    a, b, c = curve_fit[0], curve_fit[1], curve_fit[2]
    
    # 2. Find the baseline IV for your portfolio strike right now (same for Call and Put)
    base_iv = (a * strike**2) + (b * strike) + c
    
    # 3. Find the exact slope (derivative) at YOUR strike
    exact_skew_slope = (2 * a * strike) + b
    
    # 4. Calculate the price change of the underlying
    spot_change = sim_spot - current_spot
    
    # 5. Shift your IV using the exact slope derived from your curve
    sim_iv = base_iv + (spot_change * exact_skew_slope)
    
    # Failsafe: Volatility cannot be negative or drop below 1%
    sim_iv = max(sim_iv, 0.01)
    
    # 6. Reprice the option dynamically based on option_type ('C' or 'P')
    sim_option_price = calculate_price_with_dividend(
        sim_spot, float(strike), t, r, dividend, sim_iv, option_type=option_type
    )
    
    return sim_option_price, sim_iv, exact_skew_slope
#==========================================================================================
def run_risk_analysis_future(symbol, exp_date, scenario, u_specified_price, buy_sell):
    scenario1 = "Current SPAN"
    scenario2 = "What-if Analysis" 

    expiry_date_psr=exp_date
    expiry_date_psr=int(expiry_date_psr.replace("-",""))
    expiry_date_psr=str(expiry_date_psr)
    year_psr=int(expiry_date_psr[0:4])
    month_psr=int(expiry_date_psr[4:6])
    day_psr=int(expiry_date_psr[6:8])

    today_psr = date.today()    
    today_psr=str(today_psr).replace("-","")
    today_psr_year=int(today_psr[0:4])
    today_psr_month=int(today_psr[4:6])
    today_psr_day=int(today_psr[6:8])

    expiry_date_psr=date(year_psr, month_psr, day_psr)
    today_psr=date(today_psr_year, today_psr_month, today_psr_day)
    diff=expiry_date_psr - today_psr
    day_diff_psr=int(diff.days)
    t_psr=(day_diff)/365


    exp_date=int(exp_date.replace("-",""))
    exp_date=str(exp_date)
    year,mon=month_year_calculator(exp_date)
    
    exchange="NSE"

    ltp_symbol=exchange+"_"+symbol+year+mon+"FUT"
        
    ltp_response = groww.get_ltp(
    segment=groww.SEGMENT_FNO,
    exchange_trading_symbols=ltp_symbol
    )
    ltp=ltp_response[ltp_symbol]

    symbol_underlying=exchange+"_"+symbol
    ltp_underlying=groww.get_ltp(
    segment=groww.SEGMENT_CASH,
    exchange_trading_symbols=symbol_underlying
    )
    latest_price = ltp_underlying[symbol_underlying]

    min_date=1000000000   
    for j in range(len(instruments_df)):
        if symbol==instruments_df.loc[j,"underlying_symbol"] and pd.notna(instruments_df.loc[j,"expiry_date"]) and instruments_df.loc[j,"instrument_type"]=="FUT":
            #print(symbol)
            expiry_date=instruments_df.loc[j,"expiry_date"]
            expiry_date=int(expiry_date.replace("-",""))
            min_date=int(min_date)
            if expiry_date<min_date:
                min_date=expiry_date
            
    #min_date=int(min_date.replace("-",""))
    min_date=str(min_date)
    year,mon=month_year_calculator(min_date)
            
    ltp_future_symbol=exchange+"_"+symbol+year+mon+"FUT"
    ltp_response = groww.get_ltp(
    segment=groww.SEGMENT_FNO,
    exchange_trading_symbols=ltp_future_symbol
    )
    ltp_earliest_future=ltp_response[ltp_future_symbol]


    max_date=0
    for j in range(len(instruments_df)):
        if symbol==instruments_df.loc[j,"underlying_symbol"] and pd.notna(instruments_df.loc[j,"expiry_date"]) and instruments_df.loc[j,"instrument_type"]=="FUT":
            expiry_date=instruments_df.loc[j,"expiry_date"]
            expiry_date=int(expiry_date.replace("-",""))
            max_date=int(max_date)
            if expiry_date>max_date:
                max_date=expiry_date

    max_date=str(max_date)
    year_f,mon_f=month_year_calculator(max_date)
    
    ltp_symbol_f=exchange+"_"+symbol+year_f+mon_f+"FUT"
        
    ltp_response_f = groww.get_ltp(
    segment=groww.SEGMENT_FNO,
    exchange_trading_symbols=ltp_symbol_f
    )
    ltp_fartest_future=ltp_response_f[ltp_symbol_f]

    if scenario==scenario1:
        for i in range(len(previous_volatility_file)):
            if symbol==previous_volatility_file.loc[i," Symbol"]:
                previous_close=previous_volatility_file.loc[i," Underlying Close Price (A)"]
                log_return=math.log(float(latest_price)/float(previous_close))
                previous_underlying_volatility=previous_volatility_file.loc[i," Current Day Underlying Daily Volatility (E) = Sqrt (0.995*D*D + 0.005* C*C)"]
                current_underlying_volatility=math.sqrt((0.995*previous_underlying_volatility*previous_underlying_volatility)+(0.005*log_return*log_return))
                underlying_annual_volatility=current_underlying_volatility*math.sqrt(365)
                current_futures_closing_price=float(ltp_earliest_future)
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
        ltp=(ltp*u_specified_price)/latest_price
        ltp_fartest_future=(ltp_fartest_future*u_specified_price)/latest_price
        
        

    zz=0
    for i in range(len(market_lot_file)):
        if market_lot_file.loc[i,"SYMBOL    "].replace(" ","")==symbol:
            lot_size=float(market_lot_file.iloc[i,2])
            zz=i

    if zz<5:
        ltp_fartest_future=ltp_fartest_future*0.0175
        if price_scan_range_multiplier<0.093:
            price_scan_range_multiplier=0.093
        if volatility_scan_range<0.04:
                volatility_scan_range=0.04
    elif zz>5:
        ltp_fartest_future=ltp_fartest_future*0.02
        if price_scan_range_multiplier<0.142:
            price_scan_range_multiplier=0.142
        if volatility_scan_range<0.1:
                volatility_scan_range=0.01

    if t_psr>0:
        price_scan=ltp*price_scan_range_multiplier*(1+risk_free_rate_mibor*t_psr)
    else:
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
        "Initial VAR": initial_var, 
        "ELM Margin": elm_rate,
        "Total Margin": total_margin,
        "Risk Array":risk_array,
        "Price":ltp,
        "Lot Size":lot_size,
        "ELM Rate":elm_rate/(ltp*lot_size),
        "Fartest Future":ltp_fartest_future
    }
#================================================================
#Risk Analysis for options
def run_risk_analysis_option(symbol, exp_date, scenario, u_specified_price, buy_sell,strike_price,option_type):
    scenario1 = "Current SPAN"
    scenario2 = "What-if Analysis" 
    
    expiry_date_opt=exp_date
    expiry_date_opt=int(expiry_date_opt.replace("-",""))
    expiry_date_opt=str(expiry_date_opt)
    year_ex=int(expiry_date_opt[0:4])
    month_ex=int(expiry_date_opt[4:6])
    day_ex=int(expiry_date_opt[6:8])

    today = date.today()    
    today=str(today).replace("-","")
    year_t=int(today[0:4])
    month_t=int(today[4:6])
    day_t=int(today[6:8])

    expiry_date_o=date(year_ex, month_ex, day_ex)
    today=date(year_t, month_t, day_t)
    diff=expiry_date_o - today
    day_diff=int(diff.days)
    t=(day_diff)/365
    #print(t)
    exp_date=int(exp_date.replace("-",""))
    exp_date=str(exp_date)
    year,mon=month_year_calculator(exp_date)
    
    exchange="NSE"

    ltp_symbol=exchange+"_"+symbol+year+mon+"FUT"
        
    ltp_response = groww.get_ltp(
    segment=groww.SEGMENT_FNO,
    exchange_trading_symbols=ltp_symbol
    )
    ltp=ltp_response[ltp_symbol]

    min_date=10000000000
    for j in range(len(instruments_df)):
        if symbol==instruments_df.loc[j,"underlying_symbol"] and pd.notna(instruments_df.loc[j,"expiry_date"]) and instruments_df.loc[j,"instrument_type"]=="FUT":
            expiry_date=instruments_df.loc[j,"expiry_date"]
            expiry_date=int(expiry_date.replace("-",""))
            min_date=int(min_date)
            if expiry_date<=min_date:
                min_date=expiry_date

    #min_date=int(min_date.replace("-",""))
    min_date=str(min_date)
    #print(min_date)
    year,mon=month_year_calculator(min_date)
    
    ltp_symbol=exchange+"_"+symbol+year+mon+"FUT"
        
    ltp_response = groww.get_ltp(
    segment=groww.SEGMENT_FNO,
    exchange_trading_symbols=ltp_symbol
    )
    ltp_earliest_future=ltp_response[ltp_symbol]

    max_date=0
    for j in range(len(instruments_df)):
        if symbol==instruments_df.loc[j,"underlying_symbol"] and pd.notna(instruments_df.loc[j,"expiry_date"]) and instruments_df.loc[j,"instrument_type"]=="FUT":
            expiry_date=instruments_df.loc[j,"expiry_date"]
            expiry_date=int(expiry_date.replace("-",""))
            max_date=int(max_date)
            if expiry_date>=max_date:
                max_date=expiry_date

    max_date=str(max_date)
    #print(max_date)
    year_f,mon_f=month_year_calculator(max_date)
    
    ltp_symbol_f=exchange+"_"+symbol+year_f+mon_f+"FUT"

    #print(ltp_symbol_f)

    ltp_response_f = groww.get_ltp(
    segment=groww.SEGMENT_FNO,
    exchange_trading_symbols=ltp_symbol_f
    )
    ltp_fartest_future=ltp_response_f[ltp_symbol_f]

    symbol_underlying=exchange+"_"+symbol
    ltp_underlying=groww.get_ltp(
    segment=groww.SEGMENT_CASH,
    exchange_trading_symbols=symbol_underlying
    )
    latest_price = ltp_underlying[symbol_underlying]

    if option_type=="Call":
        opt_type="C"
    else:
        opt_type="P"
    

    exp_date=int(exp_date.replace("-",""))
    exp_date=str(exp_date)
    year,mon=month_year_calculator(exp_date)
    option_symbol=exchange+"_"+symbol+year+mon+str(int(strike_price))+opt_type
    
    date_opt=str(year_ex)+"-"+str(expiry_date_opt[4:6])+"-"+str(day_ex)

#nifty = get_option_chain(
#    "NIFTY",
#    "07-Jul-2026",
#    "Indices"
#)

#print(nifty)

# Stock
#abcapital = get_option_chain(
#    "ABCAPITAL",
#    "28-Jul-2026",
#    "Equity"
#)
#print(abcapital)
    
    zz=0
    for i in range(len(market_lot_file)):
        if market_lot_file.loc[i,"SYMBOL    "].replace(" ","")==symbol:
            lot_size=float(market_lot_file.iloc[i,2])
            zz=i

    if zz<=5:
        option_format="Indices"     
    elif zz>5:
        option_format="Equity"

    nn,mon_ex=month_year_calculator(exp_date)
    mon_ex=mon_ex.capitalize()

    date_format=str(day_ex)+"-"+mon_ex+"-"+str(year_ex)
    #print(date_format)


    df = get_option_chain(symbol, date_format, option_format)

    relevant_call_strike=[]
    relevant_put_strike=[]
    for i in range(len(df)):
        if float(df.loc[i,"Strike"])>=latest_price and float(df.loc[i,"CE_IV"])>0:
            iv_calc=float(df.loc[i,"CE_IV"])/100
            #iv_calc=merton_price(latest_price,float(df.loc[i,"Strike"]),t,risk_free_rate_mibor,0,iv_calc,option_type="C")
            relevant_call_strike.append((float(df.loc[i,"Strike"]),iv_calc))
        elif float(df.loc[i,"Strike"])<latest_price and float(df.loc[i,"PE_IV"])>0:
            iv_calc=float(df.loc[i,"PE_IV"])/100
            #iv_calc=merton_price(latest_price,float(df.loc[i,"Strike"]),t,risk_free_rate_mibor,0,iv_calc,option_type="P")
            relevant_put_strike.append((float(df.loc[i,"Strike"]),iv_calc))

    combined_strikes = sorted(relevant_call_strike + relevant_put_strike, key=lambda x: x[0])
    # Extract the entire 1st column (Strike Prices) -> index 0
    strike_column = [item[0] for item in combined_strikes]

    # Extract the entire 2nd column (IV Values) -> index 1
    iv_column = [item[1] for item in combined_strikes]

    curve_fit=np.polyfit(strike_column,iv_column,2)
    iv_option=np.polyval(curve_fit, float(strike_price))
    #ltp_option=merton_price(latest_price,float(strike_price),t,risk_free_rate_mibor,0,iv_option,option_type=opt_type)
    #print(ltp_option)

    if scenario==scenario1:
        for i in range(len(previous_volatility_file)):
            if symbol==previous_volatility_file.loc[i," Symbol"]:
                previous_close=previous_volatility_file.loc[i," Underlying Close Price (A)"]
                log_return=math.log(float(latest_price)/float(previous_close))
                previous_underlying_volatility=previous_volatility_file.loc[i," Current Day Underlying Daily Volatility (E) = Sqrt (0.995*D*D + 0.005* C*C)"]
                current_underlying_volatility=math.sqrt((0.995*previous_underlying_volatility*previous_underlying_volatility)+(0.005*log_return*log_return))
                underlying_annual_volatility=current_underlying_volatility*math.sqrt(365)
                current_futures_closing_price=float(ltp_earliest_future)
                previous_futures_closing_price=previous_volatility_file.loc[i," Futures Close Price (G)"]
                futures_log_return=math.log(float(current_futures_closing_price)/float(previous_futures_closing_price))
                previous_futures_volatility=previous_volatility_file.loc[i," Current Day Futures Daily Volatility (K) = Sqrt (0.995*J*J + 0.005* I*I)"]
                current_futures_volatility=math.sqrt((0.995*float(previous_futures_volatility)*float(previous_futures_volatility))+(0.005*float(futures_log_return)*float(futures_log_return)))
                futures_annual_volatility=current_futures_volatility*math.sqrt(365)
                applicable_daily_volatility=np.maximum(float(current_underlying_volatility),float(current_futures_volatility))
                applicable_annual_volatility=np.maximum(float(underlying_annual_volatility),float(futures_annual_volatility))

        price_scan_range_multiplier=6*applicable_daily_volatility*math.sqrt(2)    
        #volatility_scan_range=applicable_annual_volatility*0.25
        iv_option=applicable_annual_volatility
        ltp_option=merton_price(latest_price,float(strike_price),t,risk_free_rate_mibor,0,iv_option,option_type=opt_type)
        
        
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
        #volatility_scan_range=applicable_annual_volatility*0.25
        ltp=(ltp*u_specified_price)/latest_price
        ltp_fartest_future=(ltp_fartest_future*u_specified_price)/latest_price
        ltp_option,iv_option,xxx=simulate_option_with_exact_curve_slope(u_specified_price,latest_price,float(strike_price),t,risk_free_rate_mibor,0,curve_fit,option_type=opt_type)
        iv_option=applicable_annual_volatility
        ltp_option=merton_price(u_specified_price,float(strike_price),t,risk_free_rate_mibor,0,iv_option,option_type=opt_type)
        latest_price=u_specified_price

    volatility_scan_range=abs(0.25*applicable_annual_volatility)#iv_option
    zz=0
    for i in range(len(market_lot_file)):
        if market_lot_file.loc[i,"SYMBOL    "].replace(" ","")==symbol:
            lot_size=float(market_lot_file.iloc[i,2])
            zz=i

    if zz<=5:
        if price_scan_range_multiplier<0.093:
            price_scan_range_multiplier=0.093
        #if volatility_scan_range<0.04:
                #volatility_scan_range=0.04
            ltp_fartest_future=ltp_fartest_future*0.0175
    elif zz>5:
        if price_scan_range_multiplier<0.142:
            price_scan_range_multiplier=0.142
        #if volatility_scan_range<0.1:
                #volatility_scan_range=0.01
            ltp_fartest_future=ltp_fartest_future*0.02

    if t>0:
        price_scan=ltp*price_scan_range_multiplier*(1+risk_free_rate_mibor*t)
    else:
        price_scan=ltp*price_scan_range_multiplier

    composite_delta=calculate_composite_delta(ltp,float(strike_price),t,risk_free_rate_mibor,iv_option,price_scan,option_type=opt_type)
    
    #iv=calculate_iv_merton(float(ltp_option),float(strike_price),float(latest_price),t,risk_free_rate_mibor,0,option_type=opt_type)
    #composite_delta=calculate_composite_delta(ltp,float(strike_price),t,risk_free_rate_mibor,iv,price_scan,option_type=opt_type)

    

    zz=0
    for i in range(len(market_lot_file)):
        if market_lot_file.loc[i,"SYMBOL    "].replace(" ","")==symbol:
            lot_size=float(market_lot_file.iloc[i,2])
            zz=i

    if zz<5:
        if volatility_scan_range<0.04:
            volatility_scan_range=0.04
    elif zz>5:
        if volatility_scan_range<0.1:
            volatility_scan_range=0.1

    risk_array=[]


    scenarios = [
        (0,  1),  (0, -1),
        (1/3,  1), (1/3, -1),
        (-1/3,  1), (-1/3, -1),
        (2/3,  1), (2/3, -1),
        (-2/3,  1), (-2/3, -1),
        (1,  1), (1, -1),
        (-1,  1), (-1, -1),
        (2, 0), (-2, 0)
    ]

    therotical_option_price=calculate_price_with_dividend(
            latest_price, float(strike_price), t, risk_free_rate_mibor, 0, iv_option, option_type=opt_type)

    for idx, (price_frac, vol_direction) in enumerate(scenarios, start=1):
        sim_spot = latest_price + (price_frac * price_scan)
        if vol_direction == 1:
            sim_iv = iv_option + volatility_scan_range
        elif vol_direction == -1:
            sim_iv = max(iv_option - volatility_scan_range, 0.01)
        else:
            sim_iv = iv_option
        
        if t-2/365<0:
            t_use=0
        else:
            t_use=t-2/365

        # Calculate new option price
        sim_price = calculate_price_with_dividend(
            sim_spot, float(strike_price), t_use, risk_free_rate_mibor, 0, sim_iv, option_type=opt_type)
        
        # Calculate risk array value (Gain/Loss relative to current option price)
        # Risk arrays track the change in value: Current Price - New Price
        loss_gain = therotical_option_price - sim_price
        
        # Apply 35% scaling for extreme scenarios 15 and 16 per exchange rules
        if idx in [15, 16]:
            loss_gain = loss_gain * 0.35
            
        risk_array.append(loss_gain)


    if buy_sell=="Buy":
        initial_var=max(risk_array)*lot_size
    elif buy_sell=="Sell":
        initial_var=abs(min(risk_array)*lot_size)+ltp_option*lot_size
    #print(min(risk_array))

    if option_type=="Call":
        if float(strike_price)>1.1*latest_price:
            elm_rate=0.03*latest_price*lot_size
        else:
            elm_rate=0.02*latest_price*lot_size

        for k in range(len(exposure_file)):
            if exposure_file.loc[k,"Symbol"]==symbol and float(strike_price)>1.3*latest_price and exposure_file.loc[k,"Instrument Type"]=="OTM":
                elm_rate=latest_price*lot_size*exposure_file.loc[k,"Total applicable ELM%"]/100
            elif exposure_file.loc[k,"Symbol"]==symbol and float(strike_price)<=1.3*latest_price and exposure_file.loc[k,"Instrument Type"]=="OTH":
                elm_rate=latest_price*lot_size*exposure_file.loc[k,"Total applicable ELM%"]/100

    elif option_type=="Put":
        if float(strike_price)<0.9*latest_price:
            elm_rate=0.03*latest_price*lot_size
        else:
            elm_rate=0.02*latest_price*lot_size

        for k in range(len(exposure_file)):
            if exposure_file.loc[k,"Symbol"]==symbol and float(strike_price)<0.7*latest_price and exposure_file.loc[k,"Instrument Type"]=="OTM":
                elm_rate=latest_price*lot_size*exposure_file.loc[k,"Total applicable ELM%"]/100
            elif exposure_file.loc[k,"Symbol"]==symbol and float(strike_price)>=0.7*latest_price and exposure_file.loc[k,"Instrument Type"]=="OTH":
                elm_rate=latest_price*lot_size*exposure_file.loc[k,"Total applicable ELM%"]/100

    
    if buy_sell=="Buy":
        initial_var=0
        elm_rate=0
        premium_rate=-1
    else:
        premium_rate=1
    
    premium=premium_rate*lot_size*ltp_option

    total_margin=(elm_rate+initial_var)

    return {
        "Premium":premium,
        "Initial VAR": initial_var, 
        "ELM Margin": elm_rate,
        "Total Margin": total_margin,
        "Risk Array":risk_array,
        "Price":latest_price,
        "Lot Size":lot_size,
        "ELM Rate":elm_rate/(latest_price*lot_size),
        "Option Price":ltp_option,
        "Composite Delta":composite_delta,
        "Fartest Future":ltp_fartest_future
    }

#===============================================================
#initail_margin,elm_margin,total_margin=run_risk_analysis("NIFTY", "2026-05-26", "What-if Analysis", 26000, "Buy")


#scenario1 = "Current SPAN"
#scenario2 = "What-if Analysis" 


if "orders" not in st.session_state:
    st.session_state.orders = []

st.title("📊 Risk Scenario Analyzer")

# --- 1. Select Symbol ---
symbols = list(underlying_list.keys())
symbol = st.selectbox("Select Underlying", symbols)

instrument = st.selectbox("Instrument", ["Future", "Option"])
option_type=""
if instrument=="Option":
    option_type=st.selectbox("Option Type", ["Call", "Put"])

# --- 2. Select Expiry ---
expiry_options = expiry_list.get(symbol, [])
expiry = st.selectbox("Select Expiry Date", expiry_options)

strk=0
if instrument=="Option" and option_type=="Call":
    call_strike=strikes_dict_c.get((symbol,expiry),[])
    strk=st.selectbox("Select Strike Price",call_strike)
elif instrument=="Option" and option_type=="Put":
    put_strike=strikes_dict_p.get((symbol,expiry),[])
    strk=st.selectbox("Select Strike Price",put_strike)

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

qty = st.number_input("Quantity", min_value=1, value=1)
st.markdown("Lot Size")
#print(symbol)
#st.write(lot_size_dict.get(symbol,0))

if st.button("Add Order"):
    st.session_state.orders.append({
        "Type": instrument,
        "Symbol": symbol,
        "Expiry": expiry,
        "OptionType": option_type,
        "Strike": strk,
        "Qty": qty,
        "Side": buy_sell,
        "Scenario":scenario,
        "Specified Price":u_specified_price,
    })

if st.button("Reset"):
    st.session_state.orders = []

st.subheader("📋 Orders")

if st.session_state.orders:

    header = st.columns([1,1,1,1,1,1,1,1,1,0.75])
    headers = ["Type","Symbol","Expiry","OptType","Strike","Qty","Side","Scenario","Specified Price"]
    
    for col, h in zip(header, headers):
        col.markdown(f"**{h}**")

    for i, order in enumerate(st.session_state.orders):
        row = st.columns([1,1,1,1,1,1,1,1,1,0.75])

        row[0].write(order["Type"])
        row[1].write(order["Symbol"])
        row[2].write(order["Expiry"])
        row[3].write(order["OptionType"])
        row[4].write(order["Strike"])
        row[5].write(order["Qty"])
        row[6].write(order["Side"])
        row[7].write(order["Scenario"])
        row[8].write(order["Specified Price"])

        if row[9].button("❌", key=f"delete_{i}"):
            st.session_state.orders.pop(i)
            st.rerun()

if st.session_state.orders:

    enriched_orders = []
    return_value=[]

    for order in st.session_state.orders:
        if order["Type"]=="Future":
            gg = run_risk_analysis_future(
                order["Symbol"],
                order["Expiry"],
                order["Scenario"],
                order["Specified Price"],
                order["Side"]
            )
        else:
            gg = run_risk_analysis_option(
                order["Symbol"],
                order["Expiry"],
                order["Scenario"],
                order["Specified Price"],
                order["Side"],
                order["Strike"],
                order["OptionType"]
            )
            
        return_value=list(gg.values())

        if order["Type"]=="Future":

            enriched_orders.append({
                **order,
                "Premium": 0,
                "Initial VaR": round(return_value[0]*order["Qty"],2),
                "ELM Margin": round(return_value[1]*order["Qty"],2),
                "Total Margin":round(return_value[2]*order["Qty"],2)
            })
        else:
            enriched_orders.append({
                **order,
                "Premium": round(return_value[0]*order["Qty"],2),
                "Initial VaR": round(return_value[1]*order["Qty"],2),
                "ELM Margin": round(return_value[2]*order["Qty"],2),
                "Total Margin":round(return_value[3]*order["Qty"],2)
            })

    df = pd.DataFrame(enriched_orders)
    st.dataframe(df, use_container_width=True)
# --- Run Button ---
#=================================================================================================================
#=================================================================================================================
#Portfolio Margins

combined_orders = {}

for order in st.session_state.orders:

    key = (
        order["Type"],
        order["Symbol"],
        order["Expiry"],
        order["OptionType"],
        order["Strike"],
        order["Side"]
    )

    if key not in combined_orders:
        combined_orders[key] = order.copy()
    else:
        combined_orders[key]["Qty"] += order["Qty"]

# Replace orders safely
st.session_state.orders = list(combined_orders.values())


# ==========================================================
# Run analysis once for every consolidated position
# ==========================================================

analysis_results = {}

for order in st.session_state.orders:

    if order["Type"] == "Future":

        gg = run_risk_analysis_future(
            order["Symbol"],
            order["Expiry"],
            order["Scenario"],
            order["Specified Price"],
            order["Side"]
        )

    else:

        gg = run_risk_analysis_option(
            order["Symbol"],
            order["Expiry"],
            order["Scenario"],
            order["Specified Price"],
            order["Side"],
            order["Strike"],        # FIX #1
            order["OptionType"]     # FIX #1
        )

    key = (
        order["Type"],
        order["Symbol"],
        order["Expiry"],
        order["OptionType"],
        order["Strike"],
        order["Side"]
    )

    analysis_results[key] = gg
                        
position_details = {}

for order in st.session_state.orders:

    key = (
        order["Type"],
        order["Symbol"],
        order["Expiry"],
        order["OptionType"],
        order["Strike"],
        order["Side"]
    )

    gg = analysis_results[key]

    buy_sell = order["Side"]
    name = order["Symbol"]

    # =====================================================
    # FUTURES
    # =====================================================

    if order["Type"] == "Future":

        type1 = "Future"

        d1 = order["Type"]
        d2 = order["Side"]
        d3 = order["Expiry"]
        d4 = 0

        qty = order["Qty"]
        premium = 0
        strike = 0

        price = gg["Price"]
        lot = gg["Lot Size"]

        exposure_rate = gg["ELM Rate"] * price * lot

        risk_array = gg["Risk Array"]

        if order["Side"] == "Buy":
            delta = 1
        else:
            risk_array = [-val for val in risk_array]
            delta = -1

        exposure_margin = exposure_rate

        spread_margin = gg["Fartest Future"]

    # =====================================================
    # OPTIONS
    # =====================================================

    else:

        spread_margin = gg["Fartest Future"]

        d1 = order["OptionType"]
        d2 = order["Side"]
        d3 = order["Expiry"]
        d4 = order["Strike"]

        qty = order["Qty"]

        u_price = gg["Price"]

        strike = order["Strike"]

        price = gg["Option Price"]

        risk_array = gg["Risk Array"]

        lot = gg["Lot Size"]

        exposure_rate = gg["ELM Rate"]

        if order["OptionType"] == "Call":

            type1 = "Call"

            delta = abs(gg["Composite Delta"])

            if order["Side"] == "Sell":
                risk_array = [-val for val in risk_array]
                delta = -abs(gg["Composite Delta"])

        else:

            type1 = "Put"

            delta = -abs(gg["Composite Delta"])

            if order["Side"] == "Sell":
                risk_array = [-val for val in risk_array]
                delta = abs(gg["Composite Delta"])

        premium = -(price * lot * qty)

        if order["Side"] == "Sell":
            premium = -premium

        exposure_margin = u_price * lot * exposure_rate

        if order["Side"] == "Buy":
            exposure_margin = 0

    # =====================================================
    # STORE POSITION
    # =====================================================

    position_details[(name, d1, d2, d3, d4)] = {

        "Name": name,
        "Buy/Sell": buy_sell,
        "Type": type1,
        "Expiry Date": order["Expiry"],
        "Strike": strike,
        "Quantity": qty,
        "Risk Array": risk_array,
        "Delta": delta,
        "Price": price,
        "Exposure per unit": exposure_margin,
        "Lot Size": lot,
        "Premium": premium, 
        "Spread Margin":spread_margin
    }
array = pd.DataFrame(position_details)

    # 1. Create a dictionary to store the aggregated risk array per underlying
underlying_risk = {}

for details in position_details.values():
    name = details["Name"]
    # Multiply the risk array by quantity (and lot size if risk array is per unit)
    # Most NSE risk arrays are per lot or per unit; adjust 'multiplier' accordingly
    multiplier1 = details["Quantity"] 
    multiplier2=details["Lot Size"]
    current_risk = [val * multiplier1*multiplier2 for val in details["Risk Array"]]
    
    if name not in underlying_risk:
        # Initialize with the first found array
        underlying_risk[name] = current_risk
    else:
        # Element-wise addition for the same underlying
        underlying_risk[name] = [sum(x) for x in zip(underlying_risk[name], current_risk)]

# 2. Find the Max Loss (Span Margin) for each underlying
span_margins = {}
for name, final_array in underlying_risk.items():
    # NSE Span is usually the maximum value in the risk array
    span_margins[name] = max(final_array)

#print("SPAN Margin per Underlying:", span_margins)
final_workbook=[]
for details in position_details.values():
    workbook={
        "Name":details["Name"],
        "Type":details["Type"],
        "Buy/Sell":details["Buy/Sell"],
        "Expiry Date":float(details["Expiry Date"].replace("-","")),
        "Strike":details["Strike"],
        "Quantity":details["Quantity"],
        "Delta":details["Delta"],
        "Price":details["Price"],
        "Exposure per unit":details["Exposure per unit"],
        "Lot Size":details["Lot Size"],
        "Premium":details["Premium"],
        "Quantity_spread":details["Quantity"],
        "Spread Margin": details["Spread Margin"]

    }
    final_workbook.append(workbook)

priority = {
    "Future": 0,
    "Call": 1,
    "Put": 2
}

priority1 = {
    "Sell": 0,
    "Buy": 1
}

sorted_data = sorted(
    final_workbook,
    key=lambda x: (
        x["Name"],                   
        priority[x["Type"]], 
        priority1[x["Buy/Sell"]],         
        x["Expiry Date"],                   
        -int(x["Delta"])
    )
)

new_array = pd.DataFrame(sorted_data)
new_array1 = pd.DataFrame(sorted_data)
unique_name=new_array["Name"].unique()


#Netting all the positions that are exactly opposite to each other
for i in range(len(new_array)):
    for j in range (len(new_array)):
        if new_array.loc[i,"Quantity"]>0 and new_array.loc[j,"Quantity"]>0 and new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]==new_array.loc[j,"Type"] and new_array.loc[i,"Expiry Date"]==new_array.loc[j,"Expiry Date"] and new_array.loc[i,"Strike"]==new_array.loc[j,"Strike"] and new_array.loc[i,"Buy/Sell"]!=new_array.loc[j,"Buy/Sell"]:
            if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                m=new_array.loc[j,"Quantity"]
            else:
                m=new_array.loc[i,"Quantity"]

            new_array.loc[i,"Quantity_spread"]=new_array.loc[i,"Quantity"]-m
            new_array.loc[j,"Quantity_spread"]=new_array.loc[j,"Quantity"]-m

#Calculating spread margins
#Netting spread for long future with short call (same expiry)
#Netting spread for long future with long put (same expiry)
for name in unique_name:
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            for j in range(len(new_array)):
                if new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Future" and new_array.loc[i,"Buy/Sell"]=="Buy" and new_array.loc[i,"Quantity_spread"]>0:
                    if new_array.loc[j,"Type"]=="Call" and new_array.loc[j,"Buy/Sell"]=="Sell" and new_array.loc[j,"Quantity_spread"]>0 and new_array.loc[i,"Expiry Date"]==new_array.loc[j,"Expiry Date"]:
                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            m=new_array.loc[j,"Quantity_spread"]
                        else:
                            m=new_array.loc[i,"Quantity_spread"]
                        new_array.loc[i,"Quantity_spread"]=new_array.loc[i,"Quantity_spread"]-m
                        new_array.loc[j,"Quantity_spread"]=new_array.loc[j,"Quantity_spread"]-m
                    elif new_array.loc[j,"Type"]=="Put" and new_array.loc[j,"Buy/Sell"]=="Buy" and new_array.loc[j,"Quantity_spread"]>0 and new_array.loc[i,"Expiry Date"]==new_array.loc[j,"Expiry Date"]:
                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            m=new_array.loc[j,"Quantity_spread"]
                        else:
                            m=new_array.loc[i,"Quantity_spread"]
                        new_array.loc[i,"Quantity_spread"]=new_array.loc[i,"Quantity_spread"]-m
                        new_array.loc[j,"Quantity_spread"]=new_array.loc[j,"Quantity_spread"]-m
                    
#Netting spread for short future with short put
#Netting spread for short future with long call
for name in unique_name:
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            for j in range(len(new_array)):
                if new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Future" and new_array.loc[i,"Buy/Sell"]=="Sell" and new_array.loc[i,"Quantity_spread"]>0:
                    if new_array.loc[j,"Type"]=="Put" and new_array.loc[j,"Buy/Sell"]=="Sell" and new_array.loc[j,"Quantity_spread"]>0 and new_array.loc[i,"Expiry Date"]==new_array.loc[j,"Expiry Date"]:
                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            m=new_array.loc[j,"Quantity_spread"]
                        else:
                            m=new_array.loc[i,"Quantity_spread"]
                        new_array.loc[i,"Quantity_spread"]=new_array.loc[i,"Quantity_spread"]-m
                        new_array.loc[j,"Quantity_spread"]=new_array.loc[j,"Quantity_spread"]-m
                    elif new_array.loc[j,"Type"]=="Call" and new_array.loc[j,"Buy/Sell"]=="Buy" and new_array.loc[j,"Quantity_spread"]>0 and new_array.loc[i,"Expiry Date"]==new_array.loc[j,"Expiry Date"]:
                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            m=new_array.loc[j,"Quantity_spread"]
                        else:
                            m=new_array.loc[i,"Quantity_spread"]
                        new_array.loc[i,"Quantity_spread"]=new_array.loc[i,"Quantity_spread"]-m
                        new_array.loc[j,"Quantity_spread"]=new_array.loc[j,"Quantity_spread"]-m


#Netting spread for short call with short put (same expiry)(no consideration for strike)
for name in unique_name:
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            for j in range(len(new_array)):
                if new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Call" and new_array.loc[i,"Buy/Sell"]=="Sell" and new_array.loc[i,"Quantity_spread"]>0:
                    if new_array.loc[j,"Type"]=="Put" and new_array.loc[j,"Buy/Sell"]=="Sell" and new_array.loc[j,"Quantity_spread"]>0 and new_array.loc[i,"Expiry Date"]==new_array.loc[j,"Expiry Date"]:
                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            m=new_array.loc[j,"Quantity_spread"]
                        else:
                            m=new_array.loc[i,"Quantity_spread"]
                        new_array.loc[i,"Quantity_spread"]=new_array.loc[i,"Quantity_spread"]-m
                        new_array.loc[j,"Quantity_spread"]=new_array.loc[j,"Quantity_spread"]-m


#Netting spread for long call with long put (same expiry)(no consideration for strike)
for name in unique_name:
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            for j in range(len(new_array)):
                if new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Call" and new_array.loc[i,"Buy/Sell"]=="Buy" and new_array.loc[i,"Quantity_spread"]>0:
                    if new_array.loc[j,"Type"]=="Put" and new_array.loc[j,"Buy/Sell"]=="Buy" and new_array.loc[j,"Quantity_spread"]>0 and new_array.loc[i,"Expiry Date"]==new_array.loc[j,"Expiry Date"]:
                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            m=new_array.loc[j,"Quantity_spread"]
                        else:
                            m=new_array.loc[i,"Quantity_spread"]
                        new_array.loc[i,"Quantity_spread"]=new_array.loc[i,"Quantity_spread"]-m
                        new_array.loc[j,"Quantity_spread"]=new_array.loc[j,"Quantity_spread"]-m


option_price={}
for name in unique_name:
    option_cost=0
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            if new_array.loc[i,"Type"]!="Future":
                if new_array.loc[i,"Buy/Sell"]=="Buy":
                    option_cost=option_cost-(new_array.loc[i,"Quantity"]*new_array.loc[i,"Price"])*new_array.loc[i,"Lot Size"]
                elif new_array.loc[i,"Buy/Sell"]=="Sell":
                    option_cost=option_cost+(new_array.loc[i,"Quantity"]*new_array.loc[i,"Price"])*new_array.loc[i,"Lot Size"]
    option_price[name]={"Option Price":option_cost}

#Priority 1 poistions netting
#Netting futures with opposite direction 
#Netting short futures with short put (with same expiry)
exposure_margin1={}
spread1={}
for name in unique_name:
    exposure=0
    s_s=0
    vv=0
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            for j in range(len(new_array)):
                if new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Future" and new_array.loc[i,"Buy/Sell"]=="Sell" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Future" and new_array.loc[j,"Buy/Sell"]=="Buy" and new_array.loc[j,"Quantity"]>0:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        if new_array.loc[i,"Expiry Date"]>new_array.loc[j,"Expiry Date"]:
                            exposure=exposure+m*new_array.loc[i,"Exposure per unit"]/3
                            v=i
                        elif new_array.loc[i,"Expiry Date"]<new_array.loc[j,"Expiry Date"]:
                            exposure=exposure+m*new_array.loc[j,"Exposure per unit"]/3
                            v=j
                        min_d=min(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        max_d=max(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))

                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            vv=new_array.loc[j,"Quantity_spread"]
                        else:
                            vv=new_array.loc[i,"Quantity_spread"]
                        s_s=s_s+vv*new_array.loc[i,"Lot Size"]*new_array.loc[i,"Spread Margin"]*new_array.loc[v,"Delta"] 
                        spread1[(name)]=s_s
                            
                    elif new_array.loc[j,"Type"]=="Put" and new_array.loc[j,"Buy/Sell"]=="Sell" and new_array.loc[j,"Quantity"]>0 and new_array.loc[i,"Quantity"]>0 and new_array.loc[i,"Expiry Date"]==new_array.loc[j,"Expiry Date"] :
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+ m*(new_array.loc[i,"Exposure per unit"]+new_array.loc[j,"Exposure per unit"])
                    exposure_margin1[(name)]={"Exposure":exposure}
                    
                

#Priority 2 poistions netting
#Netting long futures with short call (with same expiry)
exposure_margin2={}
for name in unique_name:
    exposure=0
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            for j in range(len(new_array)):
                if new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Future" and new_array.loc[i,"Buy/Sell"]=="Buy" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Call" and new_array.loc[j,"Buy/Sell"]=="Sell" and new_array.loc[j,"Quantity"]>0 and new_array.loc[i,"Expiry Date"]==new_array.loc[j,"Expiry Date"] :
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+ m*(new_array.loc[i,"Exposure per unit"]+new_array.loc[j,"Exposure per unit"])
                    exposure_margin2[(name)]={"Exposure":exposure}


#Priority 5 poistions netting
#Netting short future with long call (with same expiry)
#Netting long future with long put (with same expiry)
exposure_margin5={}
for name in unique_name:
    exposure=0
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            for j in range(len(new_array)):
                if new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Future" and new_array.loc[i,"Buy/Sell"]=="Sell" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Call" and new_array.loc[j,"Buy/Sell"]=="Buy" and new_array.loc[j,"Quantity"]>0 and new_array.loc[i,"Expiry Date"]==new_array.loc[j,"Expiry Date"]:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+(m*new_array.loc[i,"Exposure per unit"])

                elif new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Future" and new_array.loc[i,"Buy/Sell"]=="Buy" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Put" and new_array.loc[j,"Buy/Sell"]=="Buy" and new_array.loc[j,"Quantity"]>0 and new_array.loc[i,"Expiry Date"]==new_array.loc[j,"Expiry Date"]:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+(m*new_array.loc[i,"Exposure per unit"])
                    
                exposure_margin5[(name)]={"Exposure":exposure}

#Priority 5.1 poistions netting
#Netting short call with short put (with same expiry)
#Netting short call with short put (with different expiry)
exposure_margin5_1={}
spread1_1={}
for name in unique_name:
    exposure=0
    s_s=0
    vv=0
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            for j in range(len(new_array)):
                if new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Call" and new_array.loc[i,"Buy/Sell"]=="Sell" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Put" and new_array.loc[j,"Buy/Sell"]=="Sell" and new_array.loc[j,"Quantity"]>0 and new_array.loc[i,"Expiry Date"]==new_array.loc[j,"Expiry Date"]:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+m*(new_array.loc[i,"Exposure per unit"]+new_array.loc[j,"Exposure per unit"])
                elif new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Call" and new_array.loc[i,"Buy/Sell"]=="Sell" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Put" and new_array.loc[j,"Buy/Sell"]=="Sell" and new_array.loc[j,"Quantity"]>0 and new_array.loc[i,"Expiry Date"]!=new_array.loc[j,"Expiry Date"]:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+m*(new_array.loc[i,"Exposure per unit"]+new_array.loc[j,"Exposure per unit"])
                        if float(new_array.loc[i,"Expiry Date"])>float(new_array.loc[j,"Expiry Date"]):
                            v=i
                        else:
                            v=j
                        min_d=min(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        max_d=max(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))

                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            vv=new_array.loc[j,"Quantity_spread"]
                        else:
                            vv=new_array.loc[i,"Quantity_spread"]
                        s_s=s_s+vv*new_array.loc[i,"Lot Size"]*new_array.loc[i,"Spread Margin"]*new_array.loc[v,"Delta"]

                        spread1_1[(name)]=s_s
                exposure_margin5_1[(name)]={"Exposure":exposure}

#Priority 6 poistions netting
#Netting short future with long call (with different expiry)
#Netting long future with long put (with different expiry)
exposure_margin6={}
spread3={}
for name in unique_name:
    exposure=0
    s_s=0
    vv=0
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            for j in range(len(new_array)):
                if new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Future" and new_array.loc[i,"Buy/Sell"]=="Sell" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Call" and new_array.loc[j,"Buy/Sell"]=="Buy" and new_array.loc[j,"Quantity"]>0:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+ m*(new_array.loc[i,"Exposure per unit"]+new_array.loc[j,"Exposure per unit"])
                        if float(new_array.loc[i,"Expiry Date"])>float(new_array.loc[j,"Expiry Date"]):
                            v=i
                        else:
                            v=j
                        min_d=min(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        max_d=max(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            vv=new_array.loc[j,"Quantity_spread"]
                        else:
                            vv=new_array.loc[i,"Quantity_spread"]
                        s_s=s_s+vv*new_array.loc[i,"Lot Size"]*new_array.loc[i,"Spread Margin"]*new_array.loc[v,"Delta"]

                elif new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Future" and new_array.loc[i,"Buy/Sell"]=="Buy" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Put" and new_array.loc[j,"Buy/Sell"]=="Buy" and new_array.loc[j,"Quantity"]>0:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+ m*(new_array.loc[i,"Exposure per unit"]+new_array.loc[j,"Exposure per unit"])
                        if float(new_array.loc[i,"Expiry Date"])>float(new_array.loc[j,"Expiry Date"]):
                            v=i
                        else:
                            v=j
                        min_d=min(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        max_d=max(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            vv=new_array.loc[j,"Quantity_spread"]
                        else:
                            vv=new_array.loc[i,"Quantity_spread"]
                        s_s=s_s+vv*new_array.loc[i,"Lot Size"]*new_array.loc[i,"Spread Margin"]*new_array.loc[v,"Delta"]

                exposure_margin6[(name)]={"Exposure":exposure}   
                spread3[(name)]=s_s 

#Priority 3 poistions netting
#Netting short call with long call (with same expiry)
#Netting short put with long put (with same expiry)
exposure_margin3={}
for name in unique_name:
    exposure=0
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            for j in range(len(new_array)):
                if new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Call" and new_array.loc[i,"Buy/Sell"]=="Sell" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Call" and new_array.loc[j,"Buy/Sell"]=="Buy" and new_array.loc[j,"Quantity"]>0 and new_array.loc[i,"Expiry Date"]==new_array.loc[j,"Expiry Date"]:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+(m*new_array.loc[i,"Exposure per unit"])

                elif new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Put" and new_array.loc[i,"Buy/Sell"]=="Sell" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Put" and new_array.loc[j,"Buy/Sell"]=="Buy" and new_array.loc[j,"Quantity"]>0 and new_array.loc[i,"Expiry Date"]==new_array.loc[j,"Expiry Date"]:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=(exposure+m*new_array.loc[i,"Exposure per unit"])

                    
                exposure_margin3[(name)]={"Exposure":exposure}


#Priority 4 poistions netting
#Netting short futures with short put (with different expiry)
#Netting long futures with short call (with different expiry)
exposure_margin4={}
spread2={}
for name in unique_name:
    exposure=0
    s_s=0
    vv=0
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            for j in range(len(new_array)):
                if new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and  new_array.loc[i,"Type"]=="Future" and new_array.loc[i,"Buy/Sell"]=="Sell" and new_array.loc[i,"Quantity"]>0 :
                    if new_array.loc[j,"Type"]=="Put" and new_array.loc[j,"Buy/Sell"]=="Sell" and new_array.loc[j,"Quantity"]>0:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+ m*(new_array.loc[i,"Exposure per unit"]+new_array.loc[j,"Exposure per unit"])
                        if float(new_array.loc[i,"Expiry Date"])>float(new_array.loc[j,"Expiry Date"]):
                            v=i
                        else:
                            v=j
                        min_d=min(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        max_d=max(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            vv=new_array.loc[j,"Quantity_spread"]
                        else:
                            vv=new_array.loc[i,"Quantity_spread"]
                        s_s=s_s+vv*new_array.loc[i,"Lot Size"]*new_array.loc[i,"Spread Margin"]*new_array.loc[v,"Delta"]

                
                elif new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Future" and new_array.loc[i,"Buy/Sell"]=="Buy" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Call" and new_array.loc[j,"Buy/Sell"]=="Sell" and new_array.loc[j,"Quantity"]>0:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+ m*(new_array.loc[i,"Exposure per unit"]+new_array.loc[j,"Exposure per unit"])  
                        min_d=min(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        max_d=max(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        if float(new_array.loc[i,"Expiry Date"])>float(new_array.loc[j,"Expiry Date"]):
                            v=i
                        else:
                            v=j
                        min_d=min(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        max_d=max(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            vv=new_array.loc[j,"Quantity_spread"]
                        else:
                            vv=new_array.loc[i,"Quantity_spread"]
                        s_s=s_s+vv*new_array.loc[i,"Lot Size"]*new_array.loc[i,"Spread Margin"]*new_array.loc[v,"Delta"]

                exposure_margin4[(name)]={"Exposure":exposure} 
                spread2[(name)]=s_s           



        
#Priority 7 poistions netting
#Netting Short call with long call (with same strike and different expiry)
#Netting Short put with long put (with same strike and different expiry)
exposure_margin7={}
spread4={}
for name in unique_name:
    exposure=0
    s_s=0
    vv=0
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            for j in range(len(new_array)):
                if new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Call" and new_array.loc[i,"Buy/Sell"]=="Sell" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Call" and new_array.loc[j,"Buy/Sell"]=="Buy" and new_array.loc[j,"Quantity"]>0 and new_array.loc[i,"Strike"]==new_array.loc[j,"Strike"]:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+(m*new_array.loc[i,"Exposure per unit"])
                        if float(new_array.loc[i,"Expiry Date"])>float(new_array.loc[j,"Expiry Date"]):
                            v=i
                        else:
                            v=j
                        min_d=min(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        max_d=max(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            vv=new_array.loc[j,"Quantity_spread"]
                        else:
                            vv=new_array.loc[i,"Quantity_spread"]
                        s_s=s_s+vv*new_array.loc[i,"Lot Size"]*new_array.loc[i,"Spread Margin"]*new_array.loc[v,"Delta"]
                        
                elif new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Put" and new_array.loc[i,"Buy/Sell"]=="Sell" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Put" and new_array.loc[j,"Buy/Sell"]=="Buy" and new_array.loc[j,"Quantity"]>0 and new_array.loc[i,"Strike"]==new_array.loc[j,"Strike"]:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+(m*new_array.loc[i,"Exposure per unit"])
                        if float(new_array.loc[i,"Expiry Date"])>float(new_array.loc[j,"Expiry Date"]):
                            v=i
                        else:
                            v=j
                        min_d=min(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        max_d=max(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            vv=new_array.loc[j,"Quantity_spread"]
                        else:
                            vv=new_array.loc[i,"Quantity_spread"]
                        s_s=s_s+vv*new_array.loc[i,"Lot Size"]*new_array.loc[i,"Spread Margin"]*new_array.loc[v,"Delta"]
                    
                exposure_margin7[(name)]={"Exposure":exposure}    
                spread4[(name)]=s_s



#Priority 8 poistions netting
#Netting Short call with long call (with different strike and different expiry)
#Netting Short put with long put (with different strike and different expiry)
exposure_margin8={}
spread5={}
for name in unique_name:
    exposure=0
    s_s=0
    vv=0
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            for j in range(len(new_array)):
                if new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Call" and new_array.loc[i,"Buy/Sell"]=="Sell" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Call" and new_array.loc[j,"Buy/Sell"]=="Buy" and new_array.loc[j,"Quantity"]>0:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+(m*new_array.loc[i,"Exposure per unit"])
                        if float(new_array.loc[i,"Expiry Date"])>float(new_array.loc[j,"Expiry Date"]):
                            v=i
                        else:
                            v=j
                        min_d=min(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        max_d=max(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            vv=new_array.loc[j,"Quantity_spread"]
                        else:
                            vv=new_array.loc[i,"Quantity_spread"]
                        s_s=s_s+vv*new_array.loc[i,"Lot Size"]*new_array.loc[i,"Spread Margin"]*new_array.loc[v,"Delta"]
                        
                elif new_array.loc[i,"Name"]==new_array.loc[j,"Name"] and new_array.loc[i,"Type"]=="Put" and new_array.loc[i,"Buy/Sell"]=="Sell" and new_array.loc[i,"Quantity"]>0:
                    if new_array.loc[j,"Type"]=="Put" and new_array.loc[j,"Buy/Sell"]=="Buy" and new_array.loc[j,"Quantity"]>0:
                        if new_array.loc[i,"Quantity"]>=new_array.loc[j,"Quantity"]:
                            m=new_array.loc[j,"Quantity"]
                        else:
                            m=new_array.loc[i,"Quantity"]
                        new_array.loc[i,"Quantity"]=new_array.loc[i,"Quantity"]-m
                        new_array.loc[j,"Quantity"]=new_array.loc[j,"Quantity"]-m
                        exposure=exposure+m*new_array.loc[i,"Exposure per unit"]
                        if float(new_array.loc[i,"Expiry Date"])>float(new_array.loc[j,"Expiry Date"]):
                            v=i
                        else:
                            v=j
                        min_d=min(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        max_d=max(float(new_array.loc[i,"Expiry Date"]),float(new_array.loc[j,"Expiry Date"]))
                        if new_array.loc[i,"Quantity_spread"]>=new_array.loc[j,"Quantity_spread"]:
                            vv=new_array.loc[j,"Quantity_spread"]
                        else:
                            vv=new_array.loc[i,"Quantity_spread"]
                        s_s=s_s+vv*new_array.loc[i,"Lot Size"]*new_array.loc[i,"Spread Margin"]*new_array.loc[v,"Delta"]
                                                
                exposure_margin8[(name)]={"Exposure":exposure}    
                spread5[(name)]=s_s

#Checking if any quantity is left and finding exposure of same
exposure_margin9={}
for name in unique_name:
    exposure=0
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            if new_array.loc[i,"Quantity"]>0:
                exposure=exposure+new_array.loc[i,"Quantity"]*new_array.loc[i,"Exposure per unit"]
            exposure_margin9[(name)]={"Exposure":exposure}    

premium_o={}
for name in unique_name:
    premium_opt=0
    for i in range(len(new_array)):
        if name==new_array.loc[i,"Name"]:
            premium_opt=premium_opt+new_array.loc[i,"Premium"]
    premium_o[(name)]={"Premium":premium_opt}

        
#Finding delta of the fartest expiry for a position


f_e=0
f_s=0
p_p=0
t_m=0
security_margins=[]

for name in unique_name:
    f_es=0
    f_ss=0
    p_ps=0
    t_ms=0
    paid_received_s=""

    
    f_es = exposure_margin1.get(name, {"Exposure": 0})["Exposure"] + \
    exposure_margin2.get(name, {"Exposure": 0})["Exposure"] + \
    exposure_margin3.get(name, {"Exposure": 0})["Exposure"] + \
    exposure_margin4.get(name, {"Exposure": 0})["Exposure"] + \
    exposure_margin5.get(name, {"Exposure": 0})["Exposure"] + \
    exposure_margin5_1.get(name, {"Exposure": 0})["Exposure"] + \
    exposure_margin6.get(name, {"Exposure": 0})["Exposure"] + \
    exposure_margin7.get(name, {"Exposure": 0})["Exposure"] + \
    exposure_margin8.get(name, {"Exposure": 0})["Exposure"] + \
    exposure_margin9.get(name, {"Exposure": 0})["Exposure"]

    f_e=f_e+f_es

    #f_e=f_e+exposure_margin1.get((name))["Exposure"]+exposure_margin2.get((name))["Exposure"]+exposure_margin3.get((name))["Exposure"]+exposure_margin4.get((name))["Exposure"]+exposure_margin5.get((name))["Exposure"]+exposure_margin6.get((name))["Exposure"]+exposure_margin7.get((name))["Exposure"]+exposure_margin8.get((name))["Exposure"]+exposure_margin9.get((name))["Exposure"]
    
    f_ss=abs(spread1.get(name,0)) + \
    abs(spread2.get(name,0)) + \
    abs(spread3.get(name,0)) + \
    abs(spread4.get(name,0)) + \
    abs(spread5.get(name,0)) + \
    abs(spread1_1.get(name,0)) + \
    span_margins.get(name, 0) + \
    option_price.get(name, {"Option Price": 0})["Option Price"]

    if f_ss<0:
        f_ss=0

    f_s=f_s+f_ss
    #f_s=f_s+delta0.get((name))["Delta"]+delta1.get((name))["Delta"]+delta2.get((name))["Delta"]+delta3.get((name))["Delta"]+delta4.get((name))["Delta"]+span_margins.get((name))+option_price.get((name))["Option Price"]
    
    p_ps= premium_o.get(name, {"Premium": 0})["Premium"]
    #p_p=p_p+premium_o.get((name))["Premium"]
    if p_ps<0:
        paid_received_s = "paid"
    elif p_ps>0:
        paid_received_s = "received"

    p_p=p_p+p_ps
    p_ps=abs(p_ps)
    
    t_ms=f_ss+f_es

    security_margins.append({
        "Symbol": name,
        "Span": round(f_ss, 2),
        "Exposure": round(f_es, 2),
        "Total Margin": round(t_ms, 2),
        "Premium": round(p_ps, 2),
        "Paid/Received": paid_received_s
    })

if f_s<0:
    f_s=0

if p_p<0:
    paid_received = "paid"
elif p_p>0:
    paid_received = "received"
else:
    paid_received = ""

if f_s<0:
    f_s=0

p_p=abs(p_p)

t_m=f_s+f_e



#================================================================================================================
if st.button("Run Analysis"):

    st.write("Span Margin", f"₹{f_s:,.2f}")
    st.write("Exposure Margin", f"₹{f_e:,.2f}")
    st.write("Total Margin", f"₹{t_m:,.2f}")
    st.write(f"Premium {paid_received}", f"₹{p_p:,.2f}")


#python -m streamlit run "Risk Array Calculator.py"


