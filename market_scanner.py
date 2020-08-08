import os
import time
import yfinance as yf
import dateutil.relativedelta
from datetime import date
import datetime
import numpy as np
import pandas as pd
import sys

from stocklist import NasdaqController

from tqdm import tqdm
from joblib import Parallel, delayed, parallel_backend
import multiprocessing


###########################
# THIS IS THE MAIN SCRIPT #
###########################

class mainObj:

    def __init__(self,_month_cuttoff=6,_day_cuttoff=3,_std_cuttoff=10,_scanopts=False): #default params 
        self.MONTH_CUTTOFF = _month_cuttoff
        self.DAY_CUTTOFF = _day_cuttoff
        self.STD_CUTTOFF = _std_cuttoff
        self.updated_at = datetime.datetime.now().strftime("%H:%M:%S")
        self.scan_options = _scanopts

    def getData(self, ticker):
        currentDate = datetime.datetime.strptime(
            date.today().strftime("%Y-%m-%d"), "%Y-%m-%d")
        pastDate = currentDate - \
            dateutil.relativedelta.relativedelta(months=self.MONTH_CUTTOFF)

        # Fixes an off-by-one bug(yfinance not getting data for currentDate)
        currentDate += dateutil.relativedelta.relativedelta(days=1)

        sys.stdout = open(os.devnull, "w")
        data = yf.download(ticker, pastDate, currentDate, threads=True)


        sys.stdout = sys.__stdout__
        return data[["Volume","Open","Close"]]

    def getOptions(self, ticker):
        opt_ticker = yf.Ticker(ticker)
        allcalls = []
        allputs = []

        try:
            opt_ticker.options
        except: # no option chain for this ticker
            return ([],[],[])

        for exp in opt_ticker.options:
            opt_calls, opt_puts = opt_ticker.option_chain(exp)

            if (opt_calls.empty or opt_calls.volume.empty) or (opt_puts.empty or opt_puts.volume.empty): # double check
                return ([],[],[])
                
            tmp_call = opt_calls.iloc[opt_calls['volume'].argmax()][['contractSymbol', 'strike', 'percentChange', 'volume', 'openInterest', 'impliedVolatility']]
            tmp_call['Exp'] = exp
            allcalls.append(tmp_call)
            tmp_put = opt_puts.iloc[opt_puts['volume'].argmax()][['contractSymbol', 'strike', 'percentChange', 'volume', 'openInterest', 'impliedVolatility']]
            tmp_put['Exp'] = exp
            allputs.append(tmp_put)
        return allcalls, allputs,opt_ticker.options


    def find_anomalies(self, data, currentDate):
        if data.empty:
            return ([],[],[],[])

        data_std = np.std(data['Volume'])
        data_mean = np.mean(data['Volume'])
        anomaly_cut_off = data_std * self.STD_CUTTOFF
        upper_limit = data_mean + anomaly_cut_off
        data.reset_index(level=0, inplace=True)
        is_outlier = data['Volume'] > upper_limit
        is_in_three_days = ((currentDate - data['Date']) <= datetime.timedelta(days=self.DAY_CUTTOFF))

        changes = data.tail().copy()
        changes['Diff'] = changes['Close'].pct_change() * 100
        changes['Premarket'] = changes['Open'].sub(changes['Close'].shift()).div(changes['Close'].shift()).fillna(0) * 100


        return data[is_outlier & is_in_three_days], changes[['Diff', 'Premarket']], data_std, data_mean

    def customPrint(self, d):
        print("\n\n*********************")
        print(d)
        print("*********************\n\n")

    def parallel_wrapper(self,x, currentDate, positive_scans):
        d, changes, deviation, mean = (self.find_anomalies(self.getData(x), currentDate))
        if (len(d) < 1) or d.empty:
            return
        stonk = dict()
        stonk['Ticker'] = x

        #getting rid of the timestamps for right now, the tickers update but the timestamp is always 00:00:00
        #we'd have to cap the monthly window at 2 if we want "true" intraday ticks
        stonk['Date'] = d['Date'].iloc[-1].strftime("%Y-%m-%d")

        stonk['Volume'] = d['Volume'].iloc[-1]
        stonk['Deviations'] = (d['Volume'].iloc[-1] - mean) / deviation  
        stonk['Open'] = d['Open'].iloc[-1]
        stonk['Close'] = d['Close'].iloc[-1]
        stonk['Premarket'] = changes['Premarket'].iloc[-1]
        stonk['Diff'] = changes['Diff'].iloc[-1]

        #self.customPrint(stonk)
        positive_scans.append(stonk)
        return

    def parallel_options(self,x,positive_scans):
        all_calls, all_puts, opts_exp = self.getOptions(x)


        if len(all_calls) < 1:
            return

        stonk_calls = dict()

        stonk_calls['Strike'] = [call.strike for call in all_calls]
        stonk_calls['Exp'] = [call.Exp for call in all_calls]
        stonk_calls['Volume'] = [call.volume for call in all_calls]
        stonk_calls['OI'] = [call.openInterest for call in all_calls]
        stonk_calls['IV'] = [call.impliedVolatility for call in all_calls]
        stonk_calls['Change'] = [call.percentChange for call in all_calls]

        stonk_puts = dict()

        stonk_puts['Strike'] = [put.strike for put in all_puts]
        stonk_puts['Exp'] = [put.Exp for put in all_calls]
        stonk_puts['Volume'] = [put.volume for put in all_puts]
        stonk_puts['OI'] = [put.openInterest for put in all_puts]
        stonk_puts['IV'] = [put.impliedVolatility for put in all_puts]
        stonk_puts['Change'] = [put.percentChange for put in all_puts]

        stonk_option = dict()
        stonk_option['Ticker'] = x
        stonk_option['Calls'] = stonk_calls
        stonk_option['Puts'] = stonk_puts

        positive_scans.append(stonk_option)
        return

    def main_func(self,doFilter=False,invertFilter=False):
        manager = multiprocessing.Manager()
        positive_scans = manager.list()

        StocksController = NasdaqController(True)
        list_of_tickers = StocksController.getList()
        if doFilter:
            if invertFilter:
                filtered_tickers = pd.read_csv('excluded_tickers.csv', names=['Ticker'], index_col=None, header=None).Ticker.astype('string')
            else:
                filtered_tickers = pd.read_csv('filtered_tickers.csv', names=['Ticker'], index_col=None, header=None).Ticker.astype('string')
            list_of_tickers = list(filtered_tickers.values)

        print(f'num tickers: {len(list_of_tickers)}')

        currentDate = datetime.datetime.strptime(
            date.today().strftime("%Y-%m-%d"), "%Y-%m-%d")
        start_time = time.time()

        print(f'num CPU threads {multiprocessing.cpu_count()}')
        # Double the number of worker threads (API seems to be the bottleneck)
        # Probably needs further testing. If your CPU ignites, remove the multiplier
        num_worker_threads = multiprocessing.cpu_count()
        print(f'num worker threads {num_worker_threads}\n')

        if self.scan_options:
            print('scanning options')
            '''for x in tqdm(list_of_tickers, miniters=1):
                self.parallel_options(x, positive_scans)'''

            with parallel_backend('loky', n_jobs=num_worker_threads):
                Parallel()(delayed(self.parallel_options)(x, positive_scans)
                       for x in tqdm(list_of_tickers, miniters=1))
        else:
            print('scanning stonks')
            with parallel_backend('loky', n_jobs=num_worker_threads * 2):
                Parallel()(delayed(self.parallel_wrapper)(x, currentDate, positive_scans)
                        for x in tqdm(list_of_tickers, miniters=1))

        self.updated_at = datetime.datetime.now().strftime("%H:%M:%S")

        print("\n\n\n\n--- this took %s seconds to run ---" %
              (time.time() - start_time))

        return positive_scans


if __name__ == '__main__':
    results = mainObj(_month_cuttoff=6,_day_cuttoff=3,_std_cuttoff=9,_scanopts=True).main_func() #customize these params to your liking
    for outlier in results:
        print(outlier)
        print('---')
    print(f'\nnum outliers: {len(results)}')