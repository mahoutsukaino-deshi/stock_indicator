import os
import datetime
import pandas as pd
import pandas_datareader
import talib
import matplotlib.pyplot as plt
import seaborn as sns

START_DATE = datetime.date(2020, 1, 1)
END_DATE = datetime.date(2020, 12, 31)
SHOW_GRAPH = False


def get_stock(ticker, start_date, end_date):
    '''
    get stock data from Yahoo Finance
    '''
    dirname = "data"
    os.makedirs(dirname, exist_ok=True)
    fname = f"{dirname}/{ticker}.pkl"
    df_stock = pd.DataFrame()
    if os.path.exists(fname):
        df_stock = pd.read_pickle(fname)
        start_date = df_stock.index.max() + datetime.timedelta(days=1)
    if end_date > start_date:
        df = pandas_datareader.data.DataReader(
            ticker, "yahoo", start_date, end_date)
        df_stock = pd.concat([df_stock, df[~df.index.isin(df_stock.index)]])
        df_stock.to_pickle(fname)
    return df_stock


def make_graph(title, data, style=None, fname=None, show=SHOW_GRAPH):
    '''
    make graph
    '''
    data.plot(style=style)
    plt.title(title)
    if fname is not None:
        plt.savefig(fname)
    if show:
        plt.show()
    plt.close()


def bband(close, graph=False, **kwargs):
    '''
    BBANDS - Bollinger Bands
    '''
    result = talib.BBANDS(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result).T], axis=1)
    df.columns = ['close', 'upperband', 'middleband', 'lowerband']
    if graph:
        title = 'BBANDS - Bollinger Bands'
        style = ['r-']+['--']*(len(df.columns)-1)
        fname = '01_bbands.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def dema(close, graph=False, **kwargs):
    '''
    DEMA - Double Exponential Moving Average
    '''
    result = talib.DEMA(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['close', 'dema']
    if graph:
        title = 'DEMA - Double Exponential Moving Average'
        style = ['r-']+['--']*(len(df.columns)-1)
        fname = '02_dema.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def ema(close, graph=False, **kwargs):
    '''
    EMA - Exponential Moving Average
    '''
    result = talib.EMA(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['close', 'ema']
    if graph:
        title = 'EMA - Exponential Moving Average'
        style = ['r-']+['--']*(len(df.columns)-1)
        fname = '03_ema.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def ht_trendline(close, graph=False, **kwargs):
    '''
    HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
    '''
    result = talib.HT_TRENDLINE(close)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['close', 'ht_trendline']
    if graph:
        title = 'HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline'
        style = ['r-']+['--']*(len(df.columns)-1)
        fname = '04_ht_trendline.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def kama(close, graph=False, **kwargs):
    '''
    KAMA - Kaufman Adaptive Moving Average
    '''
    result = talib.KAMA(close, **kwargs)
    d = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['close', 'kama']
    if graph:
        title = 'KAMA - Kaufman Adaptive Moving Average'
        style = ['r-']+['--']*(len(df.columns)-1)
        fname = '05_kama.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def ma(close, graph=False, **kwargs):
    '''
    MA - Moving average
    '''
    result = talib.MA(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['close', 'ma']
    if graph:
        title = 'MA - Moving average'
        style = ['r-']+['--']*(len(df.columns)-1)
        fname = '06_ma.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def mama(close, graph=False, **kwargs):
    '''
    MAMA - MESA Adaptive Moving Average
    Error occured: Exception: TA_MAMA function failed with error code 2: Bad Parameter (TA_BAD_PARAM)
    '''
    result = talib.MAMA(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result).T], axis=1)
    df.columns = ['close', 'mama', 'fama']
    if graph:
        title = 'MAMA - MESA Adaptive Moving Average'
        style = ['r-']+['--']*(len(df.columns)-1)
        fname = '07_mama.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def mavp(close, graph=False, **kwargs):
    '''
    MAVP - Moving average with variable period
    '''
    result = talib.MAVP(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['close', 'mavp']
    if graph:
        title = 'MAVP - Moving average with variable period'
        style = ['r-']+['--']*(len(df.columns)-1)
        fname = '08_mavp.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def midpoint(close, graph=False, **kwargs):
    '''
    MIDPOINT - MidPoint over period
    '''
    result = talib.MIDPOINT(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['close', 'midpoint']
    if graph:
        title = 'MIDPOINT - MidPoint over period'
        style = ['r-']+['--']*(len(df.columns)-1)
        fname = '09_midpoint.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def midprice(high, low, graph=False, **kwargs):
    '''
    MIDPRICE - Midpoint Price over period
    '''
    result = talib.MIDPRICE(high, low, **kwargs)
    df = pd.concat([pd.DataFrame(high), pd.DataFrame(
        low), pd.DataFrame(result)], axis=1)
    df.columns = ['high', 'low', 'midprice']
    if graph:
        title = 'MIDPRICE - Midpoint Price over period'
        style = ['r-']+['g-']+['--']*(len(df.columns)-2)
        fname = '10_midprice.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def sar(high, low, graph=False, **kwargs):
    '''
    SAR - Parabolic SAR
    '''
    result = talib.SAR(high, low, **kwargs)
    df = pd.concat([pd.DataFrame(high), pd.DataFrame(
        low), pd.DataFrame(result)], axis=1)
    df.columns = ['high', 'low', 'sar']
    if graph:
        title = 'SAR - Parabolic SAR'
        style = ['r-']+['g-']+['--']*(len(df.columns)-2)
        fname = '11_sar.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def sarext(high, low, graph=False, **kwargs):
    '''
    SAREXT - Parabolic SAR - Extended
    '''
    result = talib.SAREXT(high, low, **kwargs)
    df = pd.concat([pd.DataFrame(high), pd.DataFrame(
        low), pd.DataFrame(result)], axis=1)
    df.columns = ['high', 'low', 'sarext']
    if graph:
        title = 'SAREXT - Parabolic SAR - Extended'
        style = ['r-']+['g-']+['--']*(len(df.columns)-2)
        fname = '12_sarext.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def sma(close, graph=False, **kwargs):
    '''
    SMA - Simple Moving Average
    '''
    result = talib.SMA(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['close', 'sma']
    if graph:
        title = 'SMA - Simple Moving Average'
        style = ['r-']+['--']*(len(df.columns)-1)
        fname = '13_sma.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def t3(close, graph=False, **kwargs):
    '''
    T3 - Triple Exponential Moving Average (T3)
    '''
    result = talib.T3(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['close', 't3']
    if graph:
        title = 'T3 - Triple Exponential Moving Average (T3)'
        style = ['r-']+['--']*(len(df.columns)-1)
        fname = '14_t3.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def tema(close, graph=False, **kwargs):
    '''
    TEMA - Triple Exponential Moving Average
    '''
    result = talib.TEMA(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['close', 'tema']
    if graph:
        title = 'TEMA - Triple Exponential Moving Average'
        style = ['r-']+['--']*(len(df.columns)-1)
        fname = '15_tema.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def trima(close, graph=False, **kwargs):
    '''
    TRIMA - Triangular Moving Average
    '''
    result = talib.TRIMA(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['close', 'trima']
    if graph:
        title = 'TRIMA - Triangular Moving Average'
        style = ['r-']+['--']*(len(df.columns)-1)
        fname = '16_trima.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def wma(close, graph=False, **kwargs):
    '''
    WMA - Weighted Moving Average
    '''
    result = talib.WMA(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['close', 'wma']
    if graph:
        title = 'WMA - Weighted Moving Average'
        style = ['r-']+['--']*(len(df.columns)-1)
        fname = '17_wma.png'
        make_graph(title, df, style=style, fname=fname)
    return df


def rsi(close, graph=False, **kwargs):
    '''
    RSI - Relative Strength Index
    '''
    result = talib.RSI(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['close', 'rsi']
    if graph:
        title = 'RSI - Relative Strength Index'
        fname = '18_rsi.png'

        fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={
                                 'height_ratios': [2, 1]})
        fig.suptitle(title)
        axes[0].plot(df['close'], label='close')
        axes[1].plot(df['rsi'], label='rsi')
        axes[1].axhline(y=30, color='orange', linestyle='--')
        axes[1].axhline(y=70, color='orange', linestyle='--')
        for ax in axes:
            ax.legend(loc='upper left')
        plt.savefig(fname)
        if SHOW_GRAPH:
            plt.show()
        plt.close()
    return df


def stoch(high, low, close, graph=False, **kwargs):
    '''
    STOCH - Stochastic
    '''
    result = talib.STOCH(high, low, close, **kwargs)
    df = pd.concat([pd.DataFrame(high), pd.DataFrame(
        low), pd.DataFrame(close), pd.DataFrame(result).T], axis=1)
    df.columns = ['high', 'low', 'close', 'slowk', 'slowd']
    if graph:
        title = 'STOCH - Stochastic'
        fname = '19_stoch.png'

        fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={
                                 'height_ratios': [2, 1]})
        fig.suptitle(title)
        axes[0].plot(df['high'], label='high')
        axes[0].plot(df['low'], label='low')
        axes[0].plot(df['close'], label='close')
        axes[1].plot(df['slowk'], label='slowk')
        axes[1].plot(df['slowd'], label='slowd')
        for ax in axes:
            ax.legend(loc='upper left')
        plt.savefig(fname)
        if SHOW_GRAPH:
            plt.show()
        plt.close()
    return df


def stochrsi(close, graph=False, **kwargs):
    '''
    STOCHRSI - Stochastic Relative Strength Index
    '''
    result = talib.STOCHRSI(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result).T], axis=1)
    df.columns = ['close', 'fastk', 'fastd']
    if graph:
        title = 'STOCHRSI - Stochastic Relative Strength Index'
        fname = '20_stochrsi.png'

        fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={
                                 'height_ratios': [2, 1]})
        fig.suptitle(title)
        axes[0].plot(df['close'], label='close')
        axes[1].plot(df['fastk'], label='fastk')
        axes[1].plot(df['fastd'], label='fastd')
        for ax in axes:
            ax.legend(loc='upper left')
        plt.savefig(fname)
        if SHOW_GRAPH:
            plt.show()
        plt.close()
    return df


def macd(close, graph=False, **kwargs):
    '''
    MACD - Moving Average Convergence/Divergence
    '''
    result = talib.MACD(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result).T], axis=1)
    df.columns = ['close', 'macd', 'macdsignal', 'macdhist']
    if graph:
        title = 'MACD - Moving Average Convergence/Divergence'
        fname = '21_macd.png'

        fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={
                                 'height_ratios': [2, 1]})
        fig.suptitle(title)
        axes[0].plot(df['close'], label='close')
        axes[1].plot(df['macd'], label='macd')
        axes[1].plot(df['macdsignal'], label='macdsignal')
        axes[1].bar(df.index, height=df['macdhist'],
                    label='macdhist', color='green', ec='green')
        for ax in axes:
            ax.legend(loc='upper left')
        plt.savefig(fname)
        if SHOW_GRAPH:
            plt.show()
        plt.close()
    return df


def adx(high, low, close, graph=False, **kwargs):
    '''
    ADX - Average Directional Movement Index
    '''
    result = talib.ADX(high, low, close, **kwargs)
    df = pd.concat([pd.DataFrame(high), pd.DataFrame(
        low), pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['high', 'low', 'close', 'adx']
    if graph:
        title = 'ADX - Average Directional Movement Index'
        fname = '22_adx.png'

        fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={
                                 'height_ratios': [2, 1]})
        fig.suptitle(title)
        axes[0].plot(df['high'], label='high')
        axes[0].plot(df['low'], label='low')
        axes[0].plot(df['close'], label='close')
        axes[1].plot(df['adx'], label='adx')
        for ax in axes:
            ax.legend(loc='upper left')
        plt.savefig(fname)
        if SHOW_GRAPH:
            plt.show()
        plt.close()
    return df


def willr(high, low, close, graph=False, **kwargs):
    '''
    WILLR - Williams' %R
    '''
    result = talib.WILLR(high, low, close, **kwargs)
    df = pd.concat([pd.DataFrame(high), pd.DataFrame(
        low), pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['high', 'low', 'close', 'willr']
    if graph:
        title = 'WILLR - Williams\' %R'
        fname = '23_willr.png'

        fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={
                                 'height_ratios': [2, 1]})
        fig.suptitle(title)
        axes[0].plot(df['high'], label='high')
        axes[0].plot(df['low'], label='low')
        axes[0].plot(df['close'], label='close')
        axes[1].plot(df['willr'], label='willr')
        for ax in axes:
            ax.legend(loc='upper left')
        plt.savefig(fname)
        if SHOW_GRAPH:
            plt.show()
        plt.close()
    return df


def cci(high, low, close, graph=False, **kwargs):
    '''
    CCI - Commodity Channel Index
    '''
    result = talib.CCI(high, low, close, **kwargs)
    df = pd.concat([pd.DataFrame(high), pd.DataFrame(
        low), pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['high', 'low', 'close', 'cci']
    if graph:
        title = 'CCI - Commodity Channel Index'
        fname = '24_cci.png'

        fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={
                                 'height_ratios': [2, 1]})
        fig.suptitle(title)
        axes[0].plot(df['high'], label='high')
        axes[0].plot(df['low'], label='low')
        axes[0].plot(df['close'], label='close')
        axes[1].plot(df['cci'], label='cci')
        for ax in axes:
            ax.legend(loc='upper left')
        plt.savefig(fname)
        if SHOW_GRAPH:
            plt.show()
        plt.close()
    return df


def atr(high, low, close, graph=False, **kwargs):
    '''
    ATR - Average True Range
    '''
    result = talib.ATR(high, low, close, **kwargs)
    df = pd.concat([pd.DataFrame(high), pd.DataFrame(
        low), pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['high', 'low', 'close', 'atr']
    if graph:
        title = 'ATR - Average True Range'
        fname = '25_atr.png'

        fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={
                                 'height_ratios': [2, 1]})
        fig.suptitle(title)
        axes[0].plot(df['high'], label='high')
        axes[0].plot(df['low'], label='low')
        axes[0].plot(df['close'], label='close')
        axes[1].plot(df['atr'], label='atr')
        for ax in axes:
            ax.legend(loc='upper left')
        plt.savefig(fname)
        if SHOW_GRAPH:
            plt.show()
        plt.close()
    return df


def ultosc(high, low, close, graph=False, **kwargs):
    '''
    ULTOSC - Ultimate Oscillator
    '''
    result = talib.ULTOSC(high, low, close, **kwargs)
    df = pd.concat([pd.DataFrame(high), pd.DataFrame(
        low), pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['high', 'low', 'close', 'ultosc']
    if graph:
        title = 'ULTOSC - Ultimate Oscillator'
        fname = '26_ultosc.png'

        fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={
                                 'height_ratios': [2, 1]})
        fig.suptitle(title)
        axes[0].plot(df['high'], label='high')
        axes[0].plot(df['low'], label='low')
        axes[0].plot(df['close'], label='close')
        axes[1].plot(df['ultosc'], label='ultosc')
        for ax in axes:
            ax.legend(loc='upper left')
        plt.savefig(fname)
        if SHOW_GRAPH:
            plt.show()
        plt.close()
    return df


def roc(close, graph=False, **kwargs):
    '''
    ROC - Rate of change : ((price/prevPrice)-1)*100
    '''
    result = talib.ROC(close, **kwargs)
    df = pd.concat([pd.DataFrame(close), pd.DataFrame(result)], axis=1)
    df.columns = ['close', 'roc']
    if graph:
        title = 'ROC - Rate of change : ((price/prevPrice)-1)*100'
        fname = '27_roc.png'

        fig, axes = plt.subplots(nrows=2, sharex=True, gridspec_kw={
                                 'height_ratios': [2, 1]})
        fig.suptitle(title)
        axes[0].plot(df['close'], label='close')
        axes[1].plot(df['roc'], label='roc')
        for ax in axes:
            ax.legend(loc='upper left')
        plt.savefig(fname)
        if SHOW_GRAPH:
            plt.show()
        plt.close()
    return df


def main(ticker):
    sns.set()

    df = get_stock(ticker, START_DATE, END_DATE)

    result_all = df.copy()

    # BBANDS - Bollinger Bands
    result = bband(df['Close'], graph=True, timeperiod=5,
                   nbdevup=2, nbdevdn=2, matype=0)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # DEMA - Double Exponential Moving Average
    result = dema(df['Close'], graph=True, timeperiod=30)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # EMA - Exponential Moving Average
    result = ema(df['Close'], graph=True, timeperiod=30)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
    result = ht_trendline(df['Close'], graph=True)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # KAMA - Kaufman Adaptive Moving Average
    result = kama(df['Close'], graph=True, timeperiod=30)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # MA - Moving average
    result = ma(df['Close'], graph=True, timeperiod=30, matype=0)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # MAMA - MESA Adaptive Moving Average
    # Error occured: Exception: TA_MAMA function failed with error code 2: Bad Parameter (TA_BAD_PARAM)
    # result = mama(df['Close'], graph=True, fastlimit=0, slowlimit=0)
    # result_all = pd.concat([result_all, result.iloc[:,1:]], axis=1)

    # MAVP - Moving average with variable period
    # periodsに何を指定するか不明
    # result = mavp(df['Close'], periods, minperiod=2, maxperiod=30, matype=0)
    # result_all = pd.concat([result_all, result.iloc[:,1:]], axis=1)

    # MIDPOINT - MidPoint over period
    result = midpoint(df['Close'], graph=True, timeperiod=14)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # MIDPRICE - Midpoint Price over period
    result = midprice(df['High'], df['Low'], graph=True, timeperiod=14)
    result_all = pd.concat([result_all, result.iloc[:, 2:]], axis=1)

    # SAR - Parabolic SAR
    result = sar(df['High'], df['Low'], graph=True, acceleration=0, maximum=0)
    result_all = pd.concat([result_all, result.iloc[:, 2:]], axis=1)

    # SAREXT - Parabolic SAR - Extended
    result = sarext(df['High'], df['Low'],
                    graph=True, startvalue=0, offsetonreverse=0, accelerationinitlong=0,
                    accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0,
                    accelerationshort=0, accelerationmaxshort=0)
    result_all = pd.concat([result_all, result.iloc[:, 2:]], axis=1)

    # SMA - Simple Moving Average
    result = sma(df['Close'], graph=True, timeperiod=30)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # T3 - Triple Exponential Moving Average (T3)
    result = t3(df['Close'], graph=True, timeperiod=5, vfactor=0)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # TEMA - Triple Exponential Moving Average
    result = tema(df['Close'], graph=True, timeperiod=5)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # TRIMA - Triangular Moving Average
    result = trima(df['Close'], graph=True, timeperiod=30)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # WMA - Weighted Moving Average
    result = wma(df['Close'], graph=True, timeperiod=30)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # RSI - Relative Strength Index
    result = rsi(df['Close'], graph=True, timeperiod=15)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # STOCH - Stochastic
    result = stoch(df['High'], df['Low'], df['Close'], graph=True, fastk_period=5,
                   slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    result_all = pd.concat([result_all, result.iloc[:, 3:]], axis=1)

    # STOCHRSI - Stochastic Relative Strength Index
    result = stochrsi(df['Close'], graph=True, timeperiod=14,
                      fastk_period=5, fastd_period=3, fastd_matype=0)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # MACD - Moving Average Convergence/Divergence
    result = macd(df['Close'], graph=True, fastperiod=12,
                  slowperiod=26, signalperiod=9)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    # ADX - Average Directional Movement Index
    result = adx(df['High'], df['Low'], df['Close'], graph=True, timeperiod=14)
    result_all = pd.concat([result_all, result.iloc[:, 3:]], axis=1)

    # WILLR - Williams' %R
    result = willr(df['High'], df['Low'], df['Close'],
                   graph=True, timeperiod=14)
    result_all = pd.concat([result_all, result.iloc[:, 3:]], axis=1)

    # CCI - Commodity Channel Index
    result = cci(df['High'], df['Low'], df['Close'], graph=True, timeperiod=14)
    result_all = pd.concat([result_all, result.iloc[:, 3:]], axis=1)

    # ATR - Average True Range
    result = atr(df['High'], df['Low'], df['Close'], graph=True, timeperiod=14)
    result_all = pd.concat([result_all, result.iloc[:, 3:]], axis=1)

    # ULTOSC - Ultimate Oscillator
    result = ultosc(df['High'], df['Low'], df['Close'], graph=True,
                    timeperiod1=7, timeperiod2=14, timeperiod3=28)
    result_all = pd.concat([result_all, result.iloc[:, 3:]], axis=1)

    # ROC - Rate of change : ((price/prevPrice)-1)*100
    result = roc(df['Close'], graph=True, timeperiod=10)
    result_all = pd.concat([result_all, result.iloc[:, 1:]], axis=1)

    result_all.to_csv('stock_indicator.csv')


if __name__ == '__main__':
    main('^N225')
