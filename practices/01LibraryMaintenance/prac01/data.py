from abc import ABC, abstractmethod
import datetime

import pandas as pd
import numpy as np
import pandas_datareader as pdr
import yfinance as yf

from .util import DataLoadError

__docformat__ = 'restructuredtext en'
__author__ = "<Tommy Lee>"
__all__ = ['DataLoader','NaverData','YahooData']

class YahooData(object):
    def __init__(
            self, ticker: str,
            start: str or datetime.date,
            end: str or datetime.date = datetime.date.today(),
            interval: str = '1d',
            progress: bool = False
    ) -> None:
        self.ticker = ticker
        self.start = start
        self.end = end
        self.interval = interval
        self.progress = progress

    def get_data(self) -> pd.DataFrame:
        data = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            interval=self.interval,
            progress=self.progress
        )
        return data


class NaverData(object):
    def __init__(
            self, ticker: str,
            start: str or datetime.date,
            end: str or datetime.date = datetime.date.today(),
            interval: str = '1d',
    ) -> None:
        self.ticker = ticker
        self.start = start
        self.end = end

    def get_data(self) -> pd.DataFrame:
        data = pdr.DataReader(
            self.ticker,
            'naver',
            start=self.start,
            end=self.end
        )
        data = data.astype(float)
        data.index = pd.to_datetime(data.index)
        return data


class DataLoader():
    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    @staticmethod
    def get_data(
            ticker: str,
            start: str or datetime.date,
            end: str or datetime.date = datetime.date.today(),
            interval: str = '1d',
            progress: bool = False
    ) -> pd.DataFrame:
        print('get from yahoo finance...')

        data = YahooData(
            ticker, start, end, interval, progress
        ).get_data()

        if len(data) == 0:
            print('failed to load data from yahoo, try to get from naver finance...')
            data = NaverData(
                ticker, start, end
            ).get_data()

        if len(data) == 0:
            raise DataLoadError('data is empty')
            return None

        return data