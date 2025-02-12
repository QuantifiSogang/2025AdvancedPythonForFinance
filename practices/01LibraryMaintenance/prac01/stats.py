import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class Stats(ABC) :
    def __init__(
            self, data : pd.DataFrame
        ) -> None :
        """
        통계량을 생성하는 클래스를 위한 추상클래스입니다.
        :param data: pandas.DataFrame형태의 가격 데이터를 넣습니다.
        """
        self.data = data

    @abstractmethod
    def _get_returns(self, attr : str = 'Close') -> pd.Series :
        """
        가격 데이터를 이용해 pandas.Series 형태의 수익률 데이터를 생성합니다.
        :param attr: 수익률 데이터를 생성할 때 참조하는 컬럼 이름입니다.
        """
        pass

    @abstractmethod
    def get_mean(self, attr : str = 'Close', annualized : bool = True) -> pd.Series :
        """
        가격 데이터를 이용해 수익률을 먼저 만든 뒤, 이 수익률의 평균을 계산하는 메서드입니다.
        :param attr: 수익률 데이터를 생성할 때 참조하는 컬럼 이름입니다.
        :param annualized: 연율화를 시행할 때 사용하는 인수입니다. 기본값은 True
        :return:
        """
        pass

    @abstractmethod
    def get_std(self, attr : str = 'Close', annualized : bool = True) -> pd.Series :
        """
        가격 데이터를 이용해 수익률을 먼저 만든 뒤, 이 수익률의 표준편차를 계산하는 메서드입니다.
        :param attr: 수익률 데이터를 생성할 때 참조하는 컬럼 이름입니다.
        :param annualized: 연율화를 시행할 때 사용하는 인수입니다. 기본값은 True
        :return:
        """
        pass

    @abstractmethod
    def get_sharpe_ratio(self, attr : str = 'Close', annualized : bool = True) -> pd.Series :
        """
        가격 데이터를 이용해 수익률을 먼저 만든 뒤, 이 수익률의 sharpe ratio를 계산하는 메서드입니다.
        :param attr: 수익률 데이터를 생성할 때 참조하는 컬럼 이름입니다.
        :param annualized: 연율화를 시행할 때 사용하는 인수입니다. 기본값은 True
        :return:
        """
        pass

class StockStats(Stats) :
    def _get_returns(self, attr : str = 'Close') -> pd.Series:
        ret = self.data[attr].pct_change().dropna()
        ret.name = 'Returns'
        return ret

    def get_mean(self, attr : str = 'Close', annualized : bool = True) -> pd.Series:
        ret = self._get_returns(attr)
        mean = ret.mean()
        if annualized :
            mean *= 252
        return mean

    def get_std(self, attr : str = 'Close', annualized : bool = True) -> pd.Series:
        ret = self._get_returns(attr)
        std = ret.std()
        if annualized :
            std *= np.sqrt(252)
        return std

    def get_sharpe_ratio(self, attr = 'Close', annualized : bool = True) -> pd.Series:
        mean = self.get_mean(attr, annualized)
        std = self.get_std(attr, annualized)
        sharpe_ratio = mean / std
        return sharpe_ratio

    def get_skewness(self, attr = 'Close') -> pd.Series:
        # 추가적인 method 구현
        ret = self._get_returns(attr)
        skewness = ret.skew()
        return skewness

    def get_kurtosis(self, attr = 'Close') -> pd.Series:
        # 추가적인 method 구현
        ret = self._get_returns(attr)
        kurtosis = ret.kurtosis()
        return kurtosis

    def get_beta(self, market_returns: pd.Series, risk_free : float, attr: str = 'Close', annualized: bool = True) -> float:
        """
        문제 3번
        종목의 베타 값을 계산하는 method

        Parameters:
        - market_returns (pd.Series): 시장 지수 수익률 데이터 (예: S&P 500)
        - attr (str): 사용하려는 가격 데이터 속성 (기본값: 'Close')
        - annualized (bool): 연환산 여부 (기본값: True)

        Returns:
        - float: 종목의 베타 값
        """
        stock_returns = self._get_returns(attr)

        stock_returns, market_returns = stock_returns.align(market_returns, join='inner')

        cov_matrix = np.cov(stock_returns, market_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]

        if annualized:
            beta *= 252

        return beta
