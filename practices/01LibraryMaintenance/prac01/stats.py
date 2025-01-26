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
    """
    이 부분에 인터페이스에서 구상된 method를 구현한 뒤, 추가적인 method를 더 추가해 주세요.
    """
    pass