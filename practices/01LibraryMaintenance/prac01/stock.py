import numpy as np
import pandas as pd
from .data import DataLoader
from .stats import StockStats

class Stock(DataLoader, StockStats):
    def __init__(self):
        super().__init__()
