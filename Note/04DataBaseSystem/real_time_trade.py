import pprint
import ccxt.pro as ccxtpro
import asyncio
import datetime

async def real_time_trade():
    exchange = ccxtpro.upbit()
    while True:
        trade = await exchange.watch_trades(symbol="BTC/KRW")
        pprint.pprint(trade)

if __name__ == '__main__':
    asyncio.run(real_time_trade())