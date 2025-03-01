import ccxt.pro as ccxtpro
import asyncio
import datetime
import pprint

async def loop(exchange, symbol):
    while True:
        trade = await exchange.watch_trades(symbol)
        pprint.pprint(trade)

async def real_time_multi_asset():
    exchange = ccxtpro.upbit()
    symbols = ['BTC/KRW', 'ETH/KRW', 'XRP/KRW']

    coros = [loop(exchange, symbol) for symbol in symbols]
    await asyncio.gather(*coros)
    await exchange.close()

if __name__ == '__main__':
    asyncio.run(real_time_multi_asset())