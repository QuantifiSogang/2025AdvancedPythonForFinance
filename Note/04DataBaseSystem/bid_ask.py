import ccxt.pro as ccxtpro
import asyncio
import datetime
import pprint

async def real_time_bid_ask():
    exchange = ccxtpro.upbit()
    while True:
        orderbook = await exchange.watch_order_book(symbol="BTC/KRW")
        pprint.pprint(orderbook)
        await exchange.close()

if __name__ == '__main__':
    asyncio.run(real_time_bid_ask())