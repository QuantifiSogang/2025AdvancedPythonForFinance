import ccxt.pro as ccxtpro
import asyncio
import datetime
import pprint

async def watch_ticker(exchange, symbol):
    """
    ticker를 실시간으로 관측하는 함수
    :param exchange: 거래소를 지정한다.
    :param symbol: 관찰할 가상화폐의 심볼을 지정한다.
    """
    while True:
        ticker = await exchange.watch_ticker(symbol)
        pprint.pprint(ticker)

async def watch_order_book(exchange, symbol):
    while True:
        order_book = await exchange.watch_order_book(symbol)
        pprint.pprint(order_book)

async def web_socket():
    symbol = "BTC/KRW"
    exchange = ccxtpro.upbit()

    coros = [ # 다중 웹소켓 처리
        watch_ticker(exchange, symbol),
        watch_order_book(exchange, symbol),
    ]

    await asyncio.gather(*coros)
    await exchange.close()

if __name__ == '__main__':
    asyncio.run(web_socket())