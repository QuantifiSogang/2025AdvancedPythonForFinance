import ccxt.pro as ccxtpro
import asyncio
import datetime

async def real_time_execution_strength():
    exchange = ccxtpro.upbit()
    while True:
        ticker = await exchange.watch_ticker(symbol = "BTC/KRW")

        timestamp = ticker['timestamp']
        now = datetime.datetime.fromtimestamp(timestamp/1000)
        last = ticker['last']
        acc_ask_volume = ticker['info']['acc_ask_volume']
        acc_bid_volume = ticker['info']['acc_bid_volume']
        volume_power = acc_bid_volume / acc_ask_volume * 100

        print(f"time: {now} price: {last: ,} execution_strength: {volume_power:.2f}")

if __name__ == '__main__':
    asyncio.run(real_time_execution_strength())