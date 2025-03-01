import sqlite3
import ccxt.async_support as ccxt
import asyncio
import datetime

DB_NAME = "price.db"
exchange = ccxt.upbit()

async def fetch_and_store_btc_price():
    while True:
        try:
            ticker = await exchange.fetch_ticker("BTC/KRW")  # BTC 가격 가져오기
            price = ticker['last']
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # SQLite3에 저장
            conn = sqlite3.connect(DB_NAME)  # connect
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO BTC_price (date, price) VALUES (?, ?)", (timestamp, price))
            conn.commit()
            conn.close()

            print(f"[{timestamp}] BTC Price: {price} KRW")

        except Exception as e:
            print(f"error : {e}")

        await asyncio.sleep(1)  # 1초 간격으로 요청 (실시간 감시)


if __name__ == '__main__':
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        create table if not exists btc_price (
            date text primary key,
            price real not null
        )
    """)
    conn.commit()
    conn.close()

    asyncio.run(fetch_and_store_btc_price())