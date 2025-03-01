import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from datetime import datetime, timedelta
from utils import *

from agents import technical_agent, fundamental_agent, sentiment_agent, forecaster_agent, manager_agent
from agents import search_from_google

with open('config/api.key') as file :
    lines = file.readlines()
    api_key = lines[0].strip()
    serp_api_key = lines[1].strip()
    langsmith_api_key = lines[2].strip()

st.set_page_config(page_title = "Portfolio Management System")
st.header('Portfolio Management System')

st.sidebar.header("종목 선택")
ticker = st.sidebar.text_input("티커 입력 (예: AAPL, TSLA, MSFT)", value = "AAPL")

def get_stock_data(ticker):
    """
    yahoo finance로부터 주식 데이터를 다운받는 함수입니다.
    :param ticker:
    :return:
    """
    data = yf.download(
        ticker,
        auto_adjust = True,
        multi_level_index = False,
        interval = '1d',
        progress = False
    )
    return data

data = get_stock_data(ticker)

data['20MA'] = data['Close'].rolling(window=20).mean()
data['Upper_BB'] = data['20MA'] + (2 * data['Close'].rolling(window=20).std())
data['Lower_BB'] = data['20MA'] - (2 * data['Close'].rolling(window=20).std())

recent_date = datetime.today() + timedelta(days = 30)
start_date = datetime.today() - timedelta(days = 180)

y_range_max = data.loc[start_date:, 'High'].max() * 1.1
y_range_min = data.loc[start_date:, 'Low'].min() * 0.9

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=f'{ticker} Price',
        increasing_line_color="green",
        decreasing_line_color="red"
    )
)

fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data['20MA'],
        mode='lines',
        name='20MA',
        line=dict(color='blue')
    )
)

fig.add_trace(go.Scatter(x=data.index, y=data['Upper_BB'], mode='lines', name='Upper Bollinger Band', line=dict(color='gray', dash='dot')))
fig.add_trace(go.Scatter(x=data.index, y=data['Lower_BB'], mode='lines', name='Lower Bollinger Band', line=dict(color='gray', dash='dot')))

fig.update_layout(
    title=f"{ticker} stock chart",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=True,  # 전체 데이터 확인 가능하도록 유지
    xaxis=dict(
        range=[start_date, recent_date]  # 초기 범위 설정
    ),
    yaxis=dict(
        range=[y_range_max, y_range_min],
    )
)

st.title(f"{ticker} 종목 분석")
st.write("이 페이지에서는 기술적 분석, 펀더멘털 분석, 감성 분석, 시장 예측을 종합하여 최종 투자 결정을 제공합니다.")

st.plotly_chart(fig, use_container_width=True)

st.subheader("AI Agent 분석 결과")

momentum = MomentumIndicator(data, resample='1d')
contrarian = ContrarianIndicator(data, resample='1d')

# Simple moving average
sma_5 = momentum.simple_moving_average(5)
sma_20 = momentum.simple_moving_average(20)
sma_golden_cross = ((sma_5 >= sma_20) & (sma_5.shift(1) < sma_20.shift(1))).astype(int)
sma_dead_cross = ((sma_5 < sma_20) & (sma_5.shift(1) >= sma_20.shift(1))).astype(int)

# True Strength Index
tsi_25_13 = momentum.true_strength_index(13, 25)
tsi_up = ((tsi_25_13 >= 0) & (tsi_25_13.shift(1) < 0)).astype(int)
tsi_down = ((tsi_25_13 < 0) & (tsi_25_13.shift(1) >= 0)).astype(int)

# ADL
adl14 = momentum.average_daily_range(14)
adl50 = momentum.average_daily_range(50)
adl_up = ((adl14 >= adl50) & (adl14.shift(1) < adl50.shift(1))).astype(int)
adl_down = ((adl14 < adl50) & (adl14.shift(1) >= adl50.shift(1))).astype(int)

# ADR
adr20 = momentum.average_daily_range(20)
adr50 = momentum.average_daily_range(50)
adr_up = ((adr20 >= adr50) & (adr20.shift(1) < adr50.shift(1))).astype(int)
adr_down = ((adr20 < adr50) & (adr20.shift(1) >= adr50.shift(1))).astype(int)

# Aroon
aroon = momentum.aroon_indicator(14)
aroon_up = ((aroon['Aroon(14) up'] >= aroon['Aroon(14) down']) & (
            aroon['Aroon(14) up'].shift(1) < aroon['Aroon(14) down'].shift(1))).astype(int)
aroon_down = ((aroon['Aroon(14) up'] < aroon['Aroon(14) down']) & (
        aroon['Aroon(14) up'].shift(1) >= aroon['Aroon(14) down'].shift(1))).astype(int)

# RSI
rsi = contrarian.relative_strength_index(20)
rsi_up = ((rsi <= 70) & (rsi.shift(1) > 70)).astype(int)
rsi_down = ((rsi >= 30) & (rsi.shift(1) < 30)).astype(int)

# BB
bb = contrarian.bollinger_band(2, 20)
bb_up = ((data['Close'] < bb['BB_UP(2)']) & (data['Close'].shift(1) >= bb['BB_UP(2)'].shift(1))).astype(int)
bb_down = ((data['Close'] >= bb['BB_DOWN(2)']) & (data['Close'].shift(1) < bb['BB_DOWN(2)'].shift(1))).astype(int)

# DEMARKER
demark = contrarian.demarker_indicator(20)
demark_up = ((demark < 0.7) & (demark.shift(1) >= 0.7)).astype(int)
demark_down = ((demark >= 0.3) & (demark.shift(1) < 0.3)).astype(int)

# psycological line
psycological_line = contrarian.psycological_line(20)
pl_up = ((psycological_line < 0.7) & (psycological_line.shift(1) >= 0.7)).astype(int)
pl_down = ((psycological_line >= 0.3) & (psycological_line.shift(1) < 0.3)).astype(int)

#concatenate
technical_signals = pd.concat(
    [sma_golden_cross, sma_dead_cross,tsi_up, tsi_down, adl_up, adl_down, adr_up, adr_down, aroon_up, aroon_down, rsi_up, rsi_down,bb_up, bb_down, demark_up, demark_down,
     pl_up, pl_down], axis = 1
)
technical_signals.columns = [
    'SMA(5,20) Golden', 'SMA(5,20) Dead',
    'TSI(13,25) Golden', 'TSI(13,25) Dead',
    'ADL(14,50) Golden', 'ADL(14,50) Dead',
    'ADR(20,50) Golden', 'ADR(20,50) Dead',
    'Aroon(14) Golden', 'Aroon(14) Dead',
    'RSI(30,70) up','RSI(30,70) down',
    'Bollinger Band(2, 20) up','Bollinger Band(2, 20) down',
    'Demark(30,70) up','Demark(30,70) down',
    'Psy (30, 70) up','Psy (30, 70) down'
]

st.write("기술적 분석 수행 중...")
tech_result = technical_agent(technical_signals.tail(40))
st.success(f"**기술적 분석:** {tech_result}")

# 펀더멘털 분석
st.write("Fundamental 분석 수행 중...")
fund_result = fundamental_agent(ticker)
st.success(f"**펀더멘털 분석:** {fund_result}")

st.write("Market Sentiment 분석 수행 중...")
news_data = search_from_google(ticker, page_nums=4)
sent_result = sentiment_agent(news_data)
st.success(f"**감성 분석:** {sent_result}")

# 시장 예측
st.write("Market Forecast 수행 중...")
fore_result = forecaster_agent(ticker)
st.success(f"**시장 예측:** {fore_result}")

# 최종 투자 결정
st.subheader("최종 투자 결정")
final_decision = manager_agent(tech_result, fund_result, sent_result, fore_result)
st.success(f"**최종 결론:** {final_decision}")

st.subheader("LLM Based Stock Chatbot System")

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    openai_api_key=api_key
)

# save session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 📌 사전 분석 정보 (미리 전달할 정보)
pre_analysis_info = f"""
이전 분석 결과 :

user가 선택한 종목 : {ticker}

AI Agent가 판단한 현재 종목의 신호
Technical Agent : {tech_result}
Fundamental Agent : {fund_result}
Sentiment Agent : {sent_result}
Forecaster Agent : {fore_result}

AI Portfolio manager가 최종적으로 판단한 결과
Manager Agent : {final_decision}

User의 투자성향
- User는 위험회피 성향이 약하고, 성장주 중심의 투자를 원하고 있다.
"""

st.subheader("Chatbot")

# ✅ LLM 실행 전에 미리 정보 제공
if "pre_analysis_included" not in st.session_state:
    st.session_state.chat_history.append(HumanMessage(content="분석 정보: " + pre_analysis_info))
    st.session_state.pre_analysis_included = True  # 중복 방지

# ✅ 특정 부분부터 실행할지 여부를 세션 상태로 관리
if "llm_ready" not in st.session_state:
    st.session_state.llm_ready = False  # 처음에는 실행 안 함

# 🔘 사용자가 실행 버튼을 눌렀을 때만 실행
if st.button("LLM 실행 시작"):
    st.session_state.llm_ready = True

user_input = st.text_input("질문을 입력하세요 (예: '이 종목의 매매 전략은?', '기술적 분석을 어떻게 활용할 수 있을까?')")

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=api_key,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

if user_input and st.session_state.llm_ready:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        llm_response = llm(st.session_state.chat_history)
        message_placeholder.markdown(llm_response.content)

    st.session_state.chat_history.append(AIMessage(content=llm_response.content))

# 채팅 기록 표시
st.subheader("Chat History")
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        st.chat_message("user").markdown(f"**User:** {message.content}")
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").markdown(f"**AI:** {message.content}")