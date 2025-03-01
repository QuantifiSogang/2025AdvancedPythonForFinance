import numpy as np
import pandas as pd
import openai
from openai import OpenAI
import os
import yfinance as yf
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence
from bs4 import BeautifulSoup
import requests
import urllib
import re
from newspaper import Article
from selenium import webdriver

from utils import *
import warnings
warnings.filterwarnings('ignore')

with open('config/api.key') as file :
    lines = file.readlines()
    api_key = lines[0].strip()
    serp_api_key = lines[1].strip()
    langsmith_api_key = lines[2].strip()

openai.api_key = api_key


def get_fundamental_data(ticker):
    """
    Yahoo Finance API를 사용하여 해당 종목의 기본적 분석(Fundamental Analysis) 데이터를 가져오는 함수
    :param ticker: 종목 코드 (예: "AAPL", "TSLA")
    :return: 해당 종목의 재무 지표가 포함된 pd.DataFrame
    """
    stock = yf.Ticker(ticker)

    # 종목의 주요 재무 지표 가져오기
    try:
        info = stock.info
        data = {
            "Ticker": ticker,
            "Company": info.get("longName", "N/A"),  # 회사 이름
            "Sector": info.get("sector", "N/A"),  # 섹터 이름, GICS기준
            "Market Cap": info.get("marketCap", "N/A"),  # 시가총액
            "PER": info.get("trailingPE", "N/A"),  # Price to Earning ratio
            "Forward PER": info.get("forwardPE", "N/A"),  # forward PER
            "PBR": info.get("priceToBook", "N/A"),  # Price to Book ratio
            "EV/EBITDA": info.get("enterpriseToEbitda", "N/A"),  # EV/EBITDA
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),  # 52주 최고가
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),  # 52주 최저가
            "Dividend Yield": info.get("dividendYield", "N/A"),  # 배당수익률
            "Recommendation": info.get("recommendationKey", "N/A"),
        }
        return pd.DataFrame([data])

    except Exception as e:
        print(f"data download error: {e}")
        return None

def search_from_google(asset: str, page_nums: int) -> pd.DataFrame:
    '''
    google news로부터 검색을 한 뒤, selenium을 통해 뉴스 데이터들을 가져옵니다.
    :param asset: 검색할 자산
    :param page_nums: 뉴스를 검색할 총 페이지의 수
    :return: news data가 들어있는 DataFrame
    '''
    keyword = f'{asset} buying reason'
    news_df = pd.DataFrame()

    # selenium headless mode
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome(options=options)

    for page_num in range(0, page_nums - 1):
        # google news crawling
        url = f'https://www.google.com/search?q={keyword}&sca_esv=1814fa2a4600643d&tbas=0&tbs=qdr:m&tbm=nws&ei=rE3pZeLxNeHX1e8PpdOcMA&start={page_num}&sa=N&ved=2ahUKEwji9-zrsuGEAxXha_UHHaUpBwYQ8tMDegQIBBAE&biw=2560&bih=1313&dpr=1'
        req = requests.get(url)
        content = req.content
        soup = BeautifulSoup(content, 'html.parser')

        # last page check
        if soup.select('div.BNeawe.vvjwJb') == []: break

        title_list = [t.text for t in soup.select('div.BNeawe.vvjwJb')]  # title
        url_list = []

        # url
        for u in soup.select('a'):
            for t in title_list:
                if t in u.text:
                    temp_url = urllib.parse.unquote(u['href'])
                    temp_url = re.findall('http\S+&sa', temp_url)[0][:-3]
                    url_list.append(temp_url)

        # article
        for ind, news_url in enumerate(url_list):
            try:
                article = Article(url=news_url)
                article.download()
                article.parse()
                news_article = article.text
            except:  # ssl error
                driver.get(news_url)
                article.download(input_html=driver.page_source)
                article.parse()
                news_article = article.text

            news_df = pd.concat([news_df, pd.DataFrame([[title_list[ind], news_article, news_url]])])

        news_df[0] = news_df[0].apply(lambda x: re.sub('\s+', ' ', x))
        news_df = news_df.reset_index(drop=True)

    news_df.columns = ['Title', 'Contents', 'URL']

    return news_df

def technical_agent(signals):
    """
    OpenAI GPT API를 사용하여 0과 1로 구성된 기술적 지표 신호를 해석하고
    시장 상황을 Bullish, Bearish, Neutral 중 하나로 판단하여 매매 조언을 제공하는 함수입니다.
    :param signals: signal이 기록된 pd.DataFrame형태의
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        openai_api_key=api_key
    )

    prompt = """
    다음은 최근 금융 시장의 기술적 지표 신호(0 또는 1) 기록입니다.
    1은 해당 신호가 발생했음을 의미하고, 0은 신호가 발생하지 않았음을 의미합니다.

    최근 40개 데이터:
    {signals}

    판단 기준:
    - 매수 신호가 다수 발생하면 Bullish (강세)로 판단.
    - 매도 신호가 다수 발생하면 Bearish (약세)로 판단.
    - 신호가 혼재되어 있거나 강한 추세가 없으면 Neutral (중립)로 판단.
    - 현재 상태에 따라 투자자에게 Buy(매수), Sell(매도), Hold(관망) 조언을 제공.

    시장 상황을 분석하고, 현재 상태를 Bullish, Bearish, Neutral 중 하나로 판단한 후
    투자자에게 적절한 조언을 한 줄로 제공하세요.
    """

    # OpenAI GPT-4 호출
    prompt = PromptTemplate(
        template=prompt,
        input_variables=["signals"]
    )

    # RunnableSequence를 사용하여 체인 구성
    chain = RunnableSequence(
        {
            "signals": RunnablePassthrough()
        }
        | prompt
        | llm
        | (lambda x: x.content)
    )

    latest_signals = signals.to_string(index=False)

    # 시장 분석 실행
    result = chain.invoke({"signals": latest_signals})

    return result

def fundamental_agent(ticker):
    """
    OpenAI GPT API를 사용하여 Yahoo Finance에서 불러온 재무 데이터를 기반으로
    주식이 고평가(Bearish)인지, 저평가(Bullish)인지 분석하고, 매매 조언을 제공하는 함수.

    :param ticker: 주식 종목 코드 (예: "AAPL", "GOOGL")
    :return: LLM 분석 결과 (Bullish, Bearish, Neutral 및 매매 조언)
    """
    # Yahoo Finance에서 재무 데이터 가져오기
    fund_data = get_fundamental_data(ticker)

    if fund_data is None:
        return {"signal": "Error", "message": "재무 데이터 불러오기 실패"}

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        openai_api_key=api_key
    )

    # LLM 프롬프트 템플릿 설정
    prompt = """
    다음은 {ticker}({company})의 최신 펀더멘털(재무) 데이터입니다:
    {fundamental_data}

    판단 기준:
    - PER, PBR, EV/EBITDA가 업종 평균보다 낮으면 저평가(Bullish)로 판단.
    - 반대로 PER, PBR, EV/EBITDA가 높으면 고평가(Bearish)로 판단.
    - 업종 평균과 비슷한 수준이면 중립(Neutral)로 판단.
    - 배당률이 높거나 성장성이 좋은 경우도 고려하여 판단.
    - 현재 상태에 따라 투자자에게 Buy(매수), Sell(매도), Hold(관망) 조언을 제공.

    재무 상황을 분석하고, 현재 상태를 Bullish, Bearish, Neutral 중 하나로 판단한 후
    투자자에게 적절한 조언을 한 줄로 제공하세요.
    """

    prompt = PromptTemplate(
        template=prompt,
        input_variables=["ticker", "company", "fundamental_data"]
    )

    # RunnableSequence를 사용하여 체인 구성
    chain = RunnableSequence(
        {
            "ticker": RunnablePassthrough(),
            "company": RunnablePassthrough(),
            "fundamental_data": RunnablePassthrough()
        }
        | prompt
        | llm
        | (lambda x: x.content)
    )

    # LLM 실행
    result = chain.invoke({
        "ticker": fund_data["Ticker"][0],
        "company": fund_data["Company"][0],
        "fundamental_data": fund_data.to_string(index=False)
    })

    return result

def sentiment_agent(news_df):
    """
    OpenAI GPT API를 사용하여 뉴스 기사의 제목과 본문을 분석하고,
    해당 기업에 대한 뉴스가 긍정적인지(Bullish), 부정적인지(Bearish), 중립적인지(Neutral) 판단하여
    투자자에게 조언을 제공하는 함수.

    :param news_df: 뉴스 데이터프레임 (컬럼: ['Title', 'Contents', 'URL'])
    :return: LLM 분석 결과 (Bullish, Bearish, Neutral 및 매매 조언)
    """
    if news_df is None or news_df.empty:
        return {"signal": "Error", "message": "뉴스 데이터가 없습니다!"}

    news_df["Full_Text"] = news_df["Title"] + " " + news_df["Contents"]
    latest_news = news_df["Full_Text"].tolist()[:10]  # 최신 뉴스만을 사용

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        openai_api_key=api_key
    )

    # LLM 프롬프트 템플릿 설정
    template = """
    다음은 기업과 관련된 최신 뉴스 기사들입니다.  
    각 뉴스는 해당 기업에 대한 투자 심리를 반영할 수 있으며, 감성 분석을 통해 시장 분위기를 평가하세요.

    최근 5개 뉴스 기사:
    {news}

    판단 기준:
    - 긍정적인 뉴스 기사가 다수라면 Bullish (강세)로 판단.
    - 부정적인 뉴스 기사가 다수라면 Bearish (약세)로 판단.
    - 긍정과 부정이 혼재되어 있거나, 특별한 정보가 없다면 Neutral (중립)로 판단.
    - 시장 심리를 반영하여 투자자에게 Buy(매수), Sell(매도), Hold(관망) 조언을 제공.

    현재 상태를 Bullish, Bearish, Neutral 중 하나로 판단한 후
    투자자에게 적절한 조언을 한 줄로 제공하세요.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["news"]
    )

    # RunnableSequence를 사용하여 체인 구성
    chain = RunnableSequence(
        {
            "news": RunnablePassthrough()
        }
        | prompt
        | llm
        | (lambda x: x.content)
    )

    # LLM 실행
    result = chain.invoke({
        "news": "\n\n".join(latest_news)  # 뉴스 본문 결합
    })

    return result

def forecaster_agent(ticker):
    """
    OpenAI GPT API를 사용하여 ARMA(1,1) 모형을 기반으로 시장 예측을 수행하고,
    예측된 값이 상승이면 Bullish, 하락이면 Bearish, 중립이면 Neutral로 분류하여 투자 조언을 제공하는 함수.

    :param ticker: 주식 종목 코드 (예: "AAPL", "GOOGL")
    :return: LLM 분석 결과 (Bullish, Bearish, Neutral 및 매매 조언)
    """
    # Yahoo Finance에서 주가 데이터 가져오기
    stock_returns = get_stock_data(ticker)

    if stock_returns is None:
        return {"signal": "Error", "message": "주가 데이터 불러오기 실패"}

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        openai_api_key=api_key
    )

    # ARMA(1,1) 모형 추정 및 1-step ahead 예측
    forecast = estimate_arma_model(stock_returns)

    if forecast is None:
        return {"signal": "Error", "message": "❌ ARMA 모델 예측 실패!"}

    # LLM 프롬프트 템플릿 설정
    template = """
    다음은 {ticker}의 최근 1년간 종가 데이터를 기반으로 한 ARMA(1,1) 모형의 예측 결과입니다.

    1-step ahead 예측값: {forecast}

    판단 기준:
    - 예측값이 양수(>0)이면 Bullish (강세)로 판단.
    - 예측값이 음수(<0)이면 Bearish (약세)로 판단.
    - 예측값이 0에 가까우면 Neutral (중립)로 판단.
    - 현재 상태에 따라 투자자에게 Buy(매수), Sell(매도), Hold(관망) 조언을 제공.

    현재 상태를 Bullish, Bearish, Neutral 중 하나로 판단한 후
    투자자에게 적절한 조언을 한 줄로 제공하세요.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["ticker", "forecast"]
    )

    # RunnableSequence를 사용하여 체인 구성
    chain = RunnableSequence(
        {
            "ticker": RunnablePassthrough(),
            "forecast": RunnablePassthrough()
        }
        | prompt
        | llm
        | (lambda x: x.content)
    )

    # LLM 실행
    result = chain.invoke({
        "ticker": ticker,
        "forecast": forecast
    })

    return result

def manager_agent(tech_result, fund_result, sent_result, fore_result):
    """
    OpenAI GPT API를 사용하여 4개의 개별 분석 결과를 종합하고
    최종 투자 결정을 내려주는 함수.

    :param tech_result: 기술적 분석 결과
    :param fund_result: 펀더멘털 분석 결과
    :param sent_result: 감성 분석 결과
    :param fore_result: 시장 예측 결과
    :return: 최종 투자 판단 (Bullish, Bearish, Neutral 및 매매 조언)
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
        openai_api_key=api_key
    )

    # LLM 프롬프트 템플릿 설정
    template = """
    다음은 특정 주식에 대한 4가지 분석 결과입니다.
    각 분석은 해당 주식의 시장 상황을 반영하며, 이를 종합하여 최종 투자 결정을 내려야 합니다.

    - 기술적 분석 (Technical Analysis): {tech}
    - 펀더멘털 분석 (Fundamental Analysis): {fund}
    - 감성 분석 (Sentiment Analysis): {sent}
    - 시장 예측 (Forecasting Analysis): {fore}

    **판단 기준:**
    - Bullish가 3개 이상이면 최종적으로 Bullish (강세)로 판단.
    - Bearish가 3개 이상이면 최종적으로 Bearish (약세)로 판단.
    - 신호가 혼재되어 있거나 Neutral이 많은 경우 최종적으로 Neutral (중립)로 판단.
    - 각 분석 결과의 신뢰도를 고려하여 투자자에게 Buy(매수), Sell(매도), Hold(관망) 조언을 제공.

    현재 상태를 Bullish, Bearish, Neutral 중 하나로 판단한 후, 최종 판단한 상태의 확률(신뢰도)를 퍼센테이지로 제공한 뒤,
    투자자에게 적절한 조언을 한 줄로 제공하세요.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["tech", "fund", "sent", "fore"]
    )

    # RunnableSequence를 사용하여 체인 구성
    chain = RunnableSequence(
        {
            "tech": RunnablePassthrough(),
            "fund": RunnablePassthrough(),
            "sent": RunnablePassthrough(),
            "fore": RunnablePassthrough()
        }
        | prompt
        | llm
        | (lambda x: x.content)
    )

    # LLM 실행
    result = chain.invoke({
        "tech": tech_result,
        "fund": fund_result,
        "sent": sent_result,
        "fore": fore_result
    })

    return result