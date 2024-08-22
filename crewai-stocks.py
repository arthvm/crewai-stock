import json
import os
from datetime import datetime

import yfinance as yf

from crewai import Agent, Task, Crew
from crewai.process import Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

def fetch_stock_price(ticket):
    stock_prices = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock_prices

yahoo_finance_tool = Tool(
    name= "Yahoo Finance Tool",
    description= "Fetches stock prices for {ticket} from the last year with Yahoo Finance API",
    func= lambda ticket: fetch_stock_price(ticket)
)

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-3.5-turbo")

stockPriceAnalyst = Agent(
    role="Senior stock price Analyst",
    goal="Find {ticket} stock price and analyse trends",
    backstory=
    """
    You're a highly experienced stock price analyst and are capable of analysing an specific stock and give a prediction about it future prices
    """,
    verbose=True,
    llm = llm,
    max_iter = 5,
    memory= True,
    allow_delegation= False,
    tools=[yahoo_finance_tool]
)

getStockPrice = Task(
    description="Analyse the stock {ticket} price history and create a trend analyses of up, down or sideways",
    expected_output=
    """
    Specify the current trend of the stock price - up, down or sideways.
    eg. stock='AAPL, price UP'
    """,
    agent= stockPriceAnalyst
)

search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)

newsAnalyst = Agent(
    role="Senior Stock News Analyst",
    goal=
    """
    Create a short summary of the relevant market news related to the {ticket} stock.
    Specify the current tren - up, down or sideways with the news content.
    For each requested stock, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.
    """,
    backstory=
    """
    You're a highly experienced stock news analyst and are capable of analyzing trends and news, having deep understanding of tradicional markets and human psychology.
    You understand news, their titles and information, but look at those with a health dose of skeptism. You also take in consideration the source of the news.
    """,
    verbose=True,
    llm = llm,
    max_iter = 10,
    memory= True,
    allow_delegation= False,
    tools=[search_tool]
)

getNews = Task(
    description=
    f"""
    Take the stock and always include BTC to it (if not requested).
    Use the search tool and seach each one individually.

    The current date is {datetime.now()}.

    Compose the results into a helpfull report
    """,
    expected_output=
    """
    A one sentence summary of the overall market for each requested asset.
    Include a fear/greed score for each asset based on the news.
    Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
    """,
    agent= newsAnalyst
)

stockAnalystReporter = Agent(
    role="Senior Stock Analyst Reporter",
    goal=
    """
    Analyze the trends price and news to write an insightfull compelling and informative three paragraph long newsletter based on the stock report and price trend.
    """,
    backstory=
    """
    You're a widely accpeted and highly experienced stock analyst in the market. 
    You understand complex economical and human nature concepts and are able to create compelling stories and reports that resonates with wider audiences while still containing those concepts and technical thinking.
    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses. - to form your report.
    You're able to hold multiple opnions when analyzing an asset.
    """,
    verbose=True,
    llm = llm,
    max_iter = 5,
    memory= True,
    allow_delegation= True,
)


writeAnalyses = Task(
    description =
    """
    Use the stock price history and trens as well as the stock news report to create an analyses and write the newsletter about the {ticket} stock.
    It should be brief and highlight the most important points.
    Focus on the stock price trend, news and fear/greed score.
    What are the near future considerations?
    Include the previous analyses of stock trend and news summary.
    """,
    expected_output =
    """
    An eloquent 3 paragraphs newsletter formatted as markdown in an easy and redable manner.
    It should contain:

    - 3 bullets executive summary
    - Introduction - set the overall picture
    - Main - provides the core of the analyses including the news summary and fear/greed scores
    - Summary - key facts and concrete future trend prediction (up, down or sideways).
    """,
    agent = stockAnalystReporter,
    context = [getStockPrice, getNews]
)

crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockAnalystReporter],
    tasks=[getStockPrice, getNews, writeAnalyses],
    verbose=True,
    process= Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_inter=15
)

with st.sidebar:
    st.header("Enter the stock ticket to research: ")

    with st.form(key="research_form"):
        topic = st.text_input("Enter the ticket")
        submit_button = st.form_submit_button(label= "Run Research")

if submit_button:
    if not topic:
        st.error("Please fill the stock ticket field")
    else:
        results = crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of your research: ")
        results["final_output"]


