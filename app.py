import streamlit as st
import yfinance as yf
import statsmodels.api as sm
import pandas as pd

st.title("📊 CAPM 투자 분석기")
st.write("S&P 500 대비 종목의 저평가 여부를 분석합니다.")

# 사용자 입력
ticker_symbol = st.text_input("분석할 티커를 입력하세요", "AAPL").upper()

if st.button("분석 시작"):
    with st.spinner('데이터를 가져오는 중...'):
        # 데이터 설정
        market_ticker = "^GSPC"
        rf_ticker = "^TNX"
        
        # 데이터 다운로드 및 처리
        data = yf.download([ticker_symbol, market_ticker, rf_ticker], period="5y")['Close']
        data = data.ffill().dropna()
        returns = data[[ticker_symbol, market_ticker]].pct_change().dropna()
        
        # 변수 계산
        current_rf = data[rf_ticker].iloc[-1] / 100
        X = sm.add_constant(returns[market_ticker])
        Y = returns[ticker_symbol]
        model = sm.OLS(Y, X).fit()
        beta = model.params[market_ticker]
        
        # 수익률 분석
        actual_annual = returns[ticker_symbol].mean() * 252
        market_annual = returns[market_ticker].mean() * 252
        capm_expected = current_rf + beta * (market_annual - current_rf)
        alpha = actual_annual - capm_expected

        # 대시보드 결과 표시
        col1, col2, col3 = st.columns(3)
        col1.metric("Beta (β)", f"{beta:.2f}")
        col2.metric("CAPM 기대수익률", f"{capm_expected:.2%}")
        col3.metric("실제 과거 수익률", f"{actual_annual:.2%}")

        if alpha > 0.01:
            st.success(f"✅ 의견: **매수 권장 (저평가)** / 알파: {alpha:.2%}")
        elif alpha < -0.01:
            st.error(f"🚨 의견: **매도 권장 (고평가)** / 알파: {alpha:.2%}")
        else:
            st.info(f"⚖️ 의견: **보유 (적정가)** / 알파: {alpha:.2%}")
            
        # 수익률 차트 추가
        st.line_chart((1 + returns).cumprod())
