import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import itertools
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

#--------------------------------------------
# Page Config and Custom CSS
#--------------------------------------------
st.set_page_config(layout="wide", page_title="Cointegrated ETFs Strategy")

# Add custom CSS for modern look
st.markdown("""
<style>
    body {
        background: #f9f9f9;
        font-family: 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    }
    .stApp {
        background: #f9f9f9;
    }
    .block-container {
        padding: 2rem 3rem;
    }
    header, footer {
        visibility: hidden;
    }
    .title {
        font-size: 2.5em;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5em;
    }
    .subtitle {
        font-size: 1.2em;
        color: #555;
        margin-bottom: 1.5em;
    }
    .section-header {
        font-size: 1.5em;
        color: #222;
        border-bottom: 2px solid #eaeaea;
        padding-bottom: 0.2em;
        margin-bottom: 1em;
    }
    .sidebar-section {
        border-bottom: 1px solid #ddd;
        padding-bottom: 1em;
        margin-bottom: 1em;
    }
</style>
""", unsafe_allow_html=True)


#--------------------------------------------
# Helper Functions
#--------------------------------------------
@st.cache_data
def load_etf_classification(csv_path='etf_classification.csv'):
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=['Ticker','Classification','Leverage'])
        df.to_csv(csv_path, index=False)
        return df
    return pd.read_csv(csv_path)

def save_etf_classification(df, csv_path='etf_classification.csv'):
    df.to_csv(csv_path, index=False)

@st.cache_data
def fetch_price_data(tickers, start_date, end_date):
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data.dropna(how='all', axis=1, inplace=True)
    data.dropna(inplace=True)
    return data

def estimate_B_and_test_coint(df, stock1, stock2, significance=0.05):
    if stock1 not in df.columns or stock2 not in df.columns:
        return None, None, None
    common_idx = df[stock1].dropna().index.intersection(df[stock2].dropna().index)
    if len(common_idx) < 100:
        return None, None, None
    y = df.loc[common_idx, stock1]
    x = df.loc[common_idx, stock2]
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    B = model.params[1]
    spread = y - B*x
    spread.dropna(inplace=True)
    adf_p = adfuller(spread)[1]
    if adf_p < significance:
        return B, spread.mean(), spread.std()
    else:
        return None, None, None

def calculate_emrt(spread, C=2.0):
    theta = spread.mean()
    s = spread.std()
    is_max = (spread.shift(1)<spread)&(spread.shift(-1)<spread)
    is_min = (spread.shift(1)>spread)&(spread.shift(-1)>spread)
    important_max = is_max & ((spread - theta) >= C*s)
    important_min = is_min & ((theta - spread) >= C*s)
    extremes = spread[important_max | important_min].sort_index()
    tau = list(extremes.index)
    if len(tau)<2:
        return np.nan
    intervals = [(tau[i]-tau[i-1]).days for i in range(1, len(tau))]
    return np.mean(intervals) if intervals else np.nan

def construct_spread(df, s1, s2, B):
    if s1 not in df.columns or s2 not in df.columns:
        return pd.Series(dtype=float)
    return df[s1] - B*df[s2]

def backtest_strategy(price_data, s1, s2, B, transaction_cost=0.001, C=2.0, window=60):
    spread = construct_spread(price_data, s1, s2, B)
    df = spread.to_frame('Spread')
    df['Theta'] = df['Spread'].rolling(window).mean()
    df['Sigma'] = df['Spread'].rolling(window).std()
    df['Is_Max'] = (df['Spread'].shift(1)<df['Spread'])&(df['Spread'].shift(-1)<df['Spread'])
    df['Is_Min'] = (df['Spread'].shift(1)>df['Spread'])&(df['Spread'].shift(-1)>df['Spread'])
    df['Important_Max'] = df['Is_Max'] & ((df['Spread']-df['Theta'])>=C*df['Sigma'])
    df['Important_Min'] = df['Is_Min'] & ((df['Theta']-df['Spread'])>=C*df['Sigma'])

    trades = []
    open_trades = []

    for current_date, row in df.iterrows():
        if pd.isna(row['Theta']) or pd.isna(row['Sigma']):
            continue
        if row['Important_Max']:
            entry_price_a = price_data.loc[current_date, s1]
            entry_price_b = price_data.loc[current_date, s2]
            q_a = -1.0
            q_b = B
            trades.append({
                'Pair': f"{s1}-{s2}",
                'Entry_Date': current_date,
                'Action': 'Sell_A_Buy_B',
                'Entry_Price_A': entry_price_a,
                'Entry_Price_B': entry_price_b,
                'q_a': q_a,
                'q_b': q_b,
                'Exit_Date': None,
                'Exit_Price_A': None,
                'Exit_Price_B': None,
                'PnL': None
            })
            open_trades.append(len(trades)-1)

        elif row['Important_Min']:
            entry_price_a = price_data.loc[current_date, s1]
            entry_price_b = price_data.loc[current_date, s2]
            q_a = 1.0
            q_b = -B
            trades.append({
                'Pair': f"{s1}-{s2}",
                'Entry_Date': current_date,
                'Action': 'Buy_A_Sell_B',
                'Entry_Price_A': entry_price_a,
                'Entry_Price_B': entry_price_b,
                'q_a': q_a,
                'q_b': q_b,
                'Exit_Date': None,
                'Exit_Price_A': None,
                'Exit_Price_B': None,
                'PnL': None
            })
            open_trades.append(len(trades)-1)

        trades_to_close = []
        for idx in open_trades:
            t = trades[idx]
            q_a = t['q_a']
            q_b = t['q_b']
            entry_price_a = t['Entry_Price_A']
            entry_price_b = t['Entry_Price_B']

            if t['Action'] == 'Sell_A_Buy_B' and row['Spread']<=row['Theta']:
                exit_price_a = price_data.loc[current_date, s1]
                exit_price_b = price_data.loc[current_date, s2]
                pnl_a = q_a*(exit_price_a - entry_price_a)
                pnl_b = q_b*(exit_price_b - entry_price_b)
                total_pnl = pnl_a+pnl_b
                tc_a = transaction_cost*(abs(q_a)*entry_price_a+abs(q_a)*exit_price_a)
                tc_b = transaction_cost*(abs(q_b)*entry_price_b+abs(q_b)*exit_price_b)
                total_pnl -= (tc_a+tc_b)
                trades[idx]['Exit_Date'] = current_date
                trades[idx]['Exit_Price_A'] = exit_price_a
                trades[idx]['Exit_Price_B'] = exit_price_b
                trades[idx]['PnL'] = total_pnl
                trades_to_close.append(idx)

            elif t['Action'] == 'Buy_A_Sell_B' and row['Spread']>=row['Theta']:
                exit_price_a = price_data.loc[current_date, s1]
                exit_price_b = price_data.loc[current_date, s2]
                pnl_a = q_a*(exit_price_a - entry_price_a)
                pnl_b = q_b*(exit_price_b - entry_price_b)
                total_pnl = pnl_a+pnl_b
                tc_a = transaction_cost*(abs(q_a)*entry_price_a+abs(q_a)*exit_price_a)
                tc_b = transaction_cost*(abs(q_b)*entry_price_b+abs(q_b)*exit_price_b)
                total_pnl -= (tc_a+tc_b)
                trades[idx]['Exit_Date'] = current_date
                trades[idx]['Exit_Price_A'] = exit_price_a
                trades[idx]['Exit_Price_B'] = exit_price_b
                trades[idx]['PnL'] = total_pnl
                trades_to_close.append(idx)

        open_trades = [o for o in open_trades if o not in trades_to_close]

    trades_df = pd.DataFrame(trades)
    total_pnl = trades_df['PnL'].sum() if not trades_df.empty else 0.0
    return trades_df, total_pnl

def plot_spread(spread, pair, B):
    if spread.empty:
        return None
    theta = spread.mean()
    skewness = spread.skew()
    q_levels = {
        '2σ': (0.025, 0.975),
        '2.5σ': (0.0062, 0.9938),
        '3σ': (0.0013, 0.9987)
    }
    quantiles = {}
    for label, (lq, uq) in q_levels.items():
        quantiles[label] = {
            'lower': spread.quantile(lq),
            'upper': spread.quantile(uq)
        }

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=spread,
        mode='lines',
        name='Spread',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=spread.index,
        y=[theta]*len(spread),
        mode='lines',
        name='Mean',
        line=dict(color='red', dash='dash')
    ))
    for label, q in quantiles.items():
        style = 'dash' if label == '2σ' else 'dot' if label == '2.5σ' else 'dashdot'
        fig.add_trace(go.Scatter(
            x=spread.index,
            y=[q['upper']]*len(spread),
            mode='lines',
            name=f'Upper {label}',
            line=dict(color='green', dash=style)
        ))
        fig.add_trace(go.Scatter(
            x=spread.index,
            y=[q['lower']]*len(spread),
            mode='lines',
            name=f'Lower {label}',
            line=dict(color='orange', dash=style)
        ))

    fig.update_layout(
        title=f'Spread for {pair} | B={B:.4f}, Skew={skewness:.2f}',
        xaxis_title='Date',
        yaxis_title='Spread',
        hovermode='x unified',
        width=1000,
        height=600
    )

    return fig

def plot_spread_with_trades(price_data, s1, s2, B, trades_log, C=2.0, window=60):
    spread = construct_spread(price_data, s1, s2, B)
    df = spread.to_frame('Spread')
    df['Theta'] = df['Spread'].rolling(window).mean()
    df['Sigma'] = df['Spread'].rolling(window).std()

    df['Is_Max'] = (df['Spread'].shift(1)<df['Spread'])&(df['Spread'].shift(-1)<df['Spread'])
    df['Is_Min'] = (df['Spread'].shift(1)>df['Spread'])&(df['Spread'].shift(-1)>df['Spread'])
    df['Important_Max'] = df['Is_Max'] & ((df['Spread']-df['Theta'])>=C*df['Sigma'])
    df['Important_Min'] = df['Is_Min'] & ((df['Theta']-df['Spread'])>=C*df['Sigma'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Spread'], mode='lines', name='Spread', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Theta'], mode='lines', name='Mean', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index[df['Important_Max']], y=df['Spread'][df['Important_Max']],
                             mode='markers', name='Important Max', marker=dict(color='orange', symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=df.index[df['Important_Min']], y=df['Spread'][df['Important_Min']],
                             mode='markers', name='Important Min', marker=dict(color='green', symbol='triangle-down')))

    # Plot trades
    for _, t in trades_log.iterrows():
        if pd.notna(t['Exit_Date']) and t['Entry_Date'] in df.index and t['Exit_Date'] in df.index:
            fig.add_trace(go.Scatter(
                x=[t['Entry_Date'], t['Exit_Date']],
                y=[df.loc[t['Entry_Date'], 'Spread'], df.loc[t['Exit_Date'], 'Spread']],
                mode='lines+markers',
                line=dict(color='purple', dash='dot'),
                marker=dict(color='purple', size=8),
                showlegend=False
            ))

    fig.update_layout(
        title=f"Spread & Trades for {s1}-{s2}",
        xaxis_title='Date',
        yaxis_title='Spread',
        hovermode='x unified',
        width=1000,
        height=600
    )
    return fig


#--------------------------------------------
# UI Layout
#--------------------------------------------
st.markdown("<div class='title'>Cointegrated ETFs Trading Strategy</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Discover and visualize cointegrated ETF pairs and backtest pairs trading strategies.</div>", unsafe_allow_html=True)

classification_df = load_etf_classification()

with st.sidebar:
    st.markdown("<div class='section-header'>ETF Classification Management</div>", unsafe_allow_html=True)
    st.write("Manage your ETF universe here.")
    st.dataframe(classification_df, height=200)

    ticker_input = st.text_input("Ticker symbol (e.g. SPY)")
    classification_input = st.text_input("Classification (e.g. S&P500)")
    leverage_input = st.number_input("Leverage (1, 2, or 3)", min_value=1, max_value=3, value=1)
    if st.button("Add ETF"):
        if ticker_input and classification_input:
            new_row = {'Ticker': ticker_input.upper(), 'Classification': classification_input, 'Leverage': leverage_input}
            classification_df = pd.concat([classification_df, pd.DataFrame([new_row])], ignore_index=True)
            save_etf_classification(classification_df)
            st.success("New ETF added! Please reload the app.")
        else:
            st.error("Please provide both Ticker and Classification.")

st.markdown("<div class='section-header'>Step 1: Cointegration Analysis</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2023-12-05"), key="analysis_start_date")
with col2:
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-10"), key="analysis_end_date")

CORR_THRESHOLD = 0.90
SIGNIFICANCE_LEVEL = 0.05
C_THRESHOLD = 2.0
VARIANCE_LIMIT = 40.0

etf_classification_dict = {row['Ticker']: (row['Classification'], row['Leverage']) for _, row in classification_df.iterrows()}

if st.button("Run Cointegration Analysis"):
    st.write("Running analysis, please wait...")
    tickers = list(etf_classification_dict.keys())
    price_data = fetch_price_data(tickers, str(start_date), str(end_date))

    if price_data.empty:
        st.error("No data fetched. Check the chosen dates or ensure tickers have data.")
    else:
        corr_matrix = price_data.corr()
        all_tickers = corr_matrix.columns
        stock_pairs = list(itertools.combinations(all_tickers, 2))
        filtered_pairs = [(a, b) for (a, b) in stock_pairs if abs(corr_matrix.loc[a, b]) >= CORR_THRESHOLD]

        # Part A
        results = []
        for (s1, s2) in filtered_pairs:
            B, mu, sigma = estimate_B_and_test_coint(price_data, s1, s2, SIGNIFICANCE_LEVEL)
            if B is not None:
                results.append({'Pair': f"{s1}-{s2}", 'Stock1': s1, 'Stock2': s2, 'B': B, 'Mean': mu, 'Std': sigma})

        if len(results) == 0:
            st.error("No cointegrated pairs found.")
            cointegrated_df = pd.DataFrame(columns=['Pair','Stock1','Stock2','B','Mean','Std'])
        else:
            cointegrated_df = pd.DataFrame(results)
            cointegrated_df.sort_values(by='Std', inplace=True)
            st.write(f"Found {len(cointegrated_df)} cointegrated pairs.")

        # Part B
        optimized_results = []
        for _, row in cointegrated_df.iterrows():
            pair = row['Pair']
            s1, s2 = row['Stock1'], row['Stock2']
            B = row['B']
            spread = construct_spread(price_data, s1, s2, B)
            spread_var = spread.var()
            if spread_var > VARIANCE_LIMIT:
                continue
            emrt = calculate_emrt(spread, C_THRESHOLD)
            adf_p = adfuller(spread.dropna())[1]
            if np.isnan(emrt) or adf_p >= SIGNIFICANCE_LEVEL:
                continue
            optimized_results.append({
                'Pair': pair,
                'Stock1': s1,
                'Stock2': s2,
                'B': B,
                'EMRT': emrt,
                'Variance': spread_var,
                'Mean': spread.mean(),
                'Std': spread.std()
            })

        optimized_df = pd.DataFrame(optimized_results).dropna(subset=['EMRT'])
        if not optimized_df.empty:
            optimized_df.sort_values(by='EMRT', inplace=True)
        st.write(f"Number of optimized cointegrated pairs: {len(optimized_df)}")

        filtered_rows = []
        for _, row in optimized_df.iterrows():
            s1, s2 = row['Stock1'], row['Stock2']
            if s1 not in etf_classification_dict or s2 not in etf_classification_dict:
                continue
            underlying1, lev1 = etf_classification_dict[s1]
            underlying2, lev2 = etf_classification_dict[s2]
            if underlying1 == underlying2 and lev1 != lev2:
                filtered_rows.append(row)

        final_df = pd.DataFrame(filtered_rows)
        st.write(f"Filtered cointegrated pairs (diff leverage, same underlying): {len(final_df)}")
        st.dataframe(final_df, height=250)

        st.session_state['final_pairs'] = final_df
        st.session_state['price_data'] = price_data
        st.success("Cointegration and optimization completed.")


if 'final_pairs' not in st.session_state or st.session_state['final_pairs'].empty:
    st.info("Please run the cointegration analysis first.")
else:
    final_pairs = st.session_state['final_pairs']
    price_data = st.session_state['price_data']

    st.markdown("<div class='section-header'>Step 2: Visualize Spread</div>", unsafe_allow_html=True)
    st.write("Here you can choose a different timeframe for viewing the spreads.")
    col3, col4 = st.columns(2)
    with col3:
        view_start_date = st.date_input("Start Date", value=pd.to_datetime("2024-06-05"), key="view_start_date")
    with col4:
        view_end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-10"), key="view_end_date")

    # Choose view mode: single pair or all pairs
    c_view_mode = st.radio("View Mode", ["All Pairs", "Single Pair"])

    if c_view_mode == "Single Pair":
        selected_pair_c = st.selectbox("Select a pair to plot", final_pairs['Pair'].unique())
    else:
        selected_pair_c = None

    if st.button("View Plots"):
        c_price_data = fetch_price_data(list(set(final_pairs['Stock1']).union(set(final_pairs['Stock2']))), str(view_start_date), str(view_end_date))
        if c_price_data.empty:
            st.error("No data for the selected timeframe.")
        else:
            if c_view_mode == "Single Pair" and selected_pair_c:
                row = final_pairs[final_pairs['Pair'] == selected_pair_c].iloc[0]
                s1, s2, B = row['Stock1'], row['Stock2'], row['B']
                spread = construct_spread(c_price_data, s1, s2, B)
                fig = plot_spread(spread, selected_pair_c, B)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # All pairs mode
                st.write("Displaying all pairs' plots:")
                for _, row in final_pairs.iterrows():
                    pair = row['Pair']
                    s1, s2, B = row['Stock1'], row['Stock2'], row['B']
                    spread = construct_spread(c_price_data, s1, s2, B)
                    fig = plot_spread(spread, pair, B)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)


    st.markdown("<div class='section-header'>Step 3: Backtest Strategy</div>", unsafe_allow_html=True)
    st.write("Select a timeframe for the backtest.")

    col5, col6 = st.columns(2)
    with col5:
        BACKTEST_START_DATE = st.date_input("Start Date", value=pd.to_datetime("2023-12-06"), key="backtest_start_date")
    with col6:
        BACKTEST_END_DATE = st.date_input("End Date", value=pd.to_datetime("2024-12-10"), key="backtest_end_date")

    TRANSACTION_COST = st.number_input("Transaction Cost (fraction)", 0.000, 0.01, 0.001)
    WINDOW_SIZE = st.number_input("Window Size for Rolling Mean/Std (Days)", 1, 200, 60)

    # Choose view mode for backtest
    d_view_mode = st.radio("View Mode for Backtest Results", ["All Pairs", "Single Pair"])

    if d_view_mode == "Single Pair":
        selected_pair_d = st.selectbox("Select a pair to see backtest plot", final_pairs['Pair'].unique())
    else:
        selected_pair_d = None

    if st.button("Run Backtest"):
        bt_price_data = fetch_price_data(list(set(final_pairs['Stock1']).union(set(final_pairs['Stock2']))),
                                         str(BACKTEST_START_DATE), str(BACKTEST_END_DATE))
        if bt_price_data.empty:
            st.error("No data fetched for backtest timeframe.")
        else:
            all_trades = []
            pair_pnl = {}
            for _, row in final_pairs.iterrows():
                pair = row['Pair']
                s1, s2, B = row['Stock1'], row['Stock2'], row['B']
                trades_log, total_profit = backtest_strategy(bt_price_data, s1, s2, B,
                                                             transaction_cost=TRANSACTION_COST,
                                                             C=C_THRESHOLD,
                                                             window=WINDOW_SIZE)
                pair_pnl[pair] = total_profit
                if not trades_log.empty:
                    all_trades.append(trades_log)

            if all_trades:
                combined_trades = pd.concat(all_trades, ignore_index=True)
            else:
                combined_trades = pd.DataFrame()

            st.write("Total PnL per pair:")
            for p, pnl in pair_pnl.items():
                st.write(f"{p}: ${pnl:.2f}")

            if d_view_mode == "Single Pair" and selected_pair_d and not combined_trades.empty:
                s1, s2, B = final_pairs[final_pairs['Pair'] == selected_pair_d].iloc[0][['Stock1','Stock2','B']]
                pair_trades = combined_trades[combined_trades['Pair'] == selected_pair_d]
                fig_bt = plot_spread_with_trades(bt_price_data, s1, s2, B, pair_trades, C=C_THRESHOLD, window=WINDOW_SIZE)
                st.plotly_chart(fig_bt, use_container_width=True)
            elif d_view_mode == "All Pairs" and not combined_trades.empty:
                st.write("Displaying all pairs' backtest plots:")
                for _, row in final_pairs.iterrows():
                    pair = row['Pair']
                    s1, s2, B = row['Stock1'], row['Stock2'], row['B']
                    pair_trades = combined_trades[combined_trades['Pair'] == pair]
                    fig_bt = plot_spread_with_trades(bt_price_data, s1, s2, B, pair_trades, C=C_THRESHOLD, window=WINDOW_SIZE)
                    st.plotly_chart(fig_bt, use_container_width=True)
            else:
                if combined_trades.empty:
                    st.info("No trades were executed based on the criteria. No plots to show.")
                else:
                    st.info("Please select a pair or change the view mode.")
