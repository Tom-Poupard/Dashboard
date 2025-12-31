import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.stats import norm
from datetime import datetime, timedelta

# ==========================================
# 1. CONFIGURATION & STYLE
# ==========================================
st.set_page_config(
    page_title="Market Dashboard Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "Pro" Look
st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 3rem; }
        h1 { text-align: center; margin-bottom: 0.5rem; }
        h2 { text-align: center; border-bottom: 1px solid #333; padding-bottom: 15px; margin-top: 3rem; font-size: 2rem; }
        
        /* KPI CARDS STYLE */
        .kpi-card {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            height: 140px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .kpi-title { font-size: 14px; color: #8b949e; font-weight: 600; }
        .kpi-value { font-size: 26px; color: #f0f6fc; font-weight: 700; margin: 5px 0; }
        
        /* PILLS */
        .kpi-pill-green { background-color: rgba(63, 185, 80, 0.15); color: #3fb950; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; width: fit-content; }
        .kpi-pill-red { background-color: rgba(248, 81, 73, 0.15); color: #f85149; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; width: fit-content; }
        .kpi-pill-blue { background-color: rgba(56, 139, 253, 0.15); color: #58a6ff; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; width: fit-content; }
        .kpi-pill-neutral { background-color: rgba(110, 118, 129, 0.15); color: #c9d1d9; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; width: fit-content; }
        
        /* ANALYTICS TEXT */
        .analysis-text {
            font-size: 1rem; line-height: 1.6; color: #b0b0b0;
            background-color: #0e1117; padding: 20px;
            border-left: 3px solid #2962FF; margin-top: 20px; border-radius: 0 4px 4px 0;
        }
        .analysis-highlight { color: #fff; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("Market Dashboard - Global Macro")

# ==========================================
# PART 1: OVERVIEW
# ==========================================
st.markdown("## Macroeconomics Overview")
st.write("### Live FX & Indices")
ticker_tape_html = """
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
  {
  "symbols": [
    {"proName": "FOREXCOM:SPXUSD", "title": "S&P 500"},
    {"proName": "FOREXCOM:NSXUSD", "title": "Nasdaq 100"},
    {"proName": "FX_IDC:EURUSD", "title": "EUR/USD"},
    {"proName": "FX_IDC:USDJPY", "title": "USD/JPY"},
    {"proName": "BITSTAMP:BTCUSD", "title": "Bitcoin"},
    {"proName": "FX_IDC:GBPUSD", "title": "GBP/USD"}
  ],
  "showSymbolLogo": true, "colorTheme": "dark", "isTransparent": false, "displayMode": "adaptive", "locale": "en"
}
  </script>
</div>
"""
components.html(ticker_tape_html, height=76)
st.write("---")

col_cal, col_news = st.columns(2)
with col_cal:
    st.subheader("Economic Calendar")
    components.html("""<div class="tradingview-widget-container"><div class="tradingview-widget-container__widget"></div><script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>{"colorTheme":"dark","isTransparent":false,"width":"100%","height":"500","locale":"en","importanceFilter":"0,1","countryFilter":"us,eu,gb,jp,cn"}</script></div>""", height=500, scrolling=True)
with col_news:
    st.subheader("Latest News")
    components.html("""<div class="tradingview-widget-container"><div class="tradingview-widget-container__widget"></div><script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-timeline.js" async>{"feedMode":"market","market":"forex","colorTheme":"dark","isTransparent":false,"displayMode":"regular","width":"100%","height":"500","locale":"en"}</script></div>""", height=500, scrolling=True)

st.write("---")
st.subheader("Market Overview")
components.html("""<div class="tradingview-widget-container"><div class="tradingview-widget-container__widget"></div><script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-market-overview.js" async>{"colorTheme":"dark","dateRange":"12M","showChart":true,"locale":"en","largeChartUrl":"","isTransparent":false,"showSymbolLogo":true,"width":"100%","height":"550","tabs":[{"title":"Indices","symbols":[{"s":"FOREXCOM:SPXUSD","d":"S&P 500"},{"s":"FOREXCOM:NSXUSD","d":"Nasdaq 100"},{"s":"FOREXCOM:DJI","d":"Dow 30"},{"s":"INDEX:NKY","d":"Nikkei 225"},{"s":"INDEX:DEU40","d":"DAX 40"}]},{"title":"Crypto","symbols":[{"s":"BITSTAMP:BTCUSD","d":"Bitcoin"},{"s":"BITSTAMP:ETHUSD","d":"Ethereum"},{"s":"BITSTAMP:SOLUSD","d":"Solana"}]},{"title":"Forex","symbols":[{"s":"FX:EURUSD","d":"EUR/USD"},{"s":"FX:GBPUSD","d":"GBP/USD"},{"s":"FX:USDJPY","d":"USD/JPY"}]}]}</script></div>""", height=550)

# ==========================================
# PART 2: ASSET FOCUS
# ==========================================
st.markdown("## Asset Focus")
tab_cross, tab_bonds, tab_forex, tab_commodities, tab_tickers = st.tabs(["Cross Asset & Equity", "US Bonds (1Y-30Y)", "Forex", "Commodities", "Ticker Reference Guide"])

# -----------------------------------------------------------------------------
# TAB 1: CROSS ASSET & EQUITY
# -----------------------------------------------------------------------------
with tab_cross:
    st.subheader("Cross-Asset Comparative Analysis")
    col_input, col_period = st.columns([3, 1])
    
    with col_input:
        tickers_input = st.text_input("Assets (comma separated):", value="AAPL, AMZN, MSFT, BTC-USD", help="Ex: ^GSPC, BTC-USD, NVDA")
        tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    
    with col_period:
        time_period = st.selectbox("Horizon:", ["1M", "3M", "6M", "YTD", "1Y", "2Y", "5Y", "10Y", "20Y"], index=3)
        col_ma1, col_ma2 = st.columns(2)
        with col_ma1: ma_window_1 = st.slider("Short MA", 5, 100, 20, 5)
        with col_ma2: ma_window_2 = st.slider("Long MA", 20, 200, 50, 10)

    # Date Logic
    end_date = datetime.now()
    if time_period == "1M": start_d = end_date - timedelta(days=30)
    elif time_period == "3M": start_d = end_date - timedelta(days=90)
    elif time_period == "6M": start_d = end_date - timedelta(days=180)
    elif time_period == "YTD": start_d = datetime(end_date.year, 1, 1)
    elif time_period == "1Y": start_d = end_date - timedelta(days=365)
    elif time_period == "2Y": start_d = end_date - timedelta(days=730)
    elif time_period == "5Y": start_d = end_date - timedelta(days=365*5)
    elif time_period == "10Y": start_d = end_date - timedelta(days=365*10)
    elif time_period == "20Y": start_d = end_date - timedelta(days=365*20)
    else: start_d = end_date - timedelta(days=365)

    if tickers_list:
        try:
            with st.spinner("Loading data..."):
                data_df = yf.download(tickers_list, start=start_d, end=end_date, progress=False)['Close']
                data_df.index = data_df.index.tz_localize(None)
                if isinstance(data_df, pd.Series): data_df = data_df.to_frame(name=tickers_list[0])
                data_df = data_df.ffill().dropna()

                if not data_df.empty:
                    daily_returns = data_df.pct_change().dropna()
                    mean_daily = daily_returns.mean()
                    volatility = daily_returns.std() * (252**0.5)

                    st.markdown("### Performance Analytics")
                    m1, m2, m3, m4 = st.columns(4)
                    with m1: st.metric("Best Performer", mean_daily.idxmax(), f"+{mean_daily.max()*100:.2f}% (daily avg)")
                    with m2: st.metric("Highest Volatility", volatility.idxmax(), f"{volatility.max()*100:.2f}% (ann.)")
                    with m3: st.metric("Worst Performer", mean_daily.idxmin(), f"{mean_daily.min()*100:.2f}% (daily avg)")
                    with m4: st.metric("Lowest Volatility", volatility.idxmin(), f"{volatility.min()*100:.2f}% (ann.)")

                    # MAIN CHART
                    st.write("")
                    normalized_df = (data_df / data_df.iloc[0]) * 100
                    fig_main = go.Figure()
                    colors = ['#2962FF', '#00E676', '#FF5252', '#FFD600', '#AB47BC', '#00BCD4']
                    for i, ticker in enumerate(normalized_df.columns):
                        fig_main.add_trace(go.Scatter(x=normalized_df.index, y=normalized_df[ticker], mode='lines', name=ticker, line=dict(width=2, color=colors[i % len(colors)])))
                    fig_main.update_layout(title="Relative Performance (Base 100)", template="plotly_dark", height=450, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified", legend=dict(orientation="h", y=1.02))
                    st.plotly_chart(fig_main, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # 1. PRICE TRENDS
                    st.markdown(f"#### Price Trends (with MA{ma_window_1} & MA{ma_window_2})")
                    rows_price = st.columns(len(data_df.columns)) 
                    for i, asset in enumerate(data_df.columns):
                        with rows_price[i]:
                            fig_p = go.Figure()
                            fig_p.add_trace(go.Scatter(x=data_df.index, y=data_df[asset], mode='lines', name='Price', line=dict(color='#2962FF', width=1.5)))
                            ma_1 = data_df[asset].rolling(window=ma_window_1).mean()
                            ma_2 = data_df[asset].rolling(window=ma_window_2).mean()
                            fig_p.add_trace(go.Scatter(x=data_df.index, y=ma_1, mode='lines', name=f'MA{ma_window_1}', line=dict(color='#00E676', width=1)))
                            fig_p.add_trace(go.Scatter(x=data_df.index, y=ma_2, mode='lines', name=f'MA{ma_window_2}', line=dict(color='#FF5252', width=1)))
                            fig_p.update_layout(title=dict(text=asset, x=0.5, xanchor='center'), template="plotly_dark", height=250, showlegend=False, margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#333'))
                            st.plotly_chart(fig_p, use_container_width=True, key=f"price_{asset}")

                    # 2. DISTRIBUTIONS
                    st.write("")
                    st.markdown("#### Return Distributions")
                    rows_dist = st.columns(len(data_df.columns))
                    for i, asset in enumerate(data_df.columns):
                        with rows_dist[i]:
                            returns = daily_returns[asset].dropna()
                            fig_h = go.Figure()
                            fig_h.add_trace(go.Histogram(x=returns, nbinsx=30, histnorm='probability density', name='Dist', marker_color='rgba(255, 255, 255, 0.9)'))
                            fig_h.add_vline(x=returns.mean(), line_width=2, line_color="#FF5252")
                            fig_h.update_layout(title=dict(text=asset, x=0.5, xanchor='center'), template="plotly_dark", height=250, showlegend=False, margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
                            st.plotly_chart(fig_h, use_container_width=True, key=f"dist_{asset}")

                    # 3. TEXT ANALYSIS
                    avg_perf = (normalized_df.iloc[-1] - 100).mean()
                    best_total_asset = (normalized_df.iloc[-1] / normalized_df.iloc[0]).idxmax()
                    best_total_perf = ((normalized_df.iloc[-1] / normalized_df.iloc[0]).max() - 1) * 100
                    worst_total_asset = (normalized_df.iloc[-1] / normalized_df.iloc[0]).idxmin()
                    worst_total_perf = ((normalized_df.iloc[-1] / normalized_df.iloc[0]).min() - 1) * 100
                    
                    corr_text = "N/A"
                    if len(data_df.columns) > 1:
                        corr_matrix = daily_returns.corr()
                        np.fill_diagonal(corr_matrix.values, np.nan)
                        min_corr = corr_matrix.min().min()
                        max_corr = corr_matrix.max().max()
                        corr_text = f"Cross-asset correlation spans <span class='analysis-highlight'>{min_corr:.2f}</span> to <span class='analysis-highlight'>{max_corr:.2f}</span>; diversification benefits improve as correlation decreases."

                    analysis_html = f"""
                    <div class="analysis-text">
                        <b>Peer recap ({time_period}).</b> 
                        Average performance across the selection is <span class='analysis-highlight'>{avg_perf:+.0f}%</span>. 
                        Best performer: <span class='analysis-highlight'>{best_total_asset} ({best_total_perf:+.0f}%)</span>. 
                        Worst performer: <span class='analysis-highlight'>{worst_total_asset} ({worst_total_perf:+.0f}%)</span>. 
                        Realized volatility (annualized) ranges from <span class='analysis-highlight'>{volatility.min()*100:.1f}%</span> ({volatility.idxmin()}) 
                        to <span class='analysis-highlight'>{volatility.max()*100:.1f}%</span> ({volatility.idxmax()}). 
                        {corr_text}
                    </div>
                    """
                    st.markdown(analysis_html, unsafe_allow_html=True)

                    if len(data_df.columns) > 1:
                        st.subheader("Cross-Asset Correlation")
                        fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index, colorscale='RdBu', zmin=-1, zmax=1, text=corr_matrix.values, texttemplate="%{text:.2f}", textfont={"color": "white"}))
                        fig_corr.update_layout(template="plotly_dark", height=400 + (len(data_df.columns)*20), margin=dict(l=20, r=20, t=20, b=20), yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig_corr, use_container_width=True)

                    # =======================================================
                    # OPTIONS SECTION (EQUITY)
                    # =======================================================
                    st.write("---")
                    st.subheader("Options Volatility Analysis")
                    col_target_d, col_empty = st.columns([1, 3])
                    with col_target_d:
                        target_days = st.selectbox("Target Horizon:", [30, 60, 90, 180, 360], index=0, format_func=lambda x: f"~ {x} Days")

                    if st.button("Run Volatility Scan"):
                        comp_data = []
                        target_dt = datetime.now() + timedelta(days=target_days)
                        progress = st.progress(0)
                        for idx, tk_sym in enumerate(tickers_list):
                            try:
                                tk = yf.Ticker(tk_sym)
                                exps = tk.options
                                if not exps: continue
                                dates = [datetime.strptime(d, '%Y-%m-%d') for d in exps]
                                closest_d = min(dates, key=lambda d: abs(d - target_dt))
                                close_str = closest_d.strftime('%Y-%m-%d')
                                days_exp = (closest_d - datetime.now()).days
                                opt = tk.option_chain(close_str)
                                try: curr_px = tk.fast_info['last_price']
                                except: curr_px = data_df[tk_sym].iloc[-1]
                                call_atm = opt.calls.iloc[(opt.calls['strike'] - curr_px).abs().argsort()[:1]]
                                put_atm = opt.puts.iloc[(opt.puts['strike'] - curr_px).abs().argsort()[:1]]
                                comp_data.append({"Ticker": tk_sym, "Expiry": close_str, "Days": days_exp, "Call IV": call_atm['impliedVolatility'].values[0], "Put IV": put_atm['impliedVolatility'].values[0]})
                            except: continue
                            progress.progress((idx+1)/len(tickers_list))
                        progress.empty()
                        if comp_data:
                            df_comp = pd.DataFrame(comp_data)
                            c_tbl, c_graph = st.columns(2)
                            with c_tbl: st.dataframe(df_comp.style.format({"Call IV": "{:.2%}", "Put IV": "{:.2%}"}), use_container_width=True)
                            with c_graph:
                                f_iv = go.Figure()
                                f_iv.add_trace(go.Bar(x=df_comp['Ticker'], y=df_comp['Call IV'], name='Call IV', marker_color='#00E676'))
                                f_iv.add_trace(go.Bar(x=df_comp['Ticker'], y=df_comp['Put IV'], name='Put IV', marker_color='#FF5252'))
                                f_iv.update_layout(title="IV ATM Comparison", template="plotly_dark", barmode='group', height=300)
                                st.plotly_chart(f_iv, use_container_width=True)
                        else: st.warning("No data available.")

                    st.write("---")
                    st.markdown("#### Single Asset Focus (Term Structure & Surface)")
                    col_opt_sel, col_slide = st.columns([1, 2])
                    with col_opt_sel:
                        sel_opt = st.selectbox("Select Asset:", tickers_list)
                    with col_slide:
                        money_rng = st.slider("Moneyness Filter (3D)", 0.5, 1.5, (0.7, 1.3), 0.05)

                    if sel_opt:
                        tk = yf.Ticker(sel_opt)
                        exps = tk.options
                        if exps:
                            # 1. TERM STRUCTURE
                            ts_data = []
                            try: curr_px = data_df[sel_opt].iloc[-1]
                            except: curr_px = 100
                            for d in exps[:12]:
                                try:
                                    chain = tk.option_chain(d)
                                    c_atm = chain.calls.iloc[(chain.calls['strike'] - curr_px).abs().argsort()[:1]]
                                    p_atm = chain.puts.iloc[(chain.puts['strike'] - curr_px).abs().argsort()[:1]]
                                    ts_data.append({"Date": d, "Call IV": c_atm['impliedVolatility'].values[0], "Put IV": p_atm['impliedVolatility'].values[0]})
                                except: continue
                            
                            if ts_data:
                                df_ts = pd.DataFrame(ts_data)
                                f_ts = go.Figure()
                                f_ts.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['Call IV'], name='Call IV', mode='lines+markers', line=dict(color='#00E676', width=2)))
                                f_ts.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['Put IV'], name='Put IV', mode='lines+markers', line=dict(color='#FF5252', width=2)))
                                f_ts.update_layout(title=f"Term Structure - {sel_opt}", template="plotly_dark", height=350, yaxis_tickformat='.1%')
                                st.plotly_chart(f_ts, use_container_width=True)

                            st.write("")
                            # 2. 3D SURFACE
                            if st.button("Generate 3D Surface"):
                                with st.spinner("Building 3D Model..."):
                                    surf_data = []
                                    for d in exps[:8]:
                                        try:
                                            chain = tk.option_chain(d).calls
                                            days = (datetime.strptime(d, "%Y-%m-%d") - datetime.now()).days
                                            if days < 5: continue
                                            chain = chain[chain['impliedVolatility'] > 0.001]
                                            chain = chain[(chain['strike'] >= curr_px*money_rng[0]) & (chain['strike'] <= curr_px*money_rng[1])]
                                            for _, r in chain.iterrows():
                                                surf_data.append({'S': r['strike'], 'T': days, 'V': r['impliedVolatility']})
                                        except: continue
                                    if surf_data:
                                        df_s = pd.DataFrame(surf_data)
                                        df_s = df_s.dropna() 
                                        df_s = df_s.groupby(['S', 'T'])['V'].mean().reset_index()
                                        if len(df_s) > 10:
                                            x = df_s['S'].values
                                            y = df_s['T'].values
                                            z = df_s['V'].values
                                            xi = np.linspace(x.min(), x.max(), 40)
                                            yi = np.linspace(y.min(), y.max(), 40)
                                            X, Y = np.meshgrid(xi, yi)
                                            try:
                                                Z = griddata((x, y), z, (X, Y), method='cubic')
                                                custom_colors = [[0, 'rgb(0,0,255)'], [0.2, 'rgb(0,255,255)'], [0.4, 'rgb(0,255,0)'], [0.6, 'rgb(255,255,0)'], [0.8, 'rgb(255,128,0)'], [1, 'rgb(255,0,0)']]
                                                fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale=custom_colors, opacity=0.9)])
                                                fig_3d.update_layout(title=f"Vol Surface (Interpolated) - {sel_opt}", scene=dict(xaxis_title='Strike', yaxis_title='DTE', zaxis_title='IV', aspectratio=dict(x=1, y=1, z=0.5)), template="plotly_dark", height=600)
                                                st.plotly_chart(fig_3d, use_container_width=True)
                                            except Exception as interp_err: st.error(f"Interpolation Error: {interp_err}")
                                        else: st.warning("Not enough valid data points.")
                                    else: st.warning("Insufficient options data.")
        except Exception as e: st.error(f"Technical Error: {e}")

# -----------------------------------------------------------------------------
# TAB 2: US BONDS (FRED SOURCE) & CORPORATE BONDS
# -----------------------------------------------------------------------------
with tab_bonds:
    st.subheader("US Government Bonds (State Yields)")
    st.info("Select a Tenor and a Date Range to visualize the yield trend. Data sourced from FRED (Federal Reserve Economic Data).")

    # 1. Inputs: Tenor & Date Range
    col_sel, col_date = st.columns([1, 2])
    
    with col_sel:
        # Full FRED Series Mapping
        tenor_map = {
            "1 Month": "DGS1MO",
            "3 Month": "DGS3MO",
            "6 Month": "DGS6MO",
            "1 Year": "DGS1",
            "2 Year": "DGS2",
            "3 Year": "DGS3",
            "5 Year": "DGS5",
            "7 Year": "DGS7",
            "10 Year": "DGS10",
            "20 Year": "DGS20",
            "30 Year": "DGS30"
        }
        selected_tenor = st.selectbox("Select Bond Tenor:", list(tenor_map.keys()), index=8) # Default 10Y
        series_id = tenor_map[selected_tenor]

    with col_date:
        # Date Range Selector
        default_start = datetime.now() - timedelta(days=365)
        default_end = datetime.now()
        date_range = st.date_input("Select Date Range:", value=(default_start, default_end))
        
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_b, end_b = date_range
        else:
            start_b, end_b = default_start, default_end

    # 2. Fetch Data from FRED (CSV)
    if series_id:
        try:
            with st.spinner(f"Fetching data for {selected_tenor} from FRED..."):
                # Construct FRED CSV URL
                fred_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
                
                # Read CSV
                bond_df = pd.read_csv(fred_url, index_col=0, parse_dates=True)
                
                # Clean Data (FRED uses '.' for missing values)
                bond_df = bond_df.replace('.', np.nan).dropna()
                bond_df = bond_df.astype(float)
                
                # Filter Date Range
                mask = (bond_df.index >= pd.Timestamp(start_b)) & (bond_df.index <= pd.Timestamp(end_b))
                bond_df = bond_df.loc[mask]
                
                if not bond_df.empty:
                    # Rename column for clarity
                    bond_df.columns = ["Yield"]
                    
                    # 3. KPI Display
                    curr_yield = bond_df["Yield"].iloc[-1]
                    prev_yield = bond_df["Yield"].iloc[-2] if len(bond_df) > 1 else curr_yield
                    change_bps = (curr_yield - prev_yield) * 100
                    
                    color_cls = "kpi-pill-green" if change_bps >= 0 else "kpi-pill-red"
                    sign = "+" if change_bps >= 0 else ""
                    
                    st.write("")
                    st.markdown(f"""
                    <div class="kpi-card" style="width: 300px;">
                        <div>
                            <div class="kpi-title">{selected_tenor} Yield</div>
                            <div class="kpi-value">{curr_yield:.2f}%</div>
                            <div class="{color_cls}">1D {sign}{change_bps:.1f} bps</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 4. Chart
                    st.write("")
                    st.subheader(f"{selected_tenor} Yield History")
                    
                    fig_bond = go.Figure()
                    fig_bond.add_trace(go.Scatter(
                        x=bond_df.index, 
                        y=bond_df["Yield"], 
                        mode='lines', 
                        name='Yield',
                        line=dict(color='#2962FF', width=2.5),
                        fill='tozeroy',
                        fillcolor='rgba(41, 98, 255, 0.1)'
                    ))
                    
                    fig_bond.update_layout(
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=500,
                        hovermode="x unified",
                        yaxis_title="Yield (%)",
                        xaxis_title="Date",
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_bond, use_container_width=True)
                    
                else:
                    st.warning("No data found for this range.")
                    
        except Exception as e:
            st.error(f"Error fetching bond data from FRED: {e}")

    # 3. Corporate Bonds Section
    st.write("---")
    st.subheader("Corporate Bonds (Credit Market)")
    st.markdown("""
    * **LQD:** iShares iBoxx $ Investment Grade Corporate Bond ETF
    * **HYG:** iShares iBoxx $ High Yield Corporate Bond ETF
    * *Note: These are ETF Prices. Price goes DOWN when Yields go UP.*
    """)
    
    # Fetch Corp Data (Yahoo Finance)
    try:
        corp_tickers = ["LQD", "HYG", "SHY"]
        with st.spinner("Fetching Corporate Bond data..."):
            corp_raw = yf.download(corp_tickers, start=start_b, end=end_b, progress=False)
            
            # Handling multi-level columns
            if isinstance(corp_raw.columns, pd.MultiIndex):
                corp_df = corp_raw['Close']
            else:
                corp_df = corp_raw['Close'] # Should not happen with multiple tickers but safe to have
            
            corp_df.index = corp_df.index.tz_localize(None)
            corp_df = corp_df.dropna()

            if not corp_df.empty:
                # Normalize to 100
                corp_norm = (corp_df / corp_df.iloc[0]) * 100
                
                fig_corp = go.Figure()
                fig_corp.add_trace(go.Scatter(x=corp_norm.index, y=corp_norm['LQD'], name="Investment Grade (LQD)", line=dict(color='#00E676', width=2)))
                fig_corp.add_trace(go.Scatter(x=corp_norm.index, y=corp_norm['HYG'], name="High Yield (HYG)", line=dict(color='#FF5252', width=2)))
                fig_corp.add_trace(go.Scatter(x=corp_norm.index, y=corp_norm['SHY'], name="Short Gov (SHY - Cash Proxy)", line=dict(color='#888888', width=1, dash='dash')))
                
                fig_corp.update_layout(
                    title="Corporate Bond Price Performance (Rebased to 100)",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    hovermode="x unified",
                    yaxis_title="Price (Base 100)",
                    legend=dict(orientation="h", y=1.05)
                )
                st.plotly_chart(fig_corp, use_container_width=True, key="corp_chart")
            else:
                st.warning("No Corporate Bond data found for this range.")
    except Exception as e:
        st.error(f"Error fetching corporate bonds: {e}")

# -----------------------------------------------------------------------------
# TAB 3: FOREX (WITH RESTORED OPTIONS)
# -----------------------------------------------------------------------------
with tab_forex:
    st.subheader("Currencies")
    fx_pairs = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'USDCHF=X', 'AUDUSD=X', 'USDCAD=X']
    fx_lbl = {'EURUSD=X': 'EUR/USD', 'USDJPY=X': 'USD/JPY', 'GBPUSD=X': 'GBP/USD', 'USDCHF=X': 'USD/CHF', 'AUDUSD=X': 'AUD/USD', 'USDCAD=X': 'USD/CAD'}
    
    @st.cache_data(ttl=300)
    def get_fx(): return yf.download(fx_pairs, period="1y", progress=False)['Close']
    fx_df = get_fx()
    
    c_fx = st.columns(6)
    if not fx_df.empty:
        for i, t in enumerate(fx_pairs):
            if t in fx_df.columns:
                s = fx_df[t].dropna()
                cur = s.iloc[-1]
                ytd = ((cur - s.iloc[0])/s.iloc[0])*100
                cls_ = "kpi-pill-blue" if ytd >= 0 else "kpi-pill-neutral"
                sign = "+" if ytd >= 0 else ""
                with c_fx[i]:
                    st.markdown(f"""<div class="kpi-card" style="height:120px; padding:10px;"><div class="kpi-title">{fx_lbl[t]}</div><div class="kpi-value" style="font-size:20px;">{cur:.4f}</div><div class="{cls_}">YTD {sign}{ytd:.2f}%</div></div>""", unsafe_allow_html=True)
                    f_s = go.Figure(go.Scatter(y=s[-30:], mode='lines', line=dict(color='#00E676' if ytd>=0 else '#FF5252', width=2), fill='tozeroy', fillcolor='rgba(255,255,255,0.05)'))
                    f_s.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=30, xaxis=dict(visible=False), yaxis=dict(visible=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
                    st.plotly_chart(f_s, use_container_width=True, config={'displayModeBar': False}, key=f"fx_kpi_{i}")

    st.write("")
    col_sel, col_time = st.columns([1, 4])
    with col_sel:
        st.write("")
        sel_pair = st.selectbox("Select Pair", list(fx_lbl.values()))
        sel_t = [k for k,v in fx_lbl.items() if v == sel_pair][0]
    
    with col_time:
        st.write("")
        c_rad, c_date = st.columns([2, 2])
        with c_rad:
            fx_horizon = st.radio("Time Horizon", ["3M", "6M", "YTD", "1Y", "5Y", "10Y", "20Y"], horizontal=True, index=3, key="fx_hz")
        
        end_d = datetime.now()
        if fx_horizon == "3M": start_d = end_d - timedelta(days=90)
        elif fx_horizon == "6M": start_d = end_d - timedelta(days=180)
        elif fx_horizon == "YTD": start_d = datetime(end_d.year, 1, 1)
        elif fx_horizon == "1Y": start_d = end_d - timedelta(days=365)
        elif fx_horizon == "5Y": start_d = end_d - timedelta(days=365*5)
        elif fx_horizon == "10Y": start_d = end_d - timedelta(days=365*10)
        elif fx_horizon == "20Y": start_d = end_d - timedelta(days=365*20)
        else: start_d = end_d - timedelta(days=365)
        
        with c_date:
            sel_d = st.date_input("Date Range", value=(start_d, end_d), key="fx_date", label_visibility="collapsed")
            if isinstance(sel_d, tuple) and len(sel_d) == 2: s_date, e_date = sel_d
            else: s_date, e_date = start_d, end_d

    st.markdown(f"##### {sel_pair} Performance")
    if sel_t:
        try:
            chart_data = yf.download(sel_t, start=s_date, end=e_date, progress=False)
            if isinstance(chart_data.columns, pd.MultiIndex): chart_data = chart_data['Close'][sel_t]
            else: chart_data = chart_data['Close']
            chart_data.index = chart_data.index.tz_localize(None)
            chart_data = chart_data.dropna().astype(float)

            if not chart_data.empty:
                ma = chart_data.rolling(50).mean()
                f_main = go.Figure()
                f_main.add_trace(go.Scatter(x=chart_data.index, y=chart_data, name="Price", mode='lines', line=dict(color="#f0f6fc", width=2)))
                f_main.add_trace(go.Scatter(x=ma.index, y=ma, name="MA 50", mode='lines', line=dict(color="#8b949e", width=1, dash='dash')))
                f_main.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=20,r=20,t=20,b=20), hovermode="x unified")
                st.plotly_chart(f_main, use_container_width=True, key="fx_main_chart")
        except Exception as e: st.error(f"Error loading chart: {e}")

    # =======================================================
    # OPTIONS SECTION (FOREX) - RESTORED
    # =======================================================
    st.write("---")
    st.subheader("Options Volatility Analysis")
    st.info("Note: Spot FX pairs (e.g., EURUSD=X) rarely have public options data. Use Currency ETFs (e.g., FXE, FXY) below.")
    
    col_input_fx, col_target_fx = st.columns([2, 1])
    with col_input_fx:
        opt_tickers_fx = st.text_input("Options Assets (ETFs):", value="FXE, FXY, UUP, FXB, FXC", help="Enter tickers with options (e.g., ETFs)")
        opt_list_fx = [t.strip().upper() for t in opt_tickers_fx.split(',') if t.strip()]
    with col_target_fx:
        target_days_fx = st.selectbox("Target Horizon:", [30, 60, 90, 180, 360], index=0, format_func=lambda x: f"~ {x} Days", key="fx_opt_hz")

    if st.button("Run Volatility Scan (Forex)", key="run_vol_fx"):
        comp_data = []
        target_dt = datetime.now() + timedelta(days=target_days_fx)
        progress = st.progress(0)
        for idx, tk_sym in enumerate(opt_list_fx):
            try:
                tk = yf.Ticker(tk_sym)
                exps = tk.options
                if not exps: continue
                dates = [datetime.strptime(d, '%Y-%m-%d') for d in exps]
                closest_d = min(dates, key=lambda d: abs(d - target_dt))
                close_str = closest_d.strftime('%Y-%m-%d')
                days_exp = (closest_d - datetime.now()).days
                opt = tk.option_chain(close_str)
                try: curr_px = tk.fast_info['last_price']
                except: curr_px = tk.history(period="1d")['Close'].iloc[-1]
                call_atm = opt.calls.iloc[(opt.calls['strike'] - curr_px).abs().argsort()[:1]]
                put_atm = opt.puts.iloc[(opt.puts['strike'] - curr_px).abs().argsort()[:1]]
                comp_data.append({"Ticker": tk_sym, "Expiry": close_str, "Days": days_exp, "Call IV": call_atm['impliedVolatility'].values[0], "Put IV": put_atm['impliedVolatility'].values[0]})
            except: continue
            progress.progress((idx+1)/len(opt_list_fx))
        progress.empty()
        
        if comp_data:
            df_comp = pd.DataFrame(comp_data)
            c_tbl, c_graph = st.columns(2)
            with c_tbl: st.dataframe(df_comp.style.format({"Call IV": "{:.2%}", "Put IV": "{:.2%}"}), use_container_width=True)
            with c_graph:
                f_iv = go.Figure()
                f_iv.add_trace(go.Bar(x=df_comp['Ticker'], y=df_comp['Call IV'], name='Call IV', marker_color='#00E676'))
                f_iv.add_trace(go.Bar(x=df_comp['Ticker'], y=df_comp['Put IV'], name='Put IV', marker_color='#FF5252'))
                f_iv.update_layout(title="IV ATM Comparison (ETFs)", template="plotly_dark", barmode='group', height=300)
                st.plotly_chart(f_iv, use_container_width=True)
        else: st.warning("No options data found.")

    st.write("---")
    st.markdown("#### Single Asset Focus")
    col_opt_sel, col_slide = st.columns([1, 2])
    with col_opt_sel:
        sel_opt_fx = st.selectbox("Select Asset:", opt_list_fx, key="sel_opt_fx")
    with col_slide:
        money_rng_fx = st.slider("Moneyness Filter (3D)", 0.5, 1.5, (0.7, 1.3), 0.05, key="slide_opt_fx")

    if sel_opt_fx:
        tk = yf.Ticker(sel_opt_fx)
        exps = tk.options
        if exps:
            # 1. TERM STRUCTURE
            ts_data = []
            try: curr_px = tk.history(period="1d")['Close'].iloc[-1]
            except: curr_px = 100
            for d in exps[:12]:
                try:
                    chain = tk.option_chain(d)
                    c_atm = chain.calls.iloc[(chain.calls['strike'] - curr_px).abs().argsort()[:1]]
                    p_atm = chain.puts.iloc[(chain.puts['strike'] - curr_px).abs().argsort()[:1]]
                    ts_data.append({"Date": d, "Call IV": c_atm['impliedVolatility'].values[0], "Put IV": p_atm['impliedVolatility'].values[0]})
                except: continue
            
            if ts_data:
                df_ts = pd.DataFrame(ts_data)
                f_ts = go.Figure()
                f_ts.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['Call IV'], name='Call IV', mode='lines+markers', line=dict(color='#00E676', width=2)))
                f_ts.add_trace(go.Scatter(x=df_ts['Date'], y=df_ts['Put IV'], name='Put IV', mode='lines+markers', line=dict(color='#FF5252', width=2)))
                f_ts.update_layout(title=f"Term Structure - {sel_opt_fx}", template="plotly_dark", height=350, yaxis_tickformat='.1%')
                st.plotly_chart(f_ts, use_container_width=True)

            st.write("")
            # 2. 3D SURFACE
            if st.button("Generate 3D Surface", key="gen_3d_fx"):
                with st.spinner("Building 3D Model..."):
                    surf_data = []
                    for d in exps[:8]:
                        try:
                            chain = tk.option_chain(d).calls
                            days = (datetime.strptime(d, "%Y-%m-%d") - datetime.now()).days
                            if days < 5: continue
                            chain = chain[chain['impliedVolatility'] > 0.001]
                            chain = chain[(chain['strike'] >= curr_px*money_rng_fx[0]) & (chain['strike'] <= curr_px*money_rng_fx[1])]
                            for _, r in chain.iterrows():
                                surf_data.append({'S': r['strike'], 'T': days, 'V': r['impliedVolatility']})
                        except: continue
                    if surf_data:
                        df_s = pd.DataFrame(surf_data)
                        df_s = df_s.dropna() 
                        df_s = df_s.groupby(['S', 'T'])['V'].mean().reset_index()
                        if len(df_s) > 10:
                            x = df_s['S'].values
                            y = df_s['T'].values
                            z = df_s['V'].values
                            xi = np.linspace(x.min(), x.max(), 40)
                            yi = np.linspace(y.min(), y.max(), 40)
                            X, Y = np.meshgrid(xi, yi)
                            try:
                                Z = griddata((x, y), z, (X, Y), method='cubic')
                                custom_colors = [[0, 'rgb(0,0,255)'], [0.2, 'rgb(0,255,255)'], [0.4, 'rgb(0,255,0)'], [0.6, 'rgb(255,255,0)'], [0.8, 'rgb(255,128,0)'], [1, 'rgb(255,0,0)']]
                                fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale=custom_colors, opacity=0.9)])
                                fig_3d.update_layout(title=f"Vol Surface (Interpolated) - {sel_opt_fx}", scene=dict(xaxis_title='Strike', yaxis_title='DTE', zaxis_title='IV', aspectratio=dict(x=1, y=1, z=0.5)), template="plotly_dark", height=600)
                                st.plotly_chart(fig_3d, use_container_width=True)
                            except Exception as interp_err: st.error(f"Interpolation Error: {interp_err}")
                        else: st.warning("Not enough valid data points.")
                    else: st.warning("Insufficient options data.")

# -----------------------------------------------------------------------------
# TAB 4: COMMODITIES
# -----------------------------------------------------------------------------
with tab_commodities:
    st.subheader("Commodity Analysis")
    
    col_input, col_time = st.columns([2, 1])
    with col_input:
        com_input = st.text_input("Commodities (Tickers):", value="GC=F, SI=F, CL=F, NG=F, HG=F", help="Enter Yahoo Finance tickers separated by commas")

    with col_time:
        time_period_com = st.radio("Time Horizon", ["1M", "3M", "6M", "YTD", "1Y", "5Y", "10Y", "15Y", "20Y"], horizontal=True, index=3)
        end_d_c = datetime.now()
        if time_period_com == "1M": start_d_c = end_d_c - timedelta(days=30)
        elif time_period_com == "3M": start_d_c = end_d_c - timedelta(days=90)
        elif time_period_com == "6M": start_d_c = end_d_c - timedelta(days=180)
        elif time_period_com == "YTD": start_d_c = datetime(end_d_c.year, 1, 1)
        elif time_period_com == "1Y": start_d_c = end_d_c - timedelta(days=365)
        elif time_period_com == "5Y": start_d_c = end_d_c - timedelta(days=365*5)
        elif time_period_com == "10Y": start_d_c = end_d_c - timedelta(days=365*10)
        elif time_period_com == "15Y": start_d_c = end_d_c - timedelta(days=365*15)
        elif time_period_com == "20Y": start_d_c = end_d_c - timedelta(days=365*20)
        else: start_d_c = end_d_c - timedelta(days=365)
        
        dates_c = st.date_input("Date Range", value=(start_d_c, end_d_c), key="com_date", label_visibility="collapsed")
        if isinstance(dates_c, tuple) and len(dates_c) == 2: s_date_c, e_date_c = dates_c
        else: s_date_c, e_date_c = start_d_c, end_d_c

    com_list = [t.strip().upper() for t in com_input.split(',') if t.strip()]

    if com_list:
        try:
            with st.spinner("Loading data..."):
                com_df = yf.download(com_list, start=s_date_c, end=e_date_c, progress=False)['Close']
                com_df.index = com_df.index.tz_localize(None)
                if isinstance(com_df, pd.Series): com_df = com_df.to_frame(name=com_list[0])
                com_df = com_df.ffill().dropna()

                if not com_df.empty:
                    daily_rets = com_df.pct_change().dropna()
                    total_perf = (com_df.iloc[-1] / com_df.iloc[0]) - 1
                    best_asset = total_perf.idxmax()
                    best_val = total_perf.max()
                    worst_asset = total_perf.idxmin()
                    worst_val = total_perf.min()
                    mean_d = daily_rets.mean()
                    vol_ann = daily_rets.std() * (252**0.5)
                    max_d = daily_rets.max()
                    min_d = daily_rets.min()

                    # KPIs
                    col_best, col_worst, col_chart = st.columns([1, 1, 4])
                    with col_best:
                        st.markdown(f"""<div style="background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; text-align: center;"><div style="color: #8b949e; font-size: 12px; font-weight: 600;">Best Commodity</div><div style="color: #f0f6fc; font-size: 28px; font-weight: 700; margin: 5px 0;">{best_asset[:4]}</div><div style="background-color: rgba(63, 185, 80, 0.15); color: #3fb950; padding: 2px 8px; border-radius: 12px; font-size: 14px; font-weight: 600; display: inline-block;">↑ {best_val*100:.0f}%</div></div>""", unsafe_allow_html=True)
                    with col_worst:
                        st.markdown(f"""<div style="background-color: #161b22; padding: 15px; border-radius: 8px; border: 1px solid #30363d; text-align: center;"><div style="color: #8b949e; font-size: 12px; font-weight: 600;">Worst Commodity</div><div style="color: #f0f6fc; font-size: 28px; font-weight: 700; margin: 5px 0;">{worst_asset[:4]}</div><div style="background-color: rgba(248, 81, 73, 0.15); color: #f85149; padding: 2px 8px; border-radius: 12px; font-size: 14px; font-weight: 600; display: inline-block;">↓ {worst_val*100:.0f}%</div></div>""", unsafe_allow_html=True)
                    with col_chart:
                        norm_df = (com_df / com_df.iloc[0]) * 100
                        fig_c = go.Figure()
                        colors = px.colors.qualitative.Plotly
                        for i, col in enumerate(norm_df.columns):
                            fig_c.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], name=col, line=dict(width=2)))
                        fig_c.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20), hovermode="x unified", legend=dict(orientation="h", y=1.1))
                        st.plotly_chart(fig_c, use_container_width=True, key="com_main_chart")

                    # METRICS GRID
                    st.write("")
                    st.write("")
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    def style_metric(label, value, sub_label):
                        return f"""<div style="text-align: left;"><div style="color: #f0f6fc; font-size: 14px; font-weight: 600;">{label}</div><div style="color: #f0f6fc; font-size: 24px; font-weight: 700;">{value}</div><div style="color: #8b949e; font-size: 12px; font-weight: 600;">{sub_label}</div></div>"""
                    with m1: st.markdown(style_metric("Best Mean (Daily)", f"{mean_d.max()*100:+.2f}%", mean_d.idxmax()), unsafe_allow_html=True)
                    with m2: st.markdown(style_metric("Worst Mean (Daily)", f"{mean_d.min()*100:+.2f}%", mean_d.idxmin()), unsafe_allow_html=True)
                    with m3: st.markdown(style_metric("Highest Vol (Ann.)", f"{vol_ann.max()*100:.2f}%", vol_ann.idxmax()), unsafe_allow_html=True)
                    with m4: st.markdown(style_metric("Lowest Vol (Ann.)", f"{vol_ann.min()*100:.2f}%", vol_ann.idxmin()), unsafe_allow_html=True)
                    with m5: st.markdown(style_metric("Max Daily Return", f"{max_d.max()*100:+.2f}%", max_d.idxmax()), unsafe_allow_html=True)
                    with m6: st.markdown(style_metric("Min Daily Return", f"{min_d.min()*100:+.2f}%", min_d.idxmin()), unsafe_allow_html=True)

                    # HISTOGRAMS
                    st.write("---")
                    c_h_cols = st.columns(min(len(com_df.columns), 4))
                    for i, tick in enumerate(com_df.columns[:4]):
                        with c_h_cols[i]:
                            rets = daily_rets[tick]
                            f_h = go.Figure()
                            f_h.add_trace(go.Histogram(x=rets, nbinsx=40, histnorm='probability density', marker_color='rgba(255,255,255,0.7)'))
                            x_range = np.linspace(rets.min(), rets.max(), 100)
                            pdf = norm.pdf(x_range, rets.mean(), rets.std())
                            f_h.add_trace(go.Scatter(x=x_range, y=pdf, mode='lines', line=dict(color='#2962FF', width=2)))
                            f_h.update_layout(title=tick, template="plotly_dark", height=150, showlegend=False, margin=dict(l=10,r=10,t=30,b=10), xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
                            st.plotly_chart(f_h, use_container_width=True, key=f"com_hist_{i}")

                    # TEXT ANALYSIS
                    avg_perf = total_perf.mean() * 100
                    corr_txt = "N/A"
                    if len(com_df.columns) > 1:
                        corr_mat = daily_rets.corr()
                        np.fill_diagonal(corr_mat.values, np.nan)
                        min_c = corr_mat.min().min()
                        max_c = corr_mat.max().max()
                        corr_txt = f"Cross-commodity correlation spans <span class='analysis-highlight'>{min_c:.2f}</span> to <span class='analysis-highlight'>{max_c:.2f}</span>"

                    analysis_html_com = f"""
                    <div class="analysis-text">
                        <b>Commodity recap ({time_period_com}).</b> 
                        Average performance across the selection is <span class='analysis-highlight'>{avg_perf:+.0f}%</span>. 
                        Best performer: <span class='analysis-highlight'>{best_asset} ({best_val*100:+.0f}%)</span>. 
                        Worst performer: <span class='analysis-highlight'>{worst_asset} ({worst_val*100:+.0f}%)</span>. 
                        Realized volatility (annualized) ranges from <span class='analysis-highlight'>{vol_ann.min()*100:.1f}%</span> ({vol_ann.idxmin()}) 
                        to <span class='analysis-highlight'>{vol_ann.max()*100:.1f}%</span> ({vol_ann.idxmax()}). 
                        {corr_txt}; diversification benefits improve as correlation decreases.
                    </div>
                    """
                    st.markdown(analysis_html_com, unsafe_allow_html=True)
                else: st.warning("No data found for these tickers.")
        except Exception as e: st.error(f"Error: {e}")

# -----------------------------------------------------------------------------
# TAB 5: TICKER REFERENCE GUIDE
# -----------------------------------------------------------------------------
with tab_tickers:
    st.subheader("Ticker Reference Guide")
    st.write("Use these tickers in the input fields above to fetch data from Yahoo Finance.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # SECTION 1: INDICES
        with st.expander("Major Indices & Equities", expanded=True):
            st.markdown("""
            | Region | Name | Ticker |
            | :--- | :--- | :--- |
            | **US Indices** | S&P 500 | `^GSPC` |
            | | Nasdaq 100 | `^NDX` |
            | | Dow Jones | `^DJI` |
            | | Russell 2000 | `^RUT` |
            | | VIX (Volatility) | `^VIX` |
            | **Europe** | CAC 40 (France) | `^FCHI` |
            | | DAX (Germany) | `^GDAXI` |
            | | FTSE 100 (UK) | `^FTSE` |
            | | Euro Stoxx 50 | `^STOXX50E` |
            | | SMI (Switzerland) | `^SSMI` |
            | | IBEX 35 (Spain) | `^IBEX` |
            | | AEX (Netherlands) | `^AEX` |
            | **Asia** | Nikkei 225 (Japan) | `^N225` |
            | | Hang Seng (Hong Kong) | `^HSI` |
            | | Shanghai Composite | `000001.SS` |
            | | KOSPI (Korea) | `^KS11` |
            | | Nifty 50 (India) | `^NSEI` |
            """)
        
        # SECTION 2: STOCKS
        with st.expander("Single Stocks (Equity)", expanded=True):
            st.markdown("""
            | Market | Name | Ticker | Sector |
            | :--- | :--- | :--- | :--- |
            | **US** | Apple | `AAPL` | Technology |
            | | Microsoft | `MSFT` | Technology |
            | | NVIDIA | `NVDA` | Technology |
            | | Amazon | `AMZN` | Consumer Disc. |
            | | Tesla | `TSLA` | Consumer Disc. |
            | | JPMorgan | `JPM` | Financials |
            | | Exxon Mobil | `XOM` | Energy |
            | | Johnson & Johnson | `JNJ` | Healthcare |
            | | Procter & Gamble | `PG` | Consumer Staples |
            | | Visa | `V` | Financials |
            | **Europe** | LVMH | `MC.PA` | Consumer Disc. |
            | | TotalEnergies | `TTE.PA` | Energy |
            | | ASML | `ASML.AS` | Technology |
            | | SAP | `SAP.DE` | Technology |
            | | Novo Nordisk | `NOVO-B.CO` | Healthcare |
            | | Nestle | `NESN.SW` | Consumer Staples |
            | | Shell | `SHEL.L` | Energy |
            | | Siemens | `SIE.DE` | Industrials |
            | | Airbus | `AIR.PA` | Industrials |
            | | L'Oreal | `OR.PA` | Consumer Staples |
            | **Asia** | Toyota | `7203.T` | Consumer Disc. |
            | | Sony | `6758.T` | Technology |
            | | Samsung | `005930.KS` | Technology |
            | | Alibaba | `BABA` | Consumer Disc. |
            | | Tencent | `0700.HK` | Technology |
            | | TSMC | `TSM` | Technology |
            | | Mitsubishi UFJ | `8306.T` | Financials |
            | | SoftBank | `9984.T` | Technology |
            """)

    with col2:
        # SECTION 3: COMMODITIES & CRYPTO
        with st.expander("Commodities & Crypto", expanded=True):
            st.markdown("""
            | Sector | Name | Ticker |
            | :--- | :--- | :--- |
            | **Precious Metals** | Gold | `GC=F` |
            | | Silver | `SI=F` |
            | | Platinum | `PL=F` |
            | | Palladium | `PA=F` |
            | **Energy** | Crude Oil WTI | `CL=F` |
            | | Brent Crude | `BZ=F` |
            | | Natural Gas | `NG=F` |
            | | Heating Oil | `HO=F` |
            | **Industrial Metals** | Copper | `HG=F` |
            | | Aluminum | `ALI=F` |
            | **Agriculture** | Corn | `ZC=F` |
            | | Soybeans | `ZS=F` |
            | | Wheat | `ZW=F` |
            | | Sugar | `SB=F` |
            | | Coffee | `KC=F` |
            | | Cocoa | `CC=F` |
            | | Cotton | `CT=F` |
            | | Live Cattle | `LE=F` |
            | | Lean Hogs | `HE=F` |
            | | Lumber | `LBS=F` |
            | | Oat | `ZO=F` |
            | | Rough Rice | `ZR=F` |
            | **Crypto** | Bitcoin | `BTC-USD` |
            | | Ethereum | `ETH-USD` |
            | | Solana | `SOL-USD` |
            | | XRP | `XRP-USD` |
            | | Binance Coin | `BNB-USD` |
            | | Cardano | `ADA-USD` |
            """)
        
        # SECTION 4: CURRENCY ETFs
        with st.expander("Currency ETFs (For Options)", expanded=True):
            st.markdown("""
            | Currency | Ticker |
            | :--- | :--- |
            | **Euro** | `FXE` |
            | **Yen** | `FXY` |
            | **British Pound** | `FXB` |
            | **Canadian Dollar** | `FXC` |
            | **Swiss Franc** | `FXF` |
            | **Australian Dollar** | `FXA` |
            | **US Dollar Index** | `UUP` |
            """)
