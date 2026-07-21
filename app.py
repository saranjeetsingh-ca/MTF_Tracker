import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date
import calendar
import io
import yfinance as yf
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Market Analytics Pro",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# SECTOR MAPPING & CALENDAR HELPERS
# ==========================================
# A core mapping of major NSE stocks to identify Sectoral Cash Flows
SECTOR_MAP = {
    'HDFCBANK': 'Banking', 'ICICIBANK': 'Banking', 'SBIN': 'Banking', 'AXISBANK': 'Banking', 'KOTAKBANK': 'Banking',
    'TCS': 'IT', 'INFY': 'IT', 'HCLTECH': 'IT', 'WIPRO': 'IT', 'TECHM': 'IT',
    'RELIANCE': 'Energy', 'ONGC': 'Energy', 'NTPC': 'Energy', 'POWERGRID': 'Energy',
    'TATAMOTORS': 'Auto', 'M&M': 'Auto', 'MARUTI': 'Auto', 'BAJAJ-AUTO': 'Auto', 'HEROMOTOCO': 'Auto',
    'SUNPHARMA': 'Pharma', 'DRREDDY': 'Pharma', 'CIPLA': 'Pharma', 'DIVISLAB': 'Pharma',
    'TATASTEEL': 'Metals', 'JSWSTEEL': 'Metals', 'HINDALCO': 'Metals', 'VEDL': 'Metals',
    'ITC': 'FMCG', 'HUL': 'FMCG', 'NESTLEIND': 'FMCG', 'BRITANNIA': 'FMCG'
}

def get_sector(symbol):
    return SECTOR_MAP.get(symbol.upper(), 'Other / Broader Market')

def get_days_to_expiry(current_date):
    """Calculates days remaining until the last Thursday of the current month."""
    year = current_date.year
    month = current_date.month
    # Find the last day of the month
    last_day = calendar.monthrange(year, month)[1]
    # Find the last Thursday
    last_thursday = date(year, month, last_day)
    while last_thursday.weekday() != 3: # 3 is Thursday
        last_thursday = last_thursday.replace(day=last_thursday.day - 1)
    
    dte = (last_thursday - current_date).days
    
    # If the last Thursday has already passed, calculate for next month
    if dte < 0:
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
        last_day = calendar.monthrange(year, month)[1]
        last_thursday = date(year, month, last_day)
        while last_thursday.weekday() != 3:
            last_thursday = last_thursday.replace(day=last_thursday.day - 1)
        dte = (last_thursday - current_date).days
        
    return dte

# ==========================================
# ADVANCED RVOL (RELATIVE VOLUME) ENGINE
# ==========================================
@st.cache_data(ttl=3600) # Cache the YFinance download to avoid API limits
def calculate_rvol(symbols):
    """Fetches 15 days of history to calculate RVOL (Today's Vol / 10-Day Avg Vol)."""
    if not symbols:
        return pd.DataFrame(columns=['Symbol', 'RVOL'])
    
    try:
        yf_symbols = " ".join([f"{s}.NS" for s in symbols])
        hist = yf.download(yf_symbols, period="15d", progress=False)
        
        # yfinance returns different structures if 1 symbol vs multiple
        vol_df = hist['Volume']
        rvol_list = []
        
        if isinstance(vol_df, pd.DataFrame): # Multiple symbols
            for col in vol_df.columns:
                symbol = col.replace('.NS', '')
                avg_vol = vol_df[col].iloc[:-1].mean() # Average of previous days
                curr_vol = vol_df[col].iloc[-1]        # Most recent day
                rvol = round(curr_vol / avg_vol, 2) if avg_vol > 0 else 0
                rvol_list.append({'Symbol': symbol, 'RVOL': rvol})
        else: # Single symbol
            symbol = symbols[0]
            avg_vol = vol_df.iloc[:-1].mean()
            curr_vol = vol_df.iloc[-1]
            rvol = round(curr_vol / avg_vol, 2) if avg_vol > 0 else 0
            rvol_list.append({'Symbol': symbol, 'RVOL': rvol})
            
        return pd.DataFrame(rvol_list)
    except Exception as e:
        return pd.DataFrame(columns=['Symbol', 'RVOL'])

# ==========================================
# DATA STANDARDIZATION & CORE ENGINES
# ==========================================
def standardize_symbol(df):
    for col in df.columns:
        if col.strip().lower() in ['symbol', 'stock_symbol', 'scrip_name', 'symbol_name']:
            df = df.rename(columns={col: 'Symbol'})
            break
    if 'Symbol' in df.columns:
        df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
    return df

def process_mtf_data(df):
    df_analyzed = standardize_symbol(df.copy())
    
    # Calculate MTF Delta
    if 'Current_Day_Funded_Value' in df_analyzed.columns and 'Previous_Day_Funded_Value' in df_analyzed.columns:
        df_analyzed['Delta_MTF'] = df_analyzed['Current_Day_Funded_Value'] - df_analyzed['Previous_Day_Funded_Value']
    else:
        df_analyzed['Delta_MTF'] = df_analyzed.get('Funded_Value', 0)

    # Classify MTF Sentiments
    if 'Price_Change_Pct' in df_analyzed.columns:
        conditions = [
            (df_analyzed['Price_Change_Pct'] > 0) & (df_analyzed['Delta_MTF'] > 0),
            (df_analyzed['Price_Change_Pct'] > 0) & (df_analyzed['Delta_MTF'] < 0),
            (df_analyzed['Price_Change_Pct'] < 0) & (df_analyzed['Delta_MTF'] > 0),
            (df_analyzed['Price_Change_Pct'] < 0) & (df_analyzed['Delta_MTF'] < 0)
        ]
        choices = ['Bullish Leverage Acceleration', 'Short-Covering / Profit Booking', 'Catching a Falling Knife (Danger)', 'Long Unwinding / Capitulation']
        df_analyzed['MTF_Scenario'] = np.select(conditions, choices, default='Neutral')
    else:
        df_analyzed['MTF_Scenario'] = 'No Price Data'
    return df_analyzed

def process_fno_data(df):
    df_analyzed = standardize_symbol(df.copy())
    price_col = 'Price_Change_Pct' if 'Price_Change_Pct' in df_analyzed.columns else 'chng_in_price'
    oi_col = 'Delta_OI' if 'Delta_OI' in df_analyzed.columns else 'chng_in_OI'
    
    if price_col not in df_analyzed.columns: df_analyzed[price_col] = 0.0
    if oi_col not in df_analyzed.columns: df_analyzed[oi_col] = 0.0

    conditions = [
        (df_analyzed[price_col] > 0) & (df_analyzed[oi_col] > 0),
        (df_analyzed[price_col] > 0) & (df_analyzed[oi_col] < 0),
        (df_analyzed[price_col] < 0) & (df_analyzed[oi_col] > 0),
        (df_analyzed[price_col] < 0) & (df_analyzed[oi_col] < 0)
    ]
    choices = ['Long Buildup', 'Short Covering', 'Short Buildup', 'Long Unwinding']
    df_analyzed['FnO_Scenario'] = np.select(conditions, choices, default='Neutral')
    return df_analyzed, oi_col

def process_price_action(df):
    df_pa = standardize_symbol(df.copy())
    req_cols = ['OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE']
    for col in req_cols:
        if col not in df_pa.columns: return df_pa, f"Missing {col}"

    # Extract Delivery
    deliv_col = next((c for c in df_pa.columns if 'DELIV' in c and 'PER' in c), None)
    if deliv_col:
        df_pa['Delivery_%'] = pd.to_numeric(df_pa[deliv_col].astype(str).str.replace('-', '0'), errors='coerce').fillna(0)
    else:
        df_pa['Delivery_%'] = 0.0

    df_pa = df_pa[df_pa['CLOSE_PRICE'] > 10]
    O, H, L, C = df_pa['OPEN_PRICE'], df_pa['HIGH_PRICE'], df_pa['LOW_PRICE'], df_pa['CLOSE_PRICE']
    
    candle_range = np.where((H - L) == 0, 0.0001, H - L)
    df_pa['Gain_Holding_Score_%'] = (((C - L) / candle_range) * 100).round(2)
    
    return df_pa, None

def calculate_master_confluence(mtf_df, fno_df, pa_df):
    mtf_processed = process_mtf_data(mtf_df)
    fno_processed, oi_col = process_fno_data(fno_df)
    pa_processed, pa_err = process_price_action(pa_df)
    
    if pa_err: return None, pa_err

    # 1. Merge the 3 datasets
    merged = pd.merge(mtf_processed[['Symbol', 'MTF_Scenario', 'Delta_MTF']], fno_processed[['Symbol', 'FnO_Scenario', oi_col]], on='Symbol', how='inner')
    merged = pd.merge(merged, pa_processed[['Symbol', 'CLOSE_PRICE', 'Gain_Holding_Score_%', 'Delivery_%']], on='Symbol', how='inner')
    
    # 2. Add Sector
    merged['Sector'] = merged['Symbol'].apply(get_sector)
    
    # 3. Add RVOL (Relative Volume) via YFinance API
    rvol_df = calculate_rvol(merged['Symbol'].tolist())
    if not rvol_df.empty:
        merged = pd.merge(merged, rvol_df, on='Symbol', how='left').fillna(1.0)
    else:
        merged['RVOL'] = 1.0

    # 4. Master Confluence Logic
    is_bullish = (merged['MTF_Scenario'] == 'Bullish Leverage Acceleration') & (merged['FnO_Scenario'] == 'Long Buildup') & (merged['Gain_Holding_Score_%'] >= 80)
    is_bearish = (merged['FnO_Scenario'] == 'Short Buildup') & ((merged['Gain_Holding_Score_%'] <= 20) | (merged['MTF_Scenario'] == 'Catching a Falling Knife (Danger)'))
    
    conditions = [
        (is_bullish) & (merged['Delivery_%'] >= 50.0) & (merged['RVOL'] >= 1.5), # Supreme Conviction
        (is_bullish) & (merged['Delivery_%'] >= 50.0), # Standard Institutional
        (is_bullish) & (merged['Delivery_%'] < 50.0),  # Speculative
        (is_bearish)
    ]
    choices = [
        '🌟 SUPREME BULL (Inst. Buy + Volume Spike)',
        '🏛️ INSTITUTIONAL BULL (>50% Delivery)',
        '🎲 SPECULATIVE BULL (<50% Delivery)',
        '🚨 BEARISH CONVICTION (HIGH RISK)'
    ]
    merged['Master_Signal'] = np.select(conditions, choices, default='Neutral / Mixed')
    
    return merged, None

# ==========================================
# USER INTERFACE
# ==========================================
st.sidebar.title("🧭 The Quant Engine")
st.title("⚡ Master Confluence Engine 2.0")
st.markdown("Fusing Cash Leverage, Derivative OI, Price Action, RVOL, and Sectoral flows.")

# Check Expiry Context
today = date.today()
dte = get_days_to_expiry(today)
if dte <= 4:
    st.warning(f"⚠️ **Expiry Week Alert:** Only {dte} days to F&O Expiry. Take 'Long Unwinding' and 'Short Covering' signals with caution, as institutions are likely just rolling over contracts rather than reversing trends.")

st.sidebar.header("📁 Upload EOD Data")
mtf_file = st.sidebar.file_uploader("1. MTF Disclosure", type=["csv"])
fno_file = st.sidebar.file_uploader("2. F&O Participant OI", type=["csv"])
pa_file = st.sidebar.file_uploader("3. EOD Bhavcopy", type=["csv"])

if mtf_file and fno_file and pa_file:
    try:
        mtf_raw, fno_raw, pa_raw = pd.read_csv(mtf_file), pd.read_csv(fno_file), pd.read_csv(pa_file)
        
        with st.spinner("Crunching data, fetching RVOL from Yahoo Finance, and mapping sectors..."):
            confluence_df, error = calculate_master_confluence(mtf_raw, fno_raw, pa_raw)
        
        if error:
            st.error(error)
        else:
            supreme = confluence_df[confluence_df['Master_Signal'] == '🌟 SUPREME BULL (Inst. Buy + Volume Spike)']
            inst = confluence_df[confluence_df['Master_Signal'] == '🏛️ INSTITUTIONAL BULL (>50% Delivery)']
            
            c1, c2, c3 = st.columns(3)
            c1.metric("🌟 Supreme Conviction Trades", len(supreme))
            c2.metric("🏛️ Standard Institutional Buys", len(inst))
            c3.metric("🚨 High Risk Shorts", len(confluence_df[confluence_df['Master_Signal'] == '🚨 BEARISH CONVICTION (HIGH RISK)']))
            
            st.divider()
            
            # DASHBOARD TABS
            t1, t2, t3, t4 = st.tabs(["🎯 Top Signals", "📊 Visual Screener (Chart)", "🗺️ Sector Heatmap", "📋 Database"])
            
            with t1:
                st.subheader("High Conviction Breakouts")
                display_cols = ['Symbol', 'Sector', 'CLOSE_PRICE', 'Delivery_%', 'Gain_Holding_Score_%', 'RVOL', 'Master_Signal']
                combined_bulls = pd.concat([supreme, inst]).sort_values(by=['RVOL', 'Delivery_%'], ascending=[False, False])
                st.dataframe(combined_bulls[display_cols], use_container_width=True)
                
            with t2:
                st.subheader("Institutional Quadrant Analysis")
                st.markdown("Look for dots in the **Top-Right** (High Delivery + Held Intraday Gains). Size of dot = RVOL.")
                
                # Plotly Interactive Scatter Chart
                fig = px.scatter(
                    confluence_df[confluence_df['Master_Signal'] != 'Neutral / Mixed'],
                    x="Delivery_%", y="Gain_Holding_Score_%",
                    color="Master_Signal", size="RVOL", hover_name="Symbol",
                    hover_data=["Sector", "CLOSE_PRICE"],
                    color_discrete_map={
                        '🌟 SUPREME BULL (Inst. Buy + Volume Spike)': '#00FF00',
                        '🏛️ INSTITUTIONAL BULL (>50% Delivery)': '#008000',
                        '🎲 SPECULATIVE BULL (<50% Delivery)': '#FFA500',
                        '🚨 BEARISH CONVICTION (HIGH RISK)': '#FF0000'
                    },
                    template="plotly_dark", height=600
                )
                fig.add_hline(y=80, line_dash="dash", line_color="gray", annotation_text="Strong Close Line")
                fig.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="Inst. Delivery Line")
                st.plotly_chart(fig, use_container_width=True)

            with t3:
                st.subheader("Sectoral Heatmap (Where is the money flowing?)")
                # Group by Sector to see which sectors have the most Bullish signals today
                bulls_only = confluence_df[confluence_df['Master_Signal'].str.contains('BULL')]
                if not bulls_only.empty:
                    sector_flow = bulls_only.groupby('Sector').size().reset_index(name='Bullish_Signals').sort_values(by='Bullish_Signals', ascending=False)
                    st.bar_chart(sector_flow.set_index('Sector'))
                else:
                    st.info("No clear sector trends today.")

            with t4:
                st.dataframe(confluence_df, use_container_width=True)
                st.download_button("📥 Download Database", confluence_df.to_csv(index=False).encode('utf-8'), "Quant_Master_Report.csv", "text/csv")
                
    except Exception as e:
        st.error(f"Execution Error: {e}")
else:
    st.info("Upload MTF, F&O, and Bhavcopy CSVs to initiate the Quant Engine.")
