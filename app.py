import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, date
import calendar
import io
import zipfile
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
    year, month = current_date.year, current_date.month
    last_day = calendar.monthrange(year, month)[1]
    last_thursday = date(year, month, last_day)
    while last_thursday.weekday() != 3: 
        last_thursday = last_thursday.replace(day=last_thursday.day - 1)
    
    dte = (last_thursday - current_date).days
    
    if dte < 0:
        month = 1 if month == 12 else month + 1
        year = year + 1 if month == 1 else year
        last_day = calendar.monthrange(year, month)[1]
        last_thursday = date(year, month, last_day)
        while last_thursday.weekday() != 3:
            last_thursday = last_thursday.replace(day=last_thursday.day - 1)
        dte = (last_thursday - current_date).days
        
    return dte

# ==========================================
# ROBUST CSV LOADER (ENCODING & CLEANING)
# ==========================================
def load_and_clean_csv(file_obj):
    """
    Safely reads CSVs across different encodings and strips hidden whitespace.
    Works for both downloaded BytesIO (Requests) and UploadedFiles (Streamlit).
    """
    if file_obj is None:
        return None
        
    if isinstance(file_obj, bytes):
        file_obj = io.BytesIO(file_obj)

    for encoding in ['utf-8', 'latin1', 'cp1252']:
        try:
            if hasattr(file_obj, 'seek'):
                file_obj.seek(0)
            df = pd.read_csv(file_obj, encoding=encoding)
            # Strip hidden whitespace from headers
            df.columns = df.columns.astype(str).str.strip()
            return df
        except Exception:
            continue
    return None

# ==========================================
# DATA FETCHERS (AUTO-DOWNLOAD)
# ==========================================
def fetch_nse_data(selected_date, report_type="MTF"):
    date_str = selected_date.strftime("%d%m%Y")
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml',
        'Referer': 'https://www.nseindia.com/all-reports'
    }
    
    if report_type == "MTF":
        target_url = f"https://nsearchives.nseindia.com/archives/equities/mtf/MTF_Disclosure_{date_str}.csv"
    elif report_type == "FNO":
        # The correct NSE FO Bhavcopy URL format (requires unzipping)
        month_str = selected_date.strftime("%b").upper()
        year = selected_date.strftime("%Y")
        day = selected_date.strftime("%d")
        target_url = f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{year}/{month_str}/fo{day}{month_str}{year}bhav.csv.zip"
    else: # BHAVCOPY
        target_url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_str}.csv"
        
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        response = session.get(target_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            if report_type == "FNO":
                # F&O data is zipped. Extract the CSV directly in memory.
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_filename = [name for name in z.namelist() if name.endswith('.csv')][0]
                    with z.open(csv_filename) as f:
                        return load_and_clean_csv(f.read()), None
            else:
                return load_and_clean_csv(response.content), None
                
        elif response.status_code == 404:
            return None, f"404: {report_type} report not uploaded by NSE yet."
        elif response.status_code == 403:
            return None, f"403: NSE blocked the download for {report_type}. Use manual upload."
        else:
            return None, f"Error {response.status_code} fetching {report_type}."
    except Exception as e:
        return None, f"Network error fetching {report_type}: {str(e)}"

# ==========================================
# ADVANCED RVOL ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def calculate_rvol(symbols):
    if not symbols: return pd.DataFrame(columns=['Symbol', 'RVOL'])
    try:
        yf_symbols = " ".join([f"{s}.NS" for s in symbols])
        hist = yf.download(yf_symbols, period="15d", progress=False)
        vol_df = hist['Volume']
        rvol_list = []
        
        if isinstance(vol_df, pd.DataFrame):
            for col in vol_df.columns:
                symbol = col.replace('.NS', '')
                avg_vol = vol_df[col].iloc[:-1].mean()
                curr_vol = vol_df[col].iloc[-1]
                rvol_list.append({'Symbol': symbol, 'RVOL': round(curr_vol / avg_vol, 2) if avg_vol > 0 else 0})
        else:
            symbol = symbols[0]
            avg_vol = vol_df.iloc[:-1].mean()
            curr_vol = vol_df.iloc[-1]
            rvol_list.append({'Symbol': symbol, 'RVOL': round(curr_vol / avg_vol, 2) if avg_vol > 0 else 0})
            
        return pd.DataFrame(rvol_list)
    except:
        return pd.DataFrame(columns=['Symbol', 'RVOL'])

# ==========================================
# CORE ENGINES
# ==========================================
def standardize_symbol(df):
    """
    Expands the search vocabulary to catch NSE's inconsistent column naming.
    """
    valid_names = [
        'symbol', 'stock_symbol', 'scrip_name', 'symbol_name', 
        'scrip', 'name of scrip', 'underlying', 'ticker'
    ]
    
    for col in df.columns:
        if col.strip().lower() in valid_names:
            df = df.rename(columns={col: 'Symbol'})
            break
            
    if 'Symbol' in df.columns: 
        df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
    return df

def process_mtf_data(df):
    df_analyzed = standardize_symbol(df.copy())
    
    # Fail-safe if symbol column cannot be found
    if 'Symbol' not in df_analyzed.columns:
        return None
        
    if 'Current_Day_Funded_Value' in df_analyzed.columns and 'Previous_Day_Funded_Value' in df_analyzed.columns:
        df_analyzed['Delta_MTF'] = df_analyzed['Current_Day_Funded_Value'] - df_analyzed['Previous_Day_Funded_Value']
    else:
        df_analyzed['Delta_MTF'] = df_analyzed.get('Funded_Value', 0)

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
    
    # Fail-safe if symbol column cannot be found
    if 'Symbol' not in df_analyzed.columns:
        found_cols = ", ".join(df.columns.tolist()[:5])
        return None, f"Could not locate Symbol column. The file contains: {found_cols}..."
        
    # Filter ONLY Stock Futures (FUTSTK) to remove options noise
    if 'INSTRUMENT' in df_analyzed.columns:
        df_analyzed = df_analyzed[df_analyzed['INSTRUMENT'] == 'FUTSTK']

    # Standardize OI & Price Columns from raw NSE FO Bhavcopy
    if 'CHG_IN_OI' in df_analyzed.columns:
        df_analyzed['Delta_OI'] = df_analyzed['CHG_IN_OI']
        
    if 'CLOSE' in df_analyzed.columns and 'OPEN' in df_analyzed.columns:
        df_analyzed['Price_Change_Pct'] = df_analyzed['CLOSE'] - df_analyzed['OPEN']

    # Keep existing fallbacks for third-party processed CSVs
    price_col = 'Price_Change_Pct' if 'Price_Change_Pct' in df_analyzed.columns else 'chng_in_price'
    oi_col = 'Delta_OI' if 'Delta_OI' in df_analyzed.columns else 'chng_in_OI'
    
    if price_col not in df_analyzed.columns: df_analyzed[price_col] = 0.0
    if oi_col not in df_analyzed.columns: df_analyzed[oi_col] = 0.0

    # Aggregate OI changes across all expiry months for each stock
    df_analyzed = df_analyzed.groupby('Symbol', as_index=False).sum(numeric_only=True)

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
    
    # Fail-safe if symbol column cannot be found
    if 'Symbol' not in df_pa.columns:
        return None, "Could not locate a recognizable Symbol/Ticker column in the Bhavcopy."
        
    req_cols = ['OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE']
    for col in req_cols:
        if col not in df_pa.columns: return df_pa, f"Missing {col}"

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
    active_sources = []
    missing_sources = []
    
    pa_processed, mtf_processed, fno_processed = None, None, None
    pa_err = None

    # Track available vs missing datasets and gracefully skip unreadable ones
    if pa_df is not None:
        pa_processed, pa_err = process_price_action(pa_df)
        if pa_processed is not None and not pa_err: 
            active_sources.append("Price Action & Delivery (Bhavcopy)")
        else: 
            missing_sources.append(f"Bhavcopy Error: {pa_err}")
    else:
        missing_sources.append("Price Action & Delivery (Bhavcopy)")

    if mtf_df is not None:
        mtf_processed = process_mtf_data(mtf_df)
        if mtf_processed is not None:
            active_sources.append("MTF Retail Leverage")
        else:
            missing_sources.append("MTF Error: Could not locate Symbol column")
    else:
        missing_sources.append("MTF Retail Leverage")

    if fno_df is not None:
        fno_processed, fno_err = process_fno_data(fno_df)
        if fno_processed is not None:
            active_sources.append("F&O Smart Money OI")
            oi_col = fno_err # When successful, the second return value is the oi_col name
        else:
            missing_sources.append(f"F&O Error: {fno_err}")
    else:
        missing_sources.append("F&O Smart Money OI")

    if not active_sources:
        return None, "No valid datasets available to process.", [], []

    # Initialize Base Dataframe Dynamically
    if pa_processed is not None:
        merged = pa_processed[['Symbol', 'CLOSE_PRICE', 'Gain_Holding_Score_%', 'Delivery_%']].copy()
    elif mtf_processed is not None:
        merged = mtf_processed[['Symbol', 'MTF_Scenario', 'Delta_MTF']].copy()
    else:
        merged = fno_processed[['Symbol', 'FnO_Scenario', oi_col]].copy()

    # Left Join Active Datasets
    if pa_processed is not None and 'CLOSE_PRICE' not in merged.columns:
        merged = pd.merge(merged, pa_processed[['Symbol', 'CLOSE_PRICE', 'Gain_Holding_Score_%', 'Delivery_%']], on='Symbol', how='left')
    
    if mtf_processed is not None and 'MTF_Scenario' not in merged.columns:
        merged = pd.merge(merged, mtf_processed[['Symbol', 'MTF_Scenario', 'Delta_MTF']], on='Symbol', how='left')
        
    if fno_processed is not None and 'FnO_Scenario' not in merged.columns:
        merged = pd.merge(merged, fno_processed[['Symbol', 'FnO_Scenario', oi_col]], on='Symbol', how='left')

    # Fill Missing columns with neutrals so logic won't fail
    for col in ['CLOSE_PRICE', 'Gain_Holding_Score_%', 'Delivery_%', 'Delta_MTF']:
        if col not in merged.columns: merged[col] = 0.0
    for col in ['MTF_Scenario', 'FnO_Scenario']:
        if col not in merged.columns: merged[col] = 'Data Missing'

    merged['Sector'] = merged['Symbol'].apply(get_sector)
    
    # Calculate RVOL
    rvol_df = calculate_rvol(merged['Symbol'].tolist())
    if not rvol_df.empty:
        merged = pd.merge(merged, rvol_df, on='Symbol', how='left').fillna(1.0)
    else:
        merged['RVOL'] = 1.0

    # Dynamic Scoring Engine
    has_pa = pa_processed is not None
    has_mtf = mtf_processed is not None
    has_fno = fno_processed is not None

    def eval_signal(row):
        active_bull = 0
        active_bear = 0
        total_active = sum([has_pa, has_mtf, has_fno])

        # Evaluate what we have
        if has_pa:
            if row['Gain_Holding_Score_%'] >= 80: active_bull += 1
            elif row['Gain_Holding_Score_%'] <= 20: active_bear += 1
        
        if has_mtf:
            if row['MTF_Scenario'] == 'Bullish Leverage Acceleration': active_bull += 1
            elif row['MTF_Scenario'] == 'Catching a Falling Knife (Danger)': active_bear += 1
            
        if has_fno:
            if row['FnO_Scenario'] == 'Long Buildup': active_bull += 1
            elif row['FnO_Scenario'] == 'Short Buildup': active_bear += 1

        # Full Engine Processing
        if has_pa and has_mtf and has_fno:
            bull = (row['MTF_Scenario'] == 'Bullish Leverage Acceleration') and (row['FnO_Scenario'] == 'Long Buildup') and (row['Gain_Holding_Score_%'] >= 80)
            bear = (row['FnO_Scenario'] == 'Short Buildup') and ((row['Gain_Holding_Score_%'] <= 20) or (row['MTF_Scenario'] == 'Catching a Falling Knife (Danger)'))
            
            if bull and row['Delivery_%'] >= 50.0 and row['RVOL'] >= 1.5: return '🌟 SUPREME BULL (Inst. Buy + Volume Spike)'
            elif bull and row['Delivery_%'] >= 50.0: return '🏛️ INSTITUTIONAL BULL (>50% Delivery)'
            elif bull and row['Delivery_%'] < 50.0: return '🎲 SPECULATIVE BULL (<50% Delivery)'
            elif bear: return '🚨 BEARISH CONVICTION (HIGH RISK)'
            return 'Neutral / Mixed'
            
        # Partial Engine Processing
        else:
            if active_bull == total_active and total_active > 0:
                if has_pa and row['Delivery_%'] >= 50.0: return '🏛️ Partial Conviction Bull (Inst. Delivery)'
                return '⚠️ Partial Conviction Bull (Incomplete Data)'
            elif active_bear == total_active and total_active > 0:
                return '🚨 Partial Bearish Conviction'
            elif active_bull > 0:
                return '🎲 Speculative Bull (Partial Data)'
            return 'Neutral / Missing Data'

    merged['Master_Signal'] = merged.apply(eval_signal, axis=1)
    
    return merged, None, active_sources, missing_sources

# ==========================================
# USER INTERFACE
# ==========================================
st.sidebar.title("🧭 The Quant Engine")
st.title("⚡ Master Confluence Engine 2.0")

dte = get_days_to_expiry(date.today())
if dte <= 4:
    st.warning(f"⚠️ **Expiry Week Alert:** Only {dte} days to F&O Expiry. Take 'Long Unwinding' and 'Short Covering' signals with caution.")

# THE NEW DATA INGESTION MENU
st.sidebar.header("📥 Data Input Method")
input_mode = st.sidebar.radio("Choose how to load data:", ["Auto-Fetch from NSE", "Manual Upload Backup"])

mtf_raw, fno_raw, pa_raw = None, None, None
data_ready = False

if input_mode == "Auto-Fetch from NSE":
    st.sidebar.markdown("Attempts to download files directly. *(Note: NSE sometimes blocks cloud servers).*")
    selected_date = st.sidebar.date_input("Select Trading Date", datetime.now())
    
    if st.sidebar.button("🚀 Fetch & Analyze", type="primary"):
        with st.spinner("Knocking on NSE's servers..."):
            mtf_raw, m_err = fetch_nse_data(selected_date, "MTF")
            fno_raw, f_err = fetch_nse_data(selected_date, "FNO")
            pa_raw, p_err = fetch_nse_data(selected_date, "BHAVCOPY")
            
            # Print non-fatal errors but allow partial execution
            if m_err: st.sidebar.error(m_err)
            if f_err: st.sidebar.error(f_err)
            if p_err: st.sidebar.error(p_err)
            
            if mtf_raw is not None or fno_raw is not None or pa_raw is not None:
                data_ready = True

elif input_mode == "Manual Upload Backup":
    st.sidebar.markdown("Use this if the Auto-Fetch fails due to NSE blocks.")
    mtf_file = st.sidebar.file_uploader("1. MTF Disclosure", type=["csv"])
    fno_file = st.sidebar.file_uploader("2. F&O Participant OI", type=["csv"])
    pa_file = st.sidebar.file_uploader("3. EOD Bhavcopy", type=["csv"])
    
    if mtf_file or fno_file or pa_file:
        mtf_raw = load_and_clean_csv(mtf_file)
        fno_raw = load_and_clean_csv(fno_file)
        pa_raw = load_and_clean_csv(pa_file)
        data_ready = True

# PROCESS DATA IF READY
if data_ready:
    try:
        with st.spinner("Crunching data, fetching RVOL from Yahoo Finance, and mapping sectors..."):
            confluence_df, error, active_srcs, missing_srcs = calculate_master_confluence(mtf_raw, fno_raw, pa_raw)
        
        if error:
            st.error(error)
        else:
            # RENDER DYNAMIC DISCLAIMER BANNER
            if missing_srcs:
                st.warning(
                    f"⚠️ **PARTIAL DATA MODE ACTIVE**\n\n"
                    f"* **Active Datasets:** {', '.join(active_srcs)}\n"
                    f"* **Missing/Skipped Datasets:** {', '.join(missing_srcs)}\n\n"
                    f"*Note: Confluence conviction scores are dynamically calculated using available parameters. "
                    f"Tiers requiring missing streams have been downgraded or marked partial.*"
                )
            else:
                st.success("✅ **FULL CONFLUENCE MODE:** All 3 data streams active.")

            # Metrics
            supreme = confluence_df[confluence_df['Master_Signal'].str.contains('SUPREME BULL')]
            inst = confluence_df[confluence_df['Master_Signal'].str.contains('INSTITUTIONAL BULL|Partial Conviction Bull')]
            bears = confluence_df[confluence_df['Master_Signal'].str.contains('BEARISH CONVICTION|Partial Bearish')]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("🌟 Supreme Conviction Trades", len(supreme))
            c2.metric("🏛️ Institutional Buys", len(inst))
            c3.metric("🚨 High Risk Shorts", len(bears))
            
            st.divider()
            t1, t2, t3, t4 = st.tabs(["🎯 Top Signals", "📊 Visual Screener (Chart)", "🗺️ Sector Heatmap", "📋 Database"])
            
            with t1:
                st.subheader("High Conviction Breakouts")
                display_cols = ['Symbol', 'Sector', 'CLOSE_PRICE', 'Delivery_%', 'Gain_Holding_Score_%', 'RVOL', 'Master_Signal']
                
                # Filter down to just what we need, ignoring columns that don't exist if data is missing
                available_display_cols = [c for c in display_cols if c in confluence_df.columns]
                combined_bulls = pd.concat([supreme, inst]).sort_values(by=['RVOL', 'Delivery_%'], ascending=[False, False])
                st.dataframe(combined_bulls[available_display_cols], use_container_width=True)
                
            with t2:
                st.subheader("Institutional Quadrant Analysis")
                st.markdown("Look for dots in the **Top-Right** (High Delivery + Held Intraday Gains). Size of dot = RVOL.")
                
                fig = px.scatter(
                    confluence_df[~confluence_df['Master_Signal'].str.contains('Neutral')],
                    x="Delivery_%", y="Gain_Holding_Score_%",
                    color="Master_Signal", size="RVOL", hover_name="Symbol",
                    hover_data=["Sector", "CLOSE_PRICE"],
                    color_discrete_map={
                        '🌟 SUPREME BULL (Inst. Buy + Volume Spike)': '#00FF00',
                        '🏛️ INSTITUTIONAL BULL (>50% Delivery)': '#008000',
                        '🏛️ Partial Conviction Bull (Inst. Delivery)': '#7CFC00',
                        '⚠️ Partial Conviction Bull (Incomplete Data)': '#9ACD32',
                        '🎲 SPECULATIVE BULL (<50% Delivery)': '#FFA500',
                        '🎲 Speculative Bull (Partial Data)': '#FFD700',
                        '🚨 BEARISH CONVICTION (HIGH RISK)': '#FF0000',
                        '🚨 Partial Bearish Conviction': '#CD5C5C'
                    },
                    template="plotly_dark", height=600
                )
                fig.add_hline(y=80, line_dash="dash", line_color="gray", annotation_text="Strong Close")
                fig.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="Inst. Delivery")
                st.plotly_chart(fig, use_container_width=True)

            with t3:
                st.subheader("Sectoral Heatmap (Where is the money flowing?)")
                bulls_only = confluence_df[confluence_df['Master_Signal'].str.contains('Bull|BULL', case=False)]
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
elif not data_ready and input_mode == "Manual Upload Backup":
    st.info("Upload at least ONE dataset (MTF, F&O, or Bhavcopy) to initiate the Quant Engine.")
else:
    st.info("Select a date and click 'Fetch & Analyze' to run the engine.")
