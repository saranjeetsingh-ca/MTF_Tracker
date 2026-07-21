import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import io

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="MTF Leverage & Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# CORE ANALYTICS ENGINE
# ==========================================
def process_mtf_data(df):
    """
    Processes raw or enriched MTF data, calculates daily changes, 
    and classifies stocks into the 4 main MTF trend scenarios.
    """
    df_analyzed = df.copy()
    
    # Standardize column names (case-insensitive & whitespace cleanup)
    df_analyzed.columns = [col.strip() for col in df_analyzed.columns]
    
    # 1. Delta MTF Calculation
    if 'Current_Day_Funded_Value' in df_analyzed.columns and 'Previous_Day_Funded_Value' in df_analyzed.columns:
        df_analyzed['Delta_MTF'] = df_analyzed['Current_Day_Funded_Value'] - df_analyzed['Previous_Day_Funded_Value']
    elif 'Funded_Value' in df_analyzed.columns:
        df_analyzed['Delta_MTF'] = df_analyzed['Funded_Value']
    else:
        df_analyzed['Delta_MTF'] = 0

    # 2. Leverage Intensity Ratio
    if 'Total_Daily_Delivery_Value' in df_analyzed.columns and 'Daily_Fresh_MTF_Capital_Added' in df_analyzed.columns:
        df_analyzed['Leverage_Intensity_Ratio'] = np.where(
            df_analyzed['Total_Daily_Delivery_Value'] > 0,
            (df_analyzed['Daily_Fresh_MTF_Capital_Added'] / df_analyzed['Total_Daily_Delivery_Value']) * 100,
            0
        )
    else:
        df_analyzed['Leverage_Intensity_Ratio'] = 0.0

    # 3. Categorize into 4 MTF Trend Matrix Scenarios
    if 'Price_Change_Pct' in df_analyzed.columns:
        conditions = [
            (df_analyzed['Price_Change_Pct'] > 0) & (df_analyzed['Delta_MTF'] > 0),
            (df_analyzed['Price_Change_Pct'] > 0) & (df_analyzed['Delta_MTF'] < 0),
            (df_analyzed['Price_Change_Pct'] < 0) & (df_analyzed['Delta_MTF'] > 0),
            (df_analyzed['Price_Change_Pct'] < 0) & (df_analyzed['Delta_MTF'] < 0)
        ]
        choices = [
            'Bullish Leverage Acceleration',
            'Short-Covering / Profit Booking',
            'Catching a Falling Knife (Danger)',
            'Long Unwinding / Capitulation'
        ]
        df_analyzed['MTF_Scenario'] = np.select(conditions, choices, default='Neutral / Unchanged')
    else:
        df_analyzed['MTF_Scenario'] = 'Price Change Data Missing'

    # 4. Critical Warning Flags
    if 'Free_Float_Market_Cap' in df_analyzed.columns and 'Current_Day_Funded_Value' in df_analyzed.columns:
        df_analyzed['Over_Leveraged_Warning'] = (df_analyzed['Current_Day_Funded_Value'] / df_analyzed['Free_Float_Market_Cap']) > 0.025
    else:
        df_analyzed['Over_Leveraged_Warning'] = False

    return df_analyzed


def fetch_nse_mtf_data(selected_date):
    """
    Attempts to download raw Margin Trading Disclosure CSV directly from NSE archives.
    """
    date_str = selected_date.strftime("%d%m%Y")
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
        'Referer': 'https://www.nseindia.com/all-reports'
    }
    
    target_url = f"https://nsearchives.nseindia.com/archives/equities/mtf/MTF_Disclosure_{date_str}.csv"
    
    try:
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        response = session.get(target_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            # Read CSV content from response bytes
            df = pd.read_csv(io.BytesIO(response.content))
            return df, None
        elif response.status_code == 404:
            return None, "Error 404: NSE report for this date is not available yet."
        elif response.status_code == 403:
            return None, "Error 403: Access restricted by NSE firewall. Try uploading a CSV manually."
        else:
            return None, f"HTTP Error {response.status_code} while fetching from NSE."
    except Exception as e:
        return None, f"Network error: {str(e)}"

# ==========================================
# USER INTERFACE (STREAMLIT)
# ==========================================

st.title("📊 Daily Margin Trading Facility (MTF) Dashboard")
st.markdown("Track leveraged retail conviction, detect sentiment shifts, and spot margin trap risks across Indian equities.")

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("⚙️ Data Source Controls")

data_source_mode = st.sidebar.radio(
    "Select Data Input Method:",
    ["Fetch Live from NSE", "Upload Custom CSV / ScanX File"]
)

raw_df = None

if data_source_mode == "Fetch Live from NSE":
    selected_date = st.sidebar.date_input("Select Report Date", datetime.now())
    generate_btn = st.sidebar.button("🚀 Fetch & Generate Dashboard", type="primary")
    
    if generate_btn:
        with st.spinner("Connecting to NSE archives and fetching MTF disclosure..."):
            df_fetched, err_msg = fetch_nse_mtf_data(selected_date)
            if df_fetched is not None:
                st.session_state['data'] = df_fetched
                st.sidebar.success(f"Successfully loaded data for {selected_date.strftime('%d-%b-%Y')}!")
            else:
                st.sidebar.error(err_msg)

else:
    uploaded_file = st.sidebar.file_uploader("Upload MTF CSV File", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.session_state['data'] = raw_df
            st.sidebar.success("CSV file uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

# --- MAIN DASHBOARD BODY ---
if 'data' in st.session_state and st.session_state['data'] is not None:
    # Process the loaded data
    processed_df = process_mtf_data(st.session_state['data'])
    
    st.divider()
    
    # 1. SUMMARY METRICS ROW
    total_stocks = len(processed_df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    bullish_cnt = len(processed_df[processed_df['MTF_Scenario'] == 'Bullish Leverage Acceleration'])
    short_cov_cnt = len(processed_df[processed_df['MTF_Scenario'] == 'Short-Covering / Profit Booking'])
    falling_knife_cnt = len(processed_df[processed_df['MTF_Scenario'] == 'Catching a Falling Knife (Danger)'])
    unwinding_cnt = len(processed_df[processed_df['MTF_Scenario'] == 'Long Unwinding / Capitulation'])
    
    col1.metric("Total Stocks", f"{total_stocks:,}")
    col2.metric("🟢 Bullish Acceleration", f"{bullish_cnt}")
    col3.metric("🔵 Short-Covering", f"{short_cov_cnt}")
    col4.metric("🔴 Falling Knives (Risk)", f"{falling_knife_cnt}")
    col5.metric("🟡 Long Unwinding", f"{unwinding_cnt}")
    
    st.divider()
    
    # 2. TOP 10 STOCKS PER CATEGORY
    st.subheader("📌 Top 10 Stocks by Sentiment Category")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🟢 Bullish Leverage Acceleration", 
        "🔵 Short-Covering / Profit Booking", 
        "🔴 Catching a Falling Knife", 
        "🟡 Long Unwinding"
    ])
    
    # Sort helper to get top 10 by leverage velocity/change
    sort_column = 'Delta_MTF' if 'Delta_MTF' in processed_df.columns else processed_df.columns[0]
    
    with tab1:
        st.markdown("**Price Up + MTF Up:** Strong upward momentum backed by aggressive leveraged buying.")
        cat1_df = processed_df[processed_df['MTF_Scenario'] == 'Bullish Leverage Acceleration']
        top10_cat1 = cat1_df.sort_values(by=sort_column, ascending=False).head(10)
        st.dataframe(top10_cat1, use_container_width=True)
        
    with tab2:
        st.markdown("**Price Up + MTF Down:** Price rising while leveraged traders exit. Loss of institutional/leveraged support.")
        cat2_df = processed_df[processed_df['MTF_Scenario'] == 'Short-Covering / Profit Booking']
        top10_cat2 = cat2_df.sort_values(by=sort_column, ascending=True).head(10)
        st.dataframe(top10_cat2, use_container_width=True)

    with tab3:
        st.markdown("**Price Down + MTF Up:** High risk! Retail is adding margin to falling positions. Liquidation trap.")
        cat3_df = processed_df[processed_df['MTF_Scenario'] == 'Catching a Falling Knife (Danger)']
        top10_cat3 = cat3_df.sort_values(by=sort_column, ascending=False).head(10)
        st.dataframe(top10_cat3, use_container_width=True)

    with tab4:
        st.markdown("**Price Down + MTF Down:** Weak hands being forced out. Healthy correction or potential structural bottom.")
        cat4_df = processed_df[processed_df['MTF_Scenario'] == 'Long Unwinding / Capitulation']
        top10_cat4 = cat4_df.sort_values(by=sort_column, ascending=True).head(10)
        st.dataframe(top10_cat4, use_container_width=True)

    st.divider()
    
    # 3. COMPLETE DATASET VIEW & DOWNLOAD
    st.subheader("📋 Full Analyzed Dataset")
    
    search_query = st.text_input("🔍 Search by Stock Symbol", "")
    
    if search_query:
        # Simple symbol filter if symbol column exists
        symbol_col = [col for col in processed_df.columns if 'symbol' in col.lower() or 'scrip' in col.lower() or 'name' in col.lower()]
        if symbol_col:
            display_df = processed_df[processed_df[symbol_col[0]].astype(str).str.contains(search_query.upper(), na=False)]
        else:
            display_df = processed_df
    else:
        display_df = processed_df
        
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Convert dataframe to CSV for download
    csv_data = display_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Analyzed CSV Report",
        data=csv_data,
        file_filename=f"MTF_Analyzed_Report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        type="primary"
    )

else:
    st.info("👈 Use the sidebar on the left to select a date and click **Fetch & Generate Dashboard** or upload a custom CSV file to view the analysis.")
