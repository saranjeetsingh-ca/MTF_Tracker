import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# ==========================================
# PART 1: THE DATA ANALYZER
# ==========================================
def process_mtf_data(df):
    """Analyzes the MTF data to find leverage traps and bullish accelerations."""
    print("Crunching the MTF numbers...")
    
    df['Delta_MTF'] = df.get('Current_Day_Funded_Value', 0) - df.get('Previous_Day_Funded_Value', 0)
    
    if 'Total_Daily_Delivery_Value' in df.columns and 'Daily_Fresh_MTF_Capital_Added' in df.columns:
        df['Leverage_Intensity_Ratio'] = np.where(
            df['Total_Daily_Delivery_Value'] > 0,
            df['Daily_Fresh_MTF_Capital_Added'] / df['Total_Daily_Delivery_Value'],
            0
        )
    else:
        df['Leverage_Intensity_Ratio'] = 0

    if 'Price_Change_Pct' in df.columns:
        conditions = [
            (df['Price_Change_Pct'] > 0) & (df['Delta_MTF'] > 0),
            (df['Price_Change_Pct'] > 0) & (df['Delta_MTF'] < 0),
            (df['Price_Change_Pct'] < 0) & (df['Delta_MTF'] > 0),
            (df['Price_Change_Pct'] < 0) & (df['Delta_MTF'] < 0)
        ]
        choices = [
            'Bullish Leverage Acceleration',
            'Short-Covering / Profit Booking',
            'Catching a Falling Knife (Danger)',
            'Long Unwinding / Capitulation'
        ]
        df['MTF_Scenario'] = np.select(conditions, choices, default='Neutral')
    else:
        df['MTF_Scenario'] = 'Price Data Missing'

    if 'Free_Float_Market_Cap' in df.columns:
        df['Over_Leveraged_Warning'] = (df['Current_Day_Funded_Value'] / df['Free_Float_Market_Cap']) > 0.025
    else:
        df['Over_Leveraged_Warning'] = False

    if 'Consecutive_Down_Days' in df.columns:
        df['Averaging_Trap_Flag'] = (df['Consecutive_Down_Days'] >= 3) & (df['Delta_MTF'] > 0)
    else:
        df['Averaging_Trap_Flag'] = False

    return df

# ==========================================
# PART 2: THE NSE DOWNLOADER & EXECUTOR
# ==========================================
def main():
    date_str = datetime.now().strftime("%d%m%Y")
    print(f"Starting GitHub Actions run for date: {date_str}...")
    
    # 1. Setup connection headers
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.nseindia.com/all-reports'
    }
    
    try:
        # 2. Get NSE Cookies
        print("Connecting to NSE...")
        session.get("https://www.nseindia.com", headers=headers, timeout=15)
        
        # 3. Request the File
        target_url = f"https://nsearchives.nseindia.com/archives/equities/mtf/MTF_Disclosure_{date_str}.csv"
        print(f"Requesting file: {target_url}")
        
        response = session.get(target_url, headers=headers, timeout=20)
        
        if response.status_code == 200:
            filename = f"Raw_NSE_MTF_{date_str}.csv"
            with open(filename, 'wb') as file:
                file.write(response.content)
            print("Successfully downloaded raw data!")
            
            # 4. Analyze Data
            df = pd.read_csv(filename)
            analyzed_df = process_mtf_data(df)
            
            # 5. Save Final Output
            output_filename = f"Analyzed_MTF_{date_str}.csv"
            analyzed_df.to_csv(output_filename, index=False)
            print(f"Analysis complete and saved to {output_filename}")
            
        elif response.status_code == 404:
            print("Error 404: NSE hasn't uploaded today's report yet.")
        elif response.status_code == 403:
            print("Error 403: NSE Firewall blocked the GitHub server. (We may need to use a proxy if this persists).")
        else:
            print(f"Failed to fetch data. Status Code: {response.status_code}")
            
    except Exception as e:
        print(f"Something went wrong: {e}")

if __name__ == "__main__":
    main()
