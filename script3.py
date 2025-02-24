# script3.py - VERSION: CSV_FILES_UPLOAD_STREAMLIT_READY

import os
import pandas as pd
import sqlite3
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)
print("SCRIPT3.PY VERSION: CSV_FILES_UPLOAD_STREAMLIT_READY") # VERSION PRINT for console
logger.info("SCRIPT3.PY VERSION: CSV_FILES_UPLOAD_STREAMLIT_READY - LOGGING") # VERSION LOG for log file

def create_prices_tables(conn):
    """Creates Prices and Prices_Hourly tables in the database if they don't exist."""
    try:
        # Create Prices table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS Prices (
                Interval TEXT PRIMARY KEY,
                Date TEXT,
                Time TEXT,
                Prices REAL,
                Hour TEXT
            )
        ''')

        # Create Prices_Hourly table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS Prices_Hourly (
                Interval TEXT PRIMARY KEY,
                Date TEXT,
                Time TEXT,
                Prices REAL
            )
        ''')
        conn.commit()
        logger.info("Prices and Prices_Hourly tables created or already exist.")
    except sqlite3.Error as e:
        logger.error(f"Error creating Prices tables: {str(e)}")
        raise

def aggregate_hourly_prices(conn):
    """Aggregates prices data to hourly averages and updates Prices_Hourly table."""
    try:
        # Clear existing data in Prices_Hourly
        conn.execute('DELETE FROM Prices_Hourly')

        # Aggregate average of Prices for each Hour
        query = '''
            SELECT
                Hour AS Interval,
                AVG(Prices) AS Prices
            FROM Prices
            GROUP BY Hour
        '''
        hourly_data = pd.read_sql_query(query, conn)

        # Extract Date and Time from Interval
        hourly_data['Interval'] = pd.to_datetime(hourly_data['Interval'], format='%Y-%m-%d %H:00:00')
        hourly_data['Date'] = hourly_data['Interval'].dt.date
        hourly_data['Time'] = hourly_data['Interval'].dt.time.astype(str)

        # Insert data into Prices_Hourly table
        for _, row in hourly_data.iterrows():
            conn.execute('''
                INSERT OR REPLACE INTO Prices_Hourly (Interval, Date, Time, Prices)
                VALUES (?, ?, ?, ?)
            ''', (row['Interval'].strftime('%Y-%m-%d %H:00:00'), row['Date'], row['Time'], row['Prices']))

        conn.commit()
        logger.info("Prices_Hourly table updated with aggregated data.")
    except sqlite3.Error as e:
        logger.error(f"Error aggregating hourly prices: {str(e)}")
        raise

def process_prices_reports(file_paths, db_path):
    """
    Processes Prices reports from CSV files, updates the database, and returns a preview DataFrame.

    Args:
        file_paths (list): List of file paths to CSV files.
        db_path (str): Path to the SQLite database file.

    Returns:
        pandas.DataFrame: Preview DataFrame containing processed prices data (or empty DataFrame on error).
        bool: True if processing was successful, False otherwise.
    """
    preview_df = pd.DataFrame()
    success = False
    all_processed_data = [] # List to collect processed data for preview

    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        create_prices_tables(conn)

        if not file_paths:
            logger.warning("No CSV files provided for processing.")
            return pd.DataFrame({'Warning': ['No CSV files selected for processing.']}), False

        for file_path in file_paths:
            csv_file = os.path.basename(file_path)
            df = pd.read_csv(file_path)

            # Filter rows where RESOURCE_NAME is "14BGN_T1L1"
            df = df[df['RESOURCE_NAME'] == '14BGN_T1L1']

            # Check if 'TIME_INTERVAL' and 'LMP' columns exist
            if 'TIME_INTERVAL' not in df.columns or 'LMP' not in df.columns:
                logger.error(f"CSV file {csv_file} does not contain required columns 'TIME_INTERVAL' and 'LMP'")
                raise ValueError(f"CSV file {csv_file} does not contain required columns 'TIME_INTERVAL' and 'LMP'")

            # Rename columns to match database schema
            df = df.rename(columns={'TIME_INTERVAL': 'Interval', 'LMP': 'Prices'})

            # Add default time if missing
            df['Interval'] = df['Interval'].apply(lambda x: x if ' ' in x else x + ' 12:00:00 AM')

            # Specify the format for the Interval column
            df['Interval'] = pd.to_datetime(df['Interval'], format='%m/%d/%Y %I:%M:%S %p')

            # Separate Interval into Date and Time
            df['Date'] = df['Interval'].dt.date
            df['Time'] = df['Interval'].dt.time.astype(str)

            # Add Hour column
            df['Hour'] = df['Interval'].apply(
                lambda x: (
                    (x - timedelta(minutes=5)).strftime('%Y-%m-%d %H:00:00')
                    if x.strftime('%H:%M:%S') != '00:00:00'
                    else (x - timedelta(hours=1)).strftime('%Y-%m-%d %H:00:00')
                )
            )

            # Collect data for preview (sample from each file)
            all_processed_data.append(df)

            # Insert data into the Prices table
            for _, row in df.iterrows():
                conn.execute('''
                    INSERT OR REPLACE INTO Prices (Interval, Date, Time, Prices, Hour)
                    VALUES (?, ?, ?, ?, ?)
                ''', (row['Interval'].strftime('%Y-%m-%d %H:%M:%S'), row['Date'], row['Time'], row['Prices'], row['Hour']))

            conn.commit()

        # Aggregate hourly data and insert into Prices_Hourly table
        aggregate_hourly_prices(conn)
        conn.close()
        success = True

        # Create preview DataFrame from collected data (first 5 rows of concatenated data)
        if all_processed_data:
            preview_df = pd.concat(all_processed_data, ignore_index=True).head(10) # Preview first 10 rows
        else:
            preview_df = pd.DataFrame({'Status': ['No data processed for preview.']}) # Indicate no data if nothing processed

        logger.info("Prices reports processed successfully.")
        return preview_df, success

    except Exception as e:
        logger.error(f"Error processing prices reports: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return pd.DataFrame({'Error': [str(e)]}), False

if __name__ == "__main__":
    db_path = 'd:/Daily Data/bcq_data.db' # Adjust path as needed for testing
    # Example usage for testing with hardcoded file paths
    test_folder = 'test_prices_data' # Folder for test CSVs
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Create dummy CSV content (adjust as per your CSV structure)
    csv_content = """TIME_INTERVAL,RESOURCE_NAME,MARKET_NAME,LMP
01/01/2024 01:00:00 AM,14BGN_T1L1,HB_BUSAVG,25.50
01/01/2024 01:05:00 AM,14BGN_T1L1,HB_BUSAVG,26.75
01/01/2024 01:10:00 AM,14BGN_T1L1,HB_BUSAVG,27.00
01/01/2024 01:15:00 AM,14BGN_T1L1,HB_BUSAVG,26.50
01/01/2024 01:20:00 AM,14BGN_T1L1,HB_BUSAVG,25.90"""

    # Create two dummy CSV files in the test folder
    with open(os.path.join(test_folder, 'DIPCER_test1.csv'), 'w') as f:
        f.write(csv_content)
    with open(os.path.join(test_folder, 'DIPCER_test2.csv'), 'w') as f: # Another test file
        f.write(csv_content)

    file_paths = [os.path.join(test_folder, 'DIPCER_test1.csv'), os.path.join(test_folder, 'DIPCER_test2.csv')] # Hardcoded file paths for testing

    preview_df, success = process_prices_reports(file_paths, db_path)
    if success:
        print("Prices report processing (test) completed successfully.")
        if not preview_df.empty:
            print("\nPreview Data (first 10 rows):\n", preview_df.to_string())
    else:
        print("Prices report processing (test) failed. Check logs.")
        if not preview_df.empty:
            print("\nError Preview:\n", preview_df.to_string())