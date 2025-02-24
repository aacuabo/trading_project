import os
import pandas as pd
from datetime import datetime, time, timedelta
import sqlite3
import logging

# Configure logging - Initialize logger for script1
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('script1.log', mode='w') # 'w' mode for fresh log on each run in testing
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
stream_handler = logging.StreamHandler() # Optional: Also log to console
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.info("Script 1 execution started.") # Initial log message


def format_interval(date_str):
    """
    Formats date string to 'YYYY-MM-DD HH:MM:SS' for consistent interval representation.
    Handles missing seconds, microseconds, and parsing errors.
    Returns formatted string or None on error, logging any issues.
    """
    if not date_str: # Handle None or empty date strings explicitly
        logger.warning("format_interval received empty or None date_str.")
        return None
    try:
        if len(date_str) == 16: # Add seconds if missing
            date_str += ':00'
        if '.' in date_str: # Remove microseconds if present
            date_str = date_str.split('.')[0]
        dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') # Parse with consistent format
        return dt.strftime('%Y-%m-%d %H:%M:%SS') # Format to ensure consistent output
    except ValueError as ve:
        logger.error(f"ValueError formatting date string '{date_str}': {ve}")
        return None # Return None to indicate failure
    except Exception as e:
        logger.error(f"Unexpected error formatting date string '{date_str}': {e}")
        return None

def process_bcq_reports(file_paths, db_path, overwrite_existing_data=False):
    """
    Processes BCQ reports from CSV files, populates/updates BCQ and BCQ_Hourly tables in SQLite database.
    Handles data validation, deduplication, hourly aggregation, and redundancy (overwrite option).

    Args:
        file_paths (list): List of file paths to BCQ CSV reports.
        db_path (str): Path to the SQLite database file.
        overwrite_existing_data (bool, optional): If True, overwrite existing data for same intervals. Defaults to False.

    Returns:
        pandas.DataFrame or None: Preview DataFrame of processed data if successful, None on critical error.
    """
    logger.info(f"Starting process_bcq_reports. Files: {file_paths}, DB: {db_path}, Overwrite: {overwrite_existing_data}")
    final_df = pd.DataFrame() # Initialize DataFrame to store processed data

    conn = None # Initialize database connection outside try block for wider scope in finally
    try:
        conn = sqlite3.connect(db_path) # Establish database connection
        conn.text_factory = str # Set text factory to handle text encoding
        cursor = conn.cursor()

        # Create BCQ table if not exists (ensure schema is correct for your data)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS BCQ (
                "Interval" TEXT PRIMARY KEY,
                "Date" TEXT,
                "Time" TEXT,
                "Selling Participant" TEXT,
                "BCQ" REAL,
                "Total_BCQ" REAL
                -- Add other columns as needed, adjust data types accordingly
            )
        ''')
        # Create BCQ_Hourly table if not exists (ensure schema is correct for hourly data)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS BCQ_Hourly (
                "Interval" TEXT PRIMARY KEY,
                "Date" TEXT,
                "Time" TEXT,
                "Total_BCQ" REAL
                -- Add other columns as needed for hourly data
            )
        ''')
        conn.commit() # Commit table creation operations

        existing_intervals_bcq_data = pd.read_sql_query("SELECT Interval FROM BCQ", conn)['Interval'].tolist() if 'BCQ' in pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist() else []
        existing_intervals_hourly_bcq_data = pd.read_sql_query("SELECT Interval FROM BCQ_Hourly", conn)['Interval'].tolist() if 'BCQ_Hourly' in pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist() else []

        processed_dfs_list = [] # List to hold processed DataFrames from each file

        for file_path in file_paths:
            logger.info(f"Processing BCQ report file: {file_path}")
            try:
                df = pd.read_csv(file_path) # Read CSV into DataFrame
                logger.debug(f"Dataframe loaded from {file_path}, initial shape: {df.shape}")

                # Data Cleaning and Transformation - as per your original script
                if 'Time' not in df.columns or 'Selling Participant' not in df.columns or 'BCQ' not in df.columns:
                    logger.error(f"File {file_path} is missing required columns ('Time', 'Selling Participant', 'BCQ'). Skipping file.")
                    continue # Skip to next file if essential columns are missing

                df = df[['Selling Participant', 'Time', 'BCQ']].copy() # Select and copy relevant columns
                df['Time'] = df['Time'].astype(str).apply(format_interval) # Format 'Time' column
                df.dropna(subset=['Time'], inplace=True) # Remove rows with invalid/unformattable times
                df = df.dropna(subset=['Selling Participant', 'BCQ']) # Remove rows with NaN in Selling Participant or BCQ after time formatting

                df['BCQ'] = pd.to_numeric(df['BCQ'], errors='coerce').fillna(0.0) # Convert BCQ to numeric, coerce errors to NaN, fill NaN with 0
                df['BCQ'] = df['BCQ'].round(9) # Round BCQ values

                df.rename(columns={'Time': 'Interval'}, inplace=True) # Rename 'Time' to 'Interval' for consistency

                df['Date'] = pd.to_datetime(df['Interval']).dt.date # Extract Date and Time components
                df['Time'] = pd.to_datetime(df['Interval']).dt.time

                grouped_df = df.groupby(['Interval', 'Date', 'Time', 'Selling Participant'])['BCQ'].sum().reset_index() # Group and sum BCQ
                pivoted_df = grouped_df.pivot(index=['Interval', 'Date', 'Time'], columns='Selling Participant', values='BCQ').reset_index().fillna(0.0) # Pivot for seller columns
                pivoted_df['Time'] = pivoted_df['Time'].apply(lambda x: x.strftime('%H:%M:%S') if isinstance(x, time) else x) # Format Time part back to string HH:MM:SS

                selling_participants = df['Selling Participant'].unique() # Get unique selling participants for current file
                seller_columns = list(selling_participants) # Seller columns are dynamic based on participants

                pivoted_df[seller_columns] = pivoted_df[seller_columns].round(9) # Round seller columns

                pivoted_df['Total_BCQ'] = pivoted_df[seller_columns].sum(axis=1).round(9) # Calculate Total_BCQ

                processed_dfs_list.append(pivoted_df) # Accumulate processed DataFrame for current file
                logger.info(f"Processed data from {file_path}, shape: {pivoted_df.shape}")

            except FileNotFoundError:
                logger.error(f"File not found: {file_path}") # Log file not found error
            except pd.errors.EmptyDataError:
                logger.error(f"No data in file: {file_path}") # Log empty data error
            except Exception as file_processing_error:
                logger.error(f"Error processing file {file_path}: {file_processing_error}", exc_info=True) # Log general file processing errors

        if not processed_dfs_list: # If no files were processed successfully, exit
            logger.warning("No BCQ data processed from any files due to errors. Aborting database update.")
            return pd.DataFrame() # Return empty DataFrame to indicate no data processed

        final_df = pd.concat(processed_dfs_list, ignore_index=True) # Concatenate DataFrames from all processed files
        logger.info(f"Concatenated data from all files, final shape before deduplication: {final_df.shape}")
        final_df.drop_duplicates(subset=['Interval'], inplace=True) # Deduplicate based on 'Interval'
        logger.info(f"Data deduplicated, final shape before database update: {final_df.shape}")


        # --- BCQ Table Update Logic ---
        new_data_bcq_data = final_df[~final_df['Interval'].isin(existing_intervals_bcq_data)].copy()
        existing_data_bcq_data = final_df[final_df['Interval'].isin(existing_intervals_bcq_data)].copy()

        if not new_data_bcq_data.empty:
            new_data_bcq_data.to_sql('BCQ', conn, if_exists='append', index=False) # Insert new data
            logger.info(f"Inserted {len(new_data_bcq_data)} new records into BCQ table.")

        if not existing_data_bcq_data.empty:
            if overwrite_existing_data: # Check if overwrite is enabled
                records_to_update_bcq = existing_data_bcq_data.to_dict('records')
                update_bcq_query = f'''
                    UPDATE BCQ
                    SET {', '.join([f'"{col}" = ?' for col in existing_data_bcq_data.columns if col not in ["Interval", "Date", "Time"]])}
                    WHERE "Interval" = ?
                ''' # Dynamically build update query - EXCLUDING Interval, Date, Time from SET clause
                update_bcq_data = [tuple(row[col] for col in existing_data_bcq_data.columns if col not in ["Interval", "Date", "Time"]) + (row['Interval'],) for row in records_to_update_bcq] # Prepare update data
                cursor.executemany(update_bcq_query, update_bcq_data) # Batch update existing data
                logger.info(f"Updated {len(existing_data_bcq_data)} existing records in BCQ table (overwrite enabled).")
            else:
                logger.warning(f"Skipping update for {len(existing_data_bcq_data)} existing records in BCQ table (overwrite disabled). Intervals: {existing_data_bcq_data['Interval'].tolist()}")


        # --- Hourly BCQ Data Processing and Update ---
        hourly_df = final_df.copy()

        hourly_df['Interval'] = pd.to_datetime(hourly_df['Interval'])
        hourly_df['Hour'] = hourly_df['Interval'].apply(lambda x:
            (x - timedelta(hours=1)).replace(hour=23, minute=0, second=0, microsecond=0) # Set to 23:00:00 of PREVIOUS day
            if x.hour == 0
            else (x.replace(hour=23, minute=0, second=0, microsecond=0) if x.hour == 23 else x.floor('H')) # For 23rd hour set to 23:00:00 of SAME day, else floor to hour start
        )
        hourly_df['Date'] = hourly_df['Interval'].dt.strftime('%Y-%m-%d')
        hourly_df['Time'] = hourly_df['Interval'].dt.strftime('%H:%M:%S')

        seller_cols_hourly = [col for col in hourly_df.columns if col in selling_participants]

        hourly_aggregated_df = hourly_df.groupby(['Hour', 'Date']).sum(numeric_only=True).reset_index() # GroupBy Hour and Date
        hourly_aggregated_df = hourly_aggregated_df.rename(columns={'Hour': 'Interval'})
        hourly_aggregated_df['Interval'] = hourly_aggregated_df['Interval'].dt.strftime('%Y-%m-%d %H:%M:%S') # Format Interval
        hourly_aggregated_df['Time'] = hourly_aggregated_df['Interval'].dt.strftime('%H:%M:%S') # Format Time

        # Calculate Total_BCQ for hourly data - seller columns only
        hourly_aggregated_df['Total_BCQ'] = hourly_aggregated_df[seller_cols_hourly].sum(axis=1).round(9) * 1000

        # Reorder columns for BCQ_Hourly table
        hourly_aggregated_df = hourly_aggregated_df[['Interval', 'Date', 'Time'] + seller_cols_hourly + ['Total_BCQ']]


        # --- Database Update for Hourly Data (BCQ_Hourly table) ---
        # Dynamically create BCQ_Hourly table schema to include seller columns
        create_hourly_table_sql = f'''
            CREATE TABLE IF NOT EXISTS BCQ_Hourly (
                "Interval" TEXT PRIMARY KEY,
                "Date" TEXT,
                "Time" TEXT,
                "Total_BCQ" REAL,
                {', '.join([f'"{seller}" REAL' for seller in selling_participants])}
                -- Seller columns added dynamically to BCQ_Hourly
            )
        '''
        cursor.execute(create_hourly_table_sql)
        conn.commit()

        existing_hourly_intervals = pd.read_sql_query("SELECT Interval FROM BCQ_Hourly", conn)['Interval'].tolist()
        hourly_new_data = hourly_aggregated_df[~hourly_aggregated_df['Interval'].isin(existing_hourly_intervals)].copy()
        hourly_existing_data = hourly_aggregated_df[hourly_aggregated_df['Interval'].isin(existing_hourly_intervals)].copy()

        if not hourly_new_data.empty:
            hourly_new_data.to_sql('BCQ_Hourly', conn, if_exists='append', index=False) # Insert ALL columns including seller columns
            logger.info(f"Inserted {len(hourly_new_data)} new records into BCQ_Hourly table (including seller columns).")

        if not hourly_existing_data.empty:
            if overwrite_existing_data:
                records_to_update_hourly = hourly_existing_data.to_dict('records')
                # Dynamically build UPDATE query to include seller columns
                update_hourly_query = f'''
                    UPDATE BCQ_Hourly
                    SET "Total_BCQ" = ?,
                        {', '.join([f'"{seller}" = ?' for seller in selling_participants])}
                    WHERE "Interval" = ?
                '''
                update_hourly_cols_data = []
                for rec in records_to_update_hourly:
                    update_hourly_row_data = [rec['Total_BCQ']] # Start with Total_BCQ
                    for seller in selling_participants: # Add seller column values in correct order
                        update_hourly_row_data.append(rec[seller])
                    update_hourly_row_data.append(rec['Interval']) # Finally, add Interval for WHERE clause
                    update_hourly_cols_data.append(tuple(update_hourly_row_data))

                cursor.executemany(update_hourly_query, update_hourly_cols_data)
                logger.info(f"Updated {len(hourly_existing_data)} existing records in BCQ_Hourly table (including seller columns, overwrite enabled).")
            else:
                logger.warning(f"Skipping update for {len(hourly_existing_data)} existing hourly records in BCQ_Hourly table (overwrite disabled). Intervals: {hourly_existing_data['Interval'].tolist()}")


        conn.commit() # Final commit after all BCQ and BCQ_Hourly updates
        logger.info("BCQ and Hourly BCQ data processing completed successfully.")
        return final_df # Return final processed DataFrame for preview

    except sqlite3.Error as db_error:
        logger.error(f"Database error during BCQ processing: {db_error}", exc_info=True) # Log database errors
        return None # Return None for critical database error
    except Exception as main_error:
        logger.error(f"General error in process_bcq_reports: {main_error}", exc_info=True) # Log any other exceptions
        return None # Return None for general processing errors
    finally:
        if conn:
            conn.close() # Ensure database connection is closed in finally block


if __name__ == '__main__':
    # --- Standalone Test Execution ---
    test_db_path = 'test_database_bcq_rewrite.db' # New test database for rewrite testing
    test_file_paths = ['bcq_report_sample_hourly_final.csv'] # Example multiple files, new sample CSV

    # Create sample CSV for testing if it doesn't exist
    if not os.path.exists(test_file_paths[0]):
        sample_data_hourly_final = {
            'Selling Participant': ['SellerA', 'SellerB', 'SellerA', 'SellerB', 'SellerC', 'SellerA', 'SellerB', 'SellerC', 'SellerA', 'SellerB', 'SellerC', 'SellerA'],
            'Time': ['2025-02-01 23:05:00', '2025-02-01 23:10:00', '2025-02-01 23:15:00', '2025-02-01 23:20:00', '2025-02-01 23:25:00', '2025-02-01 23:30:00',
                     '2025-02-01 23:35:00', '2025-02-01 23:40:00', '2025-02-01 23:45:00', '2025-02-01 23:50:00', '2025-02-01 23:55:00', '2025-02-02 00:00:00'],
            'BCQ': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70]
        }
        pd.DataFrame(sample_data_hourly_final).to_csv(test_file_paths[0], index=False)
        print(f"Created sample CSV for hourly final test: {test_file_paths[0]}")


    print("\n--- Test Run: Final Hourly Aggregation Test ---")
    preview_df_hourly_final_test = process_bcq_reports(test_file_paths, test_db_path) # Overwrite disabled (default)
    if preview_df_hourly_final_test is not None:
        print("\nPreview DataFrame (Hourly Final Test):")
        print(preview_df_hourly_final_test.head(2)) # Display first 2 rows to show both hourly intervals
        print(preview_df_hourly_final_test.tail(2)) # Display last 2 rows as well just to be sure if dataframe is longer
    else:
        print("\nProcessing failed (Hourly Final Test). Check script1.log for details.")


    print("\nScript 1 final hourly aggregation test execution completed. Check 'script1.log' and database file.")