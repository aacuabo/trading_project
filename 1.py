import streamlit as st
import sqlite3
import logging
import json
import os
import sys
import subprocess
from datetime import datetime
from io import BytesIO, StringIO

# Temporary imports - make sure these scripts exist in your project
from script1 import process_bcq_reports
from script2 import process_mq_reports
from script3 import process_prices_reports
from script4 import MonthlyDataProcessor


# --- Configure logging first ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='streamlit_app.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

from st_pages import Page, show_pages, add_page_title

st.set_page_config(layout="wide")

# Define pages directly or use a TOML-based approach
# Optional: Add title dynamically
add_page_title()

def init_session_state():
    """Initialize all required session state variables"""
    defaults = {
        'db_path': None,
        'logs': [],
        'db_initialized': False,
        'processing': False,
        'bcq_processed': False,  # Track BCQ processing status
        'mq_processed': False,   # Track MQ processing status
        'prices_processed': False  # Track Prices processing status
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def handle_file_upload(uploaded_file, allowed_types):
    """Handle file uploads and return file path"""
    if uploaded_file is None:
        return None

    if uploaded_file.type not in allowed_types:
        st.error(f"Invalid file type: {uploaded_file.type}")
        return None

    try:
        # Create a temporary file
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return file_path
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        st.error("File upload failed. Check logs for details.")
        return None

def save_database_path(db_path):
    """Save the database path to the config file."""
    config_file = 'app_config.json'
    try:
        config = {'last_database': db_path}
        with open(config_file, 'w') as f:
            json.dump(config, f)
        logger.info(f"Database path saved to config: {db_path}")
    except Exception as e:
        logger.error(f"Error saving database path to config: {str(e)}")
        st.error(f"Error saving database configuration. Check logs.")

def database_section():
    """Database configuration section"""
    st.header("🔌 Database Configuration")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_db = st.file_uploader(
            "Select SQLite Database",
            type=["sqlite", "db", "sqlite3"],
            key="db_uploader"
        )

    with col2:
        if st.button("Initialize New Database"):
            try:
                new_db_path = os.path.join("databases", "new_database.db")
                os.makedirs("databases", exist_ok=True)

                conn = sqlite3.connect(new_db_path)
                initialize_database(conn)
                conn.close()

                st.session_state.db_path = new_db_path
                save_database_path(new_db_path)  # Save database path
                st.success(f"New database created: {new_db_path}")
                st.session_state.db_initialized = True  # Set db_initialized to True
            except Exception as e:
                st.error(f"Database creation failed: {str(e)}")

    if uploaded_db:
        db_path = handle_file_upload(uploaded_db, ["application/octet-stream", "application/x-sqlite3"])
        if db_path:
            try:
                conn = sqlite3.connect(db_path)
                initialize_database(conn)
                conn.close()

                st.session_state.db_path = db_path
                save_database_path(db_path)  # Save database path
                st.success("Database connected successfully!")
                st.session_state.db_initialized = True  # Set db_initialized to True
            except sqlite3.Error as e:
                st.error(f"Database connection failed: {str(e)}")
                logger.error(f"DB Connection Error: {str(e)}")

    if st.session_state.db_path:
        st.markdown(f"""
        **Connected Database:**
        `{st.session_state.db_path}`
        Last modified: {datetime.fromtimestamp(os.path.getmtime(st.session_state.db_path)).strftime('%Y-%m-%d %H:%M:%S')}
        """)

def initialize_database(conn):
    """Initialize database tables with error handling"""
    tables = {
        'Prices': '''
            CREATE TABLE IF NOT EXISTS Prices (
                Interval TEXT PRIMARY KEY,
                Date TEXT,
                Time TEXT,
                Prices REAL,
                Hour TEXT
            )''',
        'Prices_Hourly': '''
            CREATE TABLE IF NOT EXISTS Prices_Hourly (
                Interval TEXT PRIMARY KEY,
                Date TEXT,
                Time TEXT,
                Prices REAL
            )''',
        'MQ': '''
            CREATE TABLE IF NOT EXISTS MQ (
                "Interval" TEXT PRIMARY KEY,
                "Date" TEXT,
                "Time" TEXT,
                "Hour" TEXT,
                "Total_MQ" REAL
            )''',
        'MQ_Hourly': '''
            CREATE TABLE IF NOT EXISTS MQ_Hourly (
                "Interval" TEXT PRIMARY KEY,
                "Date" TEXT,
                "Time" TEXT,
                "Total_MQ" REAL
            )''',
        'BCQ': ''' # Added BCQ table here to ensure it exists upon DB init
            CREATE TABLE IF NOT EXISTS BCQ (
                "Interval" TEXT PRIMARY KEY,
                "Date" TEXT,
                "Time" TEXT,
                "Hour" TEXT,
                "Selling Participant" TEXT,
                "Volume" REAL,
                "Price" REAL
            )''',
        'BCQ_Hourly': ''' # Added BCQ_Hourly table here
            CREATE TABLE IF NOT EXISTS BCQ_Hourly (
                "Interval" TEXT PRIMARY KEY,
                "Date" TEXT,
                "Hour" TEXT,
                "Volume" REAL,
                "Price" REAL
            )'''
    }

    try:
        cursor = conn.cursor()
        for table_name, ddl in tables.items():
            cursor.execute(ddl)
            logger.info(f"Created table {table_name} if not exists")
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

def load_last_database():
    """Load the last used database path from config file"""
    config_file = 'app_config.json'
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                last_db = config.get('last_database')
                if last_db and os.path.exists(last_db):
                    try:
                        # Add this connection check
                        conn = sqlite3.connect(last_db)
                        conn.close()
                        st.session_state.db_path = last_db
                        st.session_state.db_initialized = True
                    except sqlite3.Error as e:
                        logger.error(f"Invalid database file: {str(e)}")
                        st.session_state.db_initialized = False
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        st.session_state.db_initialized = False


def processing_section():
    """Main processing section with tabs"""
    st.header("⚙️ Data Processing")

    tab1, tab2, tab3, tab4 = st.tabs([
        "BCQ Reports",
        "MQ Reports",
        "Prices Reports",
        "Monthly Processing"
    ])

    with tab1:
        process_bcq_section()
        if st.session_state.bcq_processed:  # Display status based on session state
            st.success("BCQ Processing Completed Successfully!")
            st.session_state.bcq_processed = False  # Reset status

    with tab2:
        process_mq_section()
        if st.session_state.mq_processed:  # Display status based on session state
            st.success("MQ Processing Completed Successfully!")
            st.session_state.mq_processed = False  # Reset status

    with tab3:
        process_prices_section()
        if st.session_state.prices_processed:  # Display status based on session state
            st.success("Prices Processing Completed Successfully!")
            st.session_state.prices_processed = False  # Reset status

    with tab4:
        process_monthly_section()

def process_bcq_section():
    """BCQ Reports processing"""
    st.subheader("BCQ Report Processing")

    uploaded_files = st.file_uploader(
        "Select CSV files",
        type=["csv"],
        accept_multiple_files=True,
        key="bcq_uploader"
    )

    if uploaded_files and st.button("Process BCQ Files"):
        if not validate_database():
            return

        try:
            with st.spinner("Processing BCQ Reports..."):
                # Convert uploaded files to file paths
                file_paths = [handle_file_upload(f, ["text/csv"]) for f in uploaded_files]
                if not file_paths or None in file_paths:  # Handle potential upload errors
                    st.error("File upload failed for one or more files. Check logs.")
                    return

                result_df = process_bcq_reports(file_paths, st.session_state.db_path)

                if result_df is not None:  # Check for critical error from script1.py
                    st.success("Processing completed successfully!")
                    st.subheader("Preview of Processed Data")
                    st.dataframe(result_df.head(10))

                    # Export option
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Processed Data",
                        data=csv,
                        file_name='processed_bcq_data.csv',
                        mime='text/csv'
                    )
                    st.session_state.bcq_processed = True  # Set BCQ processing status to True

                    # --- Temporary File Cleanup (Example after successful processing) ---
                    if file_paths:
                        for path in file_paths:
                            try:
                                os.remove(path)
                                logger.info(f"Temporary file deleted: {path}")
                            except Exception as e:
                                logger.warning(f"Error deleting temporary file {path}: {str(e)}")
                else:
                    st.error("BCQ processing failed. Check logs for details.")  # Error already logged in script1.py

        except Exception as e:
            handle_processing_error(e, "BCQ processing")

def process_mq_section():
    """MQ Reports processing"""
    st.subheader("MQ Report Processing")

    uploaded_files = st.file_uploader(
        "Select Excel files",
        type=["xlsx"],
        accept_multiple_files=True,
        key="mq_uploader"
    )

    if uploaded_files and st.button("Process MQ Files"):
        if not validate_database():
            return

        try:
            with st.spinner("Processing MQ Reports..."):
                file_paths = [handle_file_upload(f, ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"])
                              for f in uploaded_files]
                if not file_paths or None in file_paths:  # Handle upload errors
                    st.error("File upload failed for one or more files. Check logs.")
                    return

                preview_df, success = process_mq_reports(file_paths, st.session_state.db_path)

                if success:
                    st.success("Processing completed successfully!")
                    if not preview_df.empty:
                        st.subheader("Preview of Processed Data")
                        st.dataframe(preview_df)

                        # Export option for MQ data
                        csv = preview_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Processed MQ Data",
                            data=csv,
                            file_name='processed_mq_data.csv',
                            mime='text/csv'
                        )
                        display_database_stats('MQ')
                        st.session_state.mq_processed = True  # Set MQ processing status to True
                else:
                    st.error("MQ processing failed. Check logs for details.")  # Error already logged in script2.py

        except Exception as e:
            handle_processing_error(e, "MQ processing")

def process_prices_section():
    """Prices Reports processing - with improved folder selection instructions"""
    st.subheader("Prices Report Processing")

    st.info(
        """
        **To process Prices Reports, follow these steps:**

        1.  **Think of the folder** on your computer that contains the Prices CSV files you want to process. *(You will 'open' this folder in the next step using your operating system's file dialog.)*
        2.  Click the 'Browse files' button below. This will open your computer's file explorer.
        3.  **Navigate to the folder** you identified in step 1.
        4.  **Select all the DIPCER CSV files** you want to process from within that folder.
            *   To select multiple files, you can typically:
                *   **Click and drag** to select a range of files.
                *   **Ctrl+Click (Cmd+Click on Mac)** to select individual files.
                *   **Shift+Click** to select a continuous range after selecting the first file.
            *   **Important:** Please ensure you are selecting **only** the CSV files you intend to process from the folder.
        5.  Click 'Open' (or the equivalent button in your file explorer) to upload the selected CSV files.
        6.  Finally, click the 'Process Prices Files' button to start processing the uploaded data.
        """
    )

    uploaded_files = st.file_uploader(
        "**Step 2 & 4:** Browse to your folder, select all DIPCER CSV files, and click 'Open'",  # More descriptive label
        type=["csv"],
        accept_multiple_files=True,
        key="prices_uploader"
    )

    if uploaded_files and st.button("Process Prices Files"):
        if not validate_database():
            return

        if not uploaded_files:
            st.warning("Please upload CSV files first.")
            return

        try:
            with st.spinner("Processing Prices Reports..."):
                file_paths = [handle_file_upload(f, ["text/csv"]) for f in uploaded_files]

                if not file_paths or None in file_paths:
                    st.error("File upload failed for one or more files. Check logs.")
                    return

                preview_df, success = process_prices_reports(file_paths, st.session_state.db_path)

                if success:
                    st.success("Processing completed successfully!")
                    if not preview_df.empty:
                        st.subheader("Preview of Processed Data")
                        st.dataframe(preview_df.head(10))

                        csv_data = preview_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Processed Prices Data",
                            data=csv_data,
                            file_name='processed_prices_data.csv',
                            mime='text/csv'
                        )
                        display_database_stats('Prices')
                        st.session_state.prices_processed = True
                else:
                    st.error("Prices processing failed. Check logs for details.")

        except Exception as e:
            handle_processing_error(e, "Prices processing")

def process_monthly_section():
    """Monthly processing"""
    st.subheader("Monthly Report Generation")

    if st.button("Run Monthly Processing"):
        try:
            with st.spinner("Generating monthly reports..."):
                result = subprocess.run(
                    [sys.executable, "script4.py"],
                    capture_output=True,
                    text=True,
                    check=True
                )

                st.success("Monthly processing completed!")
                st.subheader("Processing Logs")
                st.code(result.stdout)

                if result.stderr:
                    st.subheader("Error Output")
                    st.code(result.stderr)
        except subprocess.CalledProcessError as e:
            st.error(f"Processing failed with exit code {e.returncode}")
            st.code(e.stderr)
        except Exception as e:
            handle_processing_error(e, "Monthly processing")

def validate_database():
    """Check if database is initialized"""
    if not st.session_state.db_initialized:
        st.warning("Please configure the database first!")
        return False
    return True

def handle_processing_error(error, process_name):
    """Handle and display processing errors"""
    logger.error(f"{process_name} error: {str(error)}", exc_info=True)  # Log with traceback
    st.error(f"{process_name} failed: {str(error)}")
    st.code(str(error), language='python')

def display_database_stats(table_name):
    """Show basic table statistics"""
    try:
        conn = sqlite3.connect(st.session_state.db_path)
        cursor = conn.cursor()

        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]

        cursor.execute(f"SELECT MIN(date), MAX(date) FROM {table_name}")
        min_date, max_date = cursor.fetchone()

        st.markdown(f"""
        **{table_name} Table Statistics:**
        - Total records: {count:,}
        - Date range: {min_date} to {max_date}
        """)

    except sqlite3.Error as e:
        st.warning(f"Could not retrieve stats for {table_name}: {str(e)}")
    finally:
        conn.close()


def main():
    """Main application layout - simplified for no sections"""
    st.set_page_config(
        page_title="Energy Trading Analytics",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()
    load_last_database()

    # --- st_pages setup is done outside main() and simplified ---

    # --- No direct calls to section functions in main ---

    st.title("⚡ Energy Trading Analytics Platform")
    st.markdown("Welcome to the Energy Trading Analytics Platform. Use the sidebar to navigate to different pages.")

    st.sidebar.header("System Information")
    st.sidebar.markdown(f"**Python Version:** {sys.version.split()[0]}")
    st.sidebar.markdown(f"**Streamlit Version:** {st.__version__}")
    st.sidebar.markdown("---")
    st.sidebar.button("Clear Cache", on_click=st.cache_data.clear)


if __name__ == "__main__":
    main()