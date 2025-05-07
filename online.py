import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, date
# import matplotlib.pyplot as plt # Not needed with Altair plot
# import matplotlib.ticker as ticker # Not needed with Altair plot
import altair as alt
# Removed: import streamlit_authenticator as stauth # Removed the authenticator library
# Removed: import yaml # Not needed
# Removed: from yaml.loader import SafeLoader # Not needed

# Set Streamlit page configuration
st.set_page_config(layout="wide") # Use wide layout for better display

# Removed: --- User Authentication ---
# Removed: Loading of authentication credentials and cookie from secrets
# Removed: Initialize the authenticator
# Removed: Add the login widget at the top of the main area
# Removed: Display content based on authentication status blocks (if/elif)


# --- DATABASE CONFIGURATION ---
@st.cache_resource # Cache the database engine creation
def get_sqlalchemy_engine():
    """Establishes and returns a SQLAlchemy database engine using Streamlit secrets."""
    try:
        # Read database credentials from secrets.toml
        user = st.secrets["database"]["user"]
        password = st.secrets["database"]["password"]
        host = st.secrets["database"]["host"]
        db = st.secrets["database"]["db"]
        # Ensure port is an integer
        port = int(st.secrets["database"]["port"])

        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
        # Use pool_pre_ping=True to handle potential disconnections
        engine = create_engine(url, pool_pre_ping=True)
        return engine
    except KeyError as e:
        st.error(f"Error loading database credentials: {e}. Make sure your .streamlit/secrets.toml file is correctly configured with [database] section and keys: user, password, host, db, port.")
        st.stop() # Stop the app if secrets are not found
    except ValueError:
         st.error("Error: Database port in secrets.toml is not a valid integer.")
         st.stop()
    except Exception as e:
        st.error(f"Error creating database engine: {e}")
        st.stop() # Stop the app if engine creation fails


# --- LOAD DATA ---
@st.cache_data(ttl=3600) # Cache available dates for an hour
def fetch_available_dates():
    """Fetches a list of unique dates available in the database."""
    try:
        engine = get_sqlalchemy_engine()
        # Assuming 'MQ_Hourly' table contains all relevant dates.
        query = """
            SELECT DISTINCT "Date"
            FROM "MQ_Hourly"
            ORDER BY "Date";
        """
        # Use parse_dates to ensure the "Date" column is read as datetime
        dates_df = pd.read_sql(query, engine, parse_dates=["Date"])
        # Convert datetime objects to date objects
        available_dates = dates_df["Date"].dt.date.tolist()
        return available_dates
    except Exception as e:
        st.error(f"Error fetching available dates: {e}")
        return []


@st.cache_data(ttl=600) # Cache hourly data for 10 minutes
def fetch_data(selected_date_str: str): # Added type hint for caching key
    """Fetches hourly MQ, BCQ, and Prices data for a selected date."""
    try:
        engine = get_sqlalchemy_engine()
        query = """
            SELECT mq."Time", mq."Total_MQ", bcq."Total_BCQ", p."Prices"
            FROM "MQ_Hourly" AS mq
            JOIN "BCQ_Hourly" AS bcq ON mq."Date" = bcq."Date" AND mq."Time" = bcq."Time"
            JOIN "Prices_Hourly" AS p ON mq."Date" = p."Date" AND mq."Time" = p."Time"
            WHERE mq."Date" = %s
            ORDER BY
                mq."Time"
        """
        df = pd.read_sql(query, engine, params=[(selected_date_str,)], parse_dates=["Date", "Time"])

        if not df.empty and 'Time' in df.columns and 'Date' in df.columns:
            try:
                # Combine the selected date (as a string) with the time (ensure it's string)
                df['Time_str'] = df['Time'].fillna('').astype(str).str.split().str[-1]
                df['Datetime'] = pd.to_datetime(selected_date_str + ' ' + df['Time_str'], errors='coerce')
                df.dropna(subset=['Datetime'], inplace=True)
                df['Time'] = df['Datetime']

            except Exception as e:
                st.error(f"Error converting 'Time' column to datetime after fetch: {e}. Please check the format of the 'Time' column in your database.")
                return pd.DataFrame()

        # Ensure numeric columns are indeed numeric after fetching/parsing
        for col in ["Total_MQ", "Total_BCQ", "Prices"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        # Drop rows where critical numeric conversions failed if necessary, or handle NaNs later


        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600) # Cache entity sums for 10 minutes
def fetch_bcq_entity_sums(selected_date_str: str): # Added type hint for caching key
     """Fetches the daily sum of BCQ for specific entities from BCQ_Hourly."""
     try:
         engine = get_sqlalchemy_engine()
         # Assuming these are the exact column names in BCQ_Hourly
         query = f"""
             SELECT
                 SUM("FDC Misamis Power Corporation (FDC)") AS "FDC",
                 SUM("GNPower Kauswagan Ltd. Co. (GNPKLCO)") AS "GNPK",
                 SUM("Power Sector Assets & Liabilities Management Corporation (PSALMGMIN)") AS "PSALM",
                 SUM("Sarangani Energy Corporation (SEC)") AS "SEC",
                 SUM("Therma South, Inc. (TSI)") AS "TSI",
                 SUM("Malita Power Inc. (SMCPC)") AS "MPI"
             FROM
                 "BCQ_Hourly"
             WHERE
                 "Date" = %s;
         """
         # Pass parameters as a list containing a tuple
         df = pd.read_sql(query, engine, params=[(selected_date_str,)])

         # Ensure columns are numeric after summing
         for col in ["FDC", "GNPK", "PSALM", "SEC", "TSI", "MPI"]:
             if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')

         return df
     except Exception as e:
         st.error(f"Error fetching BCQ entity sums: {e}")
         return pd.DataFrame()


# --- STREAMLIT UI (Now always displayed) ---

st.title("📊 Daily Energy Trading Dashboard") # Main title

# Create outer columns for centering the main content
# Use a ratio like [1, 4, 1] to make the central content column roughly 4/6 (2/3) of the page width
spacer_left, main_content, spacer_right = st.columns([1, 4, 1])

with main_content: # Place all the main content inside the central column
    # Fetch available dates and configure date input
    available_dates = fetch_available_dates()

    if not available_dates:
        st.error("No available dates found in the database. Please check data availability and database connection.")
        st.stop() # Stop the app if no dates are available

    # Set min, max, and default value for the date input based on available dates
    min_available_date = min(available_dates) if available_dates else date.today()
    max_available_date = max(available_dates) if available_dates else date.today()
    # Set default date to the latest available date or today if within range
    default_date = max_available_date if max_available_date > date.today() else date.today()
    default_date = max_available_date if default_date < min_available_date else default_date

    # If no dates are available, default date might still be off.
    # If available_dates is empty, min/max will be today, default will be today.
    # The fetch_data will then return empty, handled below.

    # Ensure the default date is actually one of the available dates if possible
    if available_dates and default_date not in available_dates:
         # Find the closest available date, or just default to the latest
         default_date = max_available_date


    selected_date = st.date_input(
        "Select date",
        value=default_date,
        min_value=min_available_date,
        max_value=max_available_date,
        # We don't restrict to *only* available dates in the picker itself,
        # just the min/max range.
    )

    # Optional: Re-check if the selected date is in the list of available dates
    # if available_dates and selected_date not in available_dates:
    #     st.warning(f"Data may not be available for the exact date selected: {selected_date}. Displaying data for the closest available date or period.")
        # You might want to change selected_date here to an available date before fetching


    # Format the selected date to 'YYYY-MM-DD' string for the SQL query
    selected_date_str = selected_date.strftime('%Y-%m-%d')

    # --- FETCH AND DISPLAY DATA ---
    # Fetch the hourly data for the selected date
    data = fetch_data(selected_date_str)

    # Fetch the entity specific BCQ sums
    entity_sums_df = fetch_bcq_entity_sums(selected_date_str)

    if not data.empty:
        # --- Display Daily Summary Metrics as Cards ---
        st.subheader("Daily Summary Metrics")

        # Create three columns for the metrics using the default layout
        col1, col2, col3 = st.columns(3)

        # Display Maximum Price and Average Price in the first two data columns (col1 and col2)
        # Added checks for numeric type before calculating max/mean
        if "Prices" in data.columns and not data["Prices"].empty and pd.api.types.is_numeric_dtype(data["Prices"]):
            max_price = data["Prices"].max()
            avg_price = data["Prices"].mean()
            col1.metric(label="Maximum Price (PHP/kWh)", value=f"{max_price:,.2f}") # Format for readability
            col2.metric(label="Average Price (PHP/kWh)", value=f"{avg_price:,.2f}") # Format for readability
        else:
            col1.warning("Prices data not available or not numeric.")
            col2.warning("Avg Price data not available or not numeric.")


        # --- Display Maximum Total MQ and corresponding time in the third data column (col3) ---
        # Added checks for numeric type before calculation
        if "Total_MQ" in data.columns and "Time" in data.columns and not data["Total_MQ"].empty and pd.api.types.is_numeric_dtype(data["Total_MQ"]):
            # Find the row with the maximum Total_MQ value
            max_mq_value = data["Total_MQ"].max()
            # Check if the column is all NaNs or empty, idxmax would raise an error
            if pd.notnull(max_mq_value) and not data["Total_MQ"].isnull().all(): # Also check if all values are null
                 # Ensure 'Total_MQ' is numeric before idxmax - already checked above, but safe here
                 # if pd.api.types.is_numeric_dtype(data["Total_MQ"]):
                 max_mq_row_index = data["Total_MQ"].idxmax()
                 # Get the corresponding Time value from that row
                 max_mq_time = data.loc[max_mq_row_index, "Time"]

                 # Format the time for display
                 if pd.api.types.is_datetime64_any_dtype(max_mq_time):
                      max_mq_time_str = max_mq_time.strftime("%H:%M")
                 else:
                      # Handle cases where max_mq_time might not be a datetime object
                      max_mq_time_str = str(max_mq_time) # Display as string if not datetime

                 col3.metric(label="Maximum Total MQ (kWh)", value=f"{max_mq_value:,.2f}")
                 col3.write(f"at {max_mq_time_str}") # Display time below the metric
                 # else: # Warning already handled by outer if
                 #      col3.warning("Total_MQ column is not numeric.")

            else:
                 col3.info("Total_MQ data is all zero/null or not applicable for max metric.")

        else:
             col3.warning("Max MQ or Time data not available or not numeric or empty.")


        # --- Add WESM column (Total_BCQ - Total_MQ) to Hourly Summary ---
        # Calculate the WESM column if the required columns exist and are numeric
        if all(col in data.columns for col in ["Total_BCQ", "Total_MQ"]):
             if pd.api.types.is_numeric_dtype(data["Total_BCQ"]) and pd.api.types.is_numeric_dtype(data["Total_MQ"]):
                 data['WESM'] = data['Total_BCQ'] - data['Total_MQ']
             else:
                 st.warning("Could not calculate WESM column: Total_BCQ or Total_MQ are not numeric.")
        else:
             st.warning("Could not calculate WESM column: Total_BCQ or Total_MQ columns not found.")


        st.subheader("Hourly Summary")
        st.dataframe(data) # Display fetched data including the new WESM column


        # --- Create and Display BCQ Entity Pie Chart with WESM ---
        st.subheader("BCQ Contribution and WESM")

        # Prepare data for the pie chart
        pie_chart_data = pd.DataFrame(columns=['Entity', 'Sum'])

        if not entity_sums_df.empty:
             # Melt the entity sums dataframe
             # Use .iloc[0] to get the single row as a Series
             melted_entity_sums = entity_sums_df.iloc[0].melt(var_name='Entity', value_name='Sum')
             # Append to pie chart data
             pie_chart_data = pd.concat([pie_chart_data, melted_entity_sums], ignore_index=True)


        # Calculate daily Total_BCQ and Total_MQ for WESM
        daily_total_bcq = 0
        daily_total_mq = 0
        if "Total_BCQ" in data.columns and pd.api.types.is_numeric_dtype(data["Total_BCQ"]):
             daily_total_bcq = data["Total_BCQ"].sum()
        if "Total_MQ" in data.columns and pd.api.types.is_numeric_dtype(data["Total_MQ"]):
             daily_total_mq = data["Total_MQ"].sum()

        # Calculate WESM (Total_BCQ - Total_MQ) / -1000
        wesm_value = 0
        try:
            # Check if the divisor is zero before division
            if -1000 != 0:
                wesm_value = (daily_total_bcq - daily_total_mq) / -1000
            else:
                 st.warning("Cannot calculate WESM (division by zero).")
                 wesm_value = 0 # Set to 0 if division is not possible
        except Exception as e: # Catch any other potential errors during WESM calculation
            st.error(f"Error calculating WESM: {e}")
            wesm_value = 0 # Set to 0 on error


        # Add WESM to the pie chart data
        # Only add if there's other data or WESM is non-zero
        # Use a small tolerance for checking if WESM is effectively zero
        if not pie_chart_data.empty or abs(wesm_value) > 1e-9:
            wesm_row = pd.DataFrame([{'Entity': 'WESM', 'Sum': wesm_value}])
            pie_chart_data = pd.concat([pie_chart_data, wesm_row], ignore_index=True)


        # Ensure the 'Sum' column is numeric before plotting
        if 'Sum' in pie_chart_data.columns:
            pie_chart_data['Sum'] = pd.to_numeric(pie_chart_data['Sum'], errors='coerce').fillna(0)
            # Filter out entities with zero or NaN sum, unless it's WESM and non-zero
            pie_chart_data = pie_chart_data[
                (pie_chart_data['Sum'] != 0) |
                (pie_chart_data['Entity'] == 'WESM')
            ]
            # Ensure WESM is included if its value is non-zero, even if other sums are zero
            if 'WESM' in pie_chart_data['Entity'].tolist() and abs(wesm_value) > 1e-9 and 'WESM' not in pie_chart_data[pie_chart_data['Sum'] != 0]['Entity'].tolist():
                wesm_only_df = pd.DataFrame([{'Entity': 'WESM', 'Sum': wesm_value}])
                # Concatenate, making sure to not duplicate WESM if it was already included due to its value
                pie_chart_data = pd.concat([pie_chart_data[pie_chart_data['Entity'] != 'WESM'], wesm_only_df], ignore_index=True)


        if not pie_chart_data.empty:
            # Create the pie chart
            base = alt.Chart(pie_chart_data).encode(
                theta=alt.Theta("Sum", stack=True) # Use Sum for the angle of slices
            )

            # Specify the outer radius of the arcs and encode color based on the entity
            pie = base.mark_arc(outerRadius=120).encode(
                color=alt.Color("Entity"),
                # Order the arcs by `Sum` in descending order.
                order=alt.Order("Sum", sort="descending"),
                tooltip=["Entity", alt.Tooltip("Sum", format=",.1f")] # Add tooltips for Entity and Sum
            )

            # Add text labels
            text = base.mark_text(radius=140).encode( # Adjust radius to position text outside slices
                text=alt.Text("Entity"), # Use Entity name as the label
                order=alt.Order("Sum", sort="descending"),
                color=alt.value("black") # Set the color of the labels
            )

            # Combine the pie chart and text
            chart = pie + text

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Not enough data to display the BCQ contribution pie chart.")

    else:
        st.warning(f"No data available for selected date: {selected_date_str}.")

    # Removed the section that displayed generator_data as requested
    # if not generator_data.empty:
    #     st.subheader("🔌 Generator BCQ Summary")
    #     st.dataframe(generator_data)

    # --- End of main app content ---
