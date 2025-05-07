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


# --- STREAMLIT UI (Now always displayed) ---

# Removed: Login widget
# Removed: Display content based on authentication status blocks (if/elif)
# Removed: Logout button

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

    # While min_value and max_value restrict the picker range,
    # a direct selection might still result in a date not in the DISTINCT list.
    # Re-check and inform the user if the selected date is not in the exact list.
    if selected_date not in available_dates:
        st.warning(f"Data may not be available for the exact date selected: {selected_date}. Displaying data for the closest available date or period.")
        # For simplicity here, we'll still use the selected date for fetching,
        # but the warning lets the user know. If strict adherence is needed,
        # you'd fallback to a date in available_dates here before fetching.
        # Example: selected_date_str = max_available_date.strftime('%Y-%m-%d')


    # Format the selected date to 'YYYY-MM-DD' string for the SQL query
    selected_date_str = selected_date.strftime('%Y-%m-%d')

    # --- FETCH AND DISPLAY DATA ---
    # Fetch the hourly data for the selected date
    data = fetch_data(selected_date_str)


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


        # --- Add WESM column (Total_BCQ - Total_MQ) ---
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


        # --- PLOT DATA USING ALTAIR FOR INTERACTIVITY ---
        st.subheader("📈 Energy Metrics Over Time (Interactive)")

        # --- DEBUGGING: Check columns before melting ---
        # Use this to see the actual column names returned by the database.
        # You can comment this line out once you've confirmed the column names
        # and updated the columns_to_melt list below.
        # st.write("Columns in data DataFrame:", data.columns.tolist()) # Use .tolist() for clearer display

        # Melt the DataFrame for Altair - necessary for plotting multiple metrics
        # on the same or twin axes easily with color encoding.
        # Ensure 'Time' is a datetime type before melting
        if 'Time' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Time']):

            # --- IMPORTANT: VERIFY AND UPDATE COLUMN NAMES HERE if necessary ---
            # The list below must contain the EXACT names of the columns in your
            # 'data' DataFrame that you want to plot (Total_MQ, Total_BCQ, Prices).
            # Check the output of the commented-out "st.write" line above if you
            # are still facing KeyError during melting.
            columns_to_melt = ["Total_MQ", "Total_BCQ", "Prices"] # <-- **VERIFY/UPDATE THESE NAMES if needed**

            # Check if all columns to melt exist in the DataFrame
            if all(col in data.columns for col in columns_to_melt):
                melted_data = data.melt(
                    id_vars=["Time"], # Use id_vars to specify identifier columns
                    value_vars=columns_to_melt, # Columns to unpivot
                    var_name="Metric", # Name for the new column holding metric names
                    value_name="Value" # Name for the new new column holding metric values
                )

                # Debugging: Display melted data
                # st.subheader("Melted Data for Plotting")
                # st.dataframe(melted_data)
                # st.write("Melted Data Types:", melted_data.dtypes)


                # Create charts for the left y-axis (MQ and BCQ) - as lines
                chart_energy = alt.Chart(melted_data[melted_data["Metric"].isin(["Total_MQ", "Total_BCQ"])]).mark_line(point=True).encode(
                    x=alt.X("Time", axis=alt.Axis(title="Time", format="%H:%M")), # Format time axis
                    # --- Align zero for the energy axis - 'zero=True' goes inside alt.Scale() ---
                    y=alt.Y("Value", title="Energy (kWh)", axis=alt.Axis(titleColor="tab:blue"), scale=alt.Scale(zero=True)),
                    # --- Use specified colors for MQ and BCQ ---
                    # --- Move legend to the bottom ---
                    color=alt.Color(
                        "Metric",
                        legend=alt.Legend(title="Metric", orient='bottom'),
                        scale=alt.Scale(domain=['Total_MQ', 'Total_BCQ'], range=['#FFC20A', '#1A85FF']) # Set specific colors
                    ),
                    tooltip=[alt.Tooltip("Time", format="%Y-%m-%d %H:%M"), "Metric", "Value"] # Add tooltips with formatted time
                ).properties(
                     title="Energy Metrics" # Title for this layer's legend
                )

                # Create chart for the right y-axis (Prices) - as bars
                # --- Change the color of the price bars to apple green (#40B0A6 is a pleasant shade) ---
                chart_price = alt.Chart(melted_data[melted_data["Metric"] == "Prices"]).mark_bar(color="#40B0A6").encode(
                    x=alt.X("Time", axis=alt.Axis(title="")), # Empty title as it's shared
                    # --- Align zero for the price axis - 'zero=True' goes inside alt.Scale() ---
                    y=alt.Y("Value", title="Price (PHP/kWh)", axis=alt.Axis(titleColor="tab:red"), scale=alt.Scale(zero=True)),
                    tooltip=[alt.Tooltip("Time", format="%Y-%m-%d %H:%M"), "Metric", "Value"] # Add tooltips with formatted time
                ).properties(
                     title="Prices" # Title for this layer's legend
                )

                # Combine the charts with independent y-axes
                # --- Reverse the layering order to put bars behind lines ---
                # List chart_price first to draw its bars at the bottom, then chart_energy lines on top.
                combined_chart = alt.layer(chart_price, chart_energy).resolve_scale(
                    y='independent' # Allow y-axes to have different scales
                ).properties(
                    title=f"Energy Metrics and Prices for {selected_date_str}"
                ).interactive() # Add interactivity for zooming and panning

                # Display the chart in Streamlit
                st.altair_chart(combined_chart, use_container_width=True)
            else:
                # If columns are missing for plotting, display an informative warning
                missing_cols = [col for col in columns_to_melt if col not in data.columns]
                st.warning(f"Data fetched but required columns for plotting are missing: {missing_cols}. Check your database tables ('MQ_Hourly', 'BCQ_Hourly', 'Prices_Hourly') and SQL query result.")


        else:
            st.warning("Time column is not in the expected datetime format for plotting or data is empty after fetch.")


    else:
        st.warning(f"No data available for selected date: {selected_date_str}.")

    # Removed the section that displayed generator_data as requested
    # if not generator_data.empty:
    #     st.subheader("🔌 Generator BCQ Summary")
    #     st.dataframe(generator_data)
