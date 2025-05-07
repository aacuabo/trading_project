import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, date
import altair as alt
import numpy as np # Import numpy for handling potential NaN sums
import plotly.graph_objects as go # Import Plotly for Sankey chart

# Set Streamlit page configuration
st.set_page_config(layout="wide") # Use wide layout for better display

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
        engine = create_engine(url, pool_pre_ping=True) # Corrected typo here
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
        # Fetch dates as DATE type if possible or cast, then convert to datetime.date
        query = """
            SELECT DISTINCT CAST("Date" AS DATE) AS "Date"
            FROM "MQ_Hourly"
            ORDER BY "Date";
        """
        # Use parse_dates to ensure the "Date" column is read as datetime
        # Ensure the column name here matches the one in the query (CAST AS "Date")
        dates_df = pd.read_sql(query, engine, parse_dates=["Date"])
        # Convert datetime objects to date objects
        available_dates = dates_df["Date"].dt.date.tolist()
        return available_dates
    except Exception as e:
        st.error(f"Error fetching available dates: {e}")
        return []


# Define mappings for component names (Using exact database column names as keys)
BCQ_SOURCE_MAP = {
    "FDC Misamis Power Corporation  (FDC)": 'FDC',
    "GNPower Kauswagan Ltd. Co. (GNPKLCO)": 'GNPK',
    "Power Sector Assets & Liabilities Management Corporation (PSALMGMIN)": 'PSALM',
    "Sarangani Energy Corporation (SEC)": 'SEC',
    "Therma South, Inc. (TSI)": 'TSI',
    "Malita Power Inc. (SMCPC)": 'MPI'
}

MQ_DEST_MAP = {
    "14BGN_T1L1_KIDCOTE01_NET": 'M1/M6/M8',
    "14BGN_T1L1_KIDCOTE02_NET": 'M2',
    "14BGN_T1L1_KIDCOTE03_NET": 'M3',
    "14BGN_T1L1_KIDCOTE04_NET": 'M4',
    "14BGN_T2L1_KIDCOTE05_NET": 'M5',
    "14BGN_T1L1_KIDCOTE08_NET": 'M7',
    "14BGN_T1L1_KIDCOTE10_NET": 'M9',
    "14BGN_T1L1_KIDCSCV01_DEL": 'KIDCSCV01_DEL',
    "14BGN_T1L1_KIDCSCV02_DEL": 'KIDCSCV02_DEL'
}


@st.cache_data(ttl=600) # Cache hourly data for 10 minutes
def fetch_data(selected_date_str: str): # Added type hint for caching key
    """
    Fetches hourly MQ, BCQ, Prices, and individual component data for a selected date.
    Includes error handling for potentially missing individual component columns.
    Uses exact database column names provided by the user.
    """
    try:
        engine = get_sqlalchemy_engine()

        # Define the columns we want to fetch, including individual components
        # Use exact column names from your database, including quotes if they have spaces or special characters
        mq_cols_raw = list(MQ_DEST_MAP.keys()) + ["Total_MQ"] # Include Total_MQ from MQ table
        bcq_cols_raw = list(BCQ_SOURCE_MAP.keys()) + ["Total_BCQ"] # Include Total_BCQ from BCQ table
        price_cols_raw = ["Prices"] # Assuming Prices is in a separate table

        # Quote column names for SQL query if they contain special characters or spaces
        mq_cols_quoted = [f'"{col}"' for col in mq_cols_raw]
        bcq_cols_quoted = [f'"{col}"' for col in bcq_cols_raw]
        price_cols_quoted = [f'"{col}"' for col in price_cols_raw]

        # Select columns, aliasing the BCQ columns to their simpler names for easier pandas handling
        select_cols = [f'mq.{col}' for col in mq_cols_quoted]

        # Corrected logic for aliasing BCQ columns to avoid SyntaxError
        for raw_col, alias in BCQ_SOURCE_MAP.items():
            quoted_col = f'"{raw_col}"' # Get the quoted version of the raw column name
            # Add the select statement with alias: bcq."Raw Column Name" AS "Alias"
            select_cols.append(f'bcq.{quoted_col} AS "{alias}"')

        # Add the Total_BCQ column separately if it's not in the map keys (it is, but for clarity)
        # Check if the raw Total_BCQ column name is in the original list and not aliased in the loop
        if "Total_BCQ" in bcq_cols_raw and "Total_BCQ" not in BCQ_SOURCE_MAP.values():
             select_cols.append('bcq."Total_BCQ"') # Add it without aliasing

        select_cols.extend([f'p.{col}' for col in price_cols_quoted])


        query = f"""
            SELECT
                mq."Date",
                mq."Time",
                {', '.join(select_cols)}
            FROM "MQ_Hourly" AS mq
            JOIN "BCQ_Hourly" AS bcq ON mq."Date" = bcq."Date" AND mq."Time" = bcq."Time"
            JOIN "Prices_Hourly" AS p ON mq."Date" = p."Date" AND mq."Time" = p."Time"
            WHERE mq."Date" = %s
            ORDER BY mq."Time";
        """

        # Use parse_dates for known datetime columns
        # Fix 3: Corrected params usage
        df = pd.read_sql(query, engine, params=[selected_date_str], parse_dates=["Date", "Time"])

        # Ensure Time is treated as a datetime object by combining with Date if necessary
        if not df.empty and 'Date' in df.columns and 'Time' in df.columns:
             try:
                 if not pd.api.types.is_datetime64_any_dtype(df['Time']):
                     date_str_col = df['Date'].dt.strftime('%Y-%m-%d') if pd.api.types.is_datetime64_any_dtype(df['Date']) else df['Date'].astype(str)
                     time_str_col = df['Time'].dt.strftime('%H:%M:%S') if pd.api.types.is_datetime64_any_dtype(df['Time']) else df['Time'].astype(str)
                     df['Datetime'] = pd.to_datetime(date_str_col + ' ' + time_str_col, errors='coerce')
                     df.dropna(subset=['Datetime'], inplace=True)
                     df['Time'] = df['Datetime']

                 # Drop the original 'Date' column if 'Datetime' was created
                 if 'Datetime' in df.columns:
                     df.drop(columns=['Date', 'Datetime'], errors='ignore', inplace=True)
                 elif 'Date' in df.columns:
                      df.drop(columns=['Date'], errors='ignore', inplace=True)


             except Exception as e:
                 st.error(f"Error converting 'Time' column to datetime after fetch: {e}. Please check the format of the 'Date' and 'Time' columns in your database.")
                 return pd.DataFrame()

        # Ensure all fetched columns are numeric where expected, coercing errors to NaN
        # Use the raw MQ column names and the ALIASED BCQ column names for DataFrame operations
        all_numeric_cols = mq_cols_raw + list(BCQ_SOURCE_MAP.values()) + price_cols_raw # Use values() for BCQ aliases
        for col in all_numeric_cols:
             if col in df.columns:
                  df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where the 'Time' column is missing after conversion
        if 'Time' in df.columns:
             df.dropna(subset=['Time'], inplace=True)


        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        # Provide more specific error message if a column might be missing
        if "column" in str(e) and "does not exist" in str(e):
             missing_col_hint = str(e).split('"')[-2] # Attempt to extract the column name
             st.error(f"Database column '{missing_col_hint}' not found. Ensure column names in your database tables match the names specified in the code for MQ Units and BCQ Generators.")
        return pd.DataFrame()


# --- STREAMLIT UI (Now always displayed) ---

st.title("📊 Daily Energy Trading Dashboard") # Main title

# Create outer columns for centering the main content
spacer_left, main_content, spacer_right = st.columns([1, 4, 1])

with main_content: # Place all the main content inside the central column
    # Fetch available dates and configure date input
    available_dates = fetch_available_dates()

    if not available_dates:
        st.error("No available dates found in the database. Please check data availability and database connection.")
        st.stop() # Stop the app if no dates are available

    # Set min, max, and default value for the date input based on available dates
    min_available_date = min(available_dates)
    max_available_date = max(available_dates)
    default_date = max_available_date

    selected_date = st.date_input(
        "Select date",
        value=default_date,
        min_value=min_available_date,
        max_value=max_available_date,
    )

    # Format the selected date to 'YYYY-MM-DD' string for the SQL query and caching key
    selected_date_str = selected_date.strftime('%Y-%m-%d')

    # --- FETCH AND DISPLAY DATA ---
    # Fetch the hourly data for the selected date
    data = fetch_data(selected_date_str)

    if not data.empty:
        # --- Display Daily Summary Metrics as Cards ---
        st.subheader("Daily Summary Metrics")

        col1, col2, col3 = st.columns(3)

        # Display Maximum Price and Average Price
        if "Prices" in data.columns and not data["Prices"].empty and pd.api.types.is_numeric_dtype(data["Prices"]):
            max_price = data["Prices"].max()
            avg_price = data["Prices"].mean()
            if pd.notnull(max_price):
                 col1.metric(label="Maximum Price (PHP/kWh)", value=f"{max_price:,.2f}")
            else:
                 col1.info("Maximum Price is not available.")

            if pd.notnull(avg_price):
                 col2.metric(label="Average Price (PHP/kWh)", value=f"{avg_price:,.2f}")
            else:
                 col2.info("Average Price is not available.")
        else:
            col1.warning("Prices data not available or not numeric for metrics.")
            col2.warning("Avg Price data not available or not numeric for metrics.")

        # --- Display Maximum Total MQ and corresponding time ---
        if "Total_MQ" in data.columns and "Time" in data.columns and not data["Total_MQ"].empty and pd.api.types.is_numeric_dtype(data["Total_MQ"]) and pd.api.types.is_datetime64_any_dtype(data["Time"]):
            if pd.notnull(data["Total_MQ"]).any():
                 max_mq_value = data["Total_MQ"].max()
                 max_mq_row_index = data["Total_MQ"].idxmax()
                 max_mq_time = data.loc[max_mq_row_index, "Time"]
                 max_mq_time_str = max_mq_time.strftime("%H:%M")
                 col3.metric(label="Maximum Total MQ (kWh)", value=f"{max_mq_value:,.2f}")
                 col3.write(f"at {max_mq_time_str}")
            else:
                 col3.info("Total_MQ data contains no valid numbers for maximum metric.")
        else:
             col3.warning("Max MQ or Time data not available, not numeric, empty, or Time is not datetime.")

        # --- Add WESM column (Total_BCQ - Total_MQ) ---
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

        if 'Time' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Time']):

            columns_to_melt = ["Total_MQ", "Total_BCQ", "Prices"]

            if all(col in data.columns and pd.api.types.is_numeric_dtype(data[col]) for col in columns_to_melt):
                melted_data = data.melt(
                    id_vars=["Time"],
                    value_vars=columns_to_melt,
                    var_name="Metric",
                    value_name="Value"
                )

                chart_energy_data = melted_data[melted_data["Metric"].isin(["Total_MQ", "Total_BCQ"])].dropna(subset=['Value'])
                chart_energy = alt.Chart(chart_energy_data).mark_line(point=True).encode(
                    x=alt.X("Time", axis=alt.Axis(title="Time", format="%H:%M")),
                    y=alt.Y("Value", title="Energy (kWh)", axis=alt.Axis(titleColor="tab:blue"), scale=alt.Scale(zero=True)),
                    color=alt.Color(
                        "Metric",
                        legend=alt.Legend(title="Metric", orient='bottom'),
                        scale=alt.Scale(domain=['Total_MQ', 'Total_BCQ'], range=['#FFC20A', '#1A85FF']) # Yellow, Blue
                    ),
                    tooltip=[alt.Tooltip("Time", format="%Y-%m-%d %H:%M"), "Metric", alt.Tooltip("Value", format=".2f")]
                ).properties(
                )

                chart_price_data = melted_data[melted_data["Metric"] == "Prices"].dropna(subset=['Value'])
                chart_price = alt.Chart(chart_price_data).mark_bar(color="#40B0A6").encode( # Apple green
                    x=alt.X("Time", axis=alt.Axis(title="")),
                    y=alt.Y("Value", title="Price (PHP/kWh)", axis=alt.Axis(titleColor="tab:red"), scale=alt.Scale(zero=True)),
                    tooltip=[alt.Tooltip("Time", format="%Y-%m-%d %H:%M"), "Metric", alt.Tooltip("Value", format=".2f")]
                ).properties(
                )

                combined_chart = alt.layer(chart_price, chart_energy).resolve_scale(
                    y='independent'
                ).properties(
                    title=f"Energy Metrics and Prices for {selected_date_str}"
                ).interactive()

                st.altair_chart(combined_chart, use_container_width=True)
            else:
                missing_or_non_numeric_cols = [col for col in columns_to_melt if col not in data.columns or (col in data.columns and not pd.api.types.is_numeric_dtype(data[col]))]
                st.warning(f"Data fetched but required columns for line/bar plotting are missing or not numeric: {missing_or_non_numeric_cols}. Check your database tables.")

        else:
            st.warning("Time column is not in the expected datetime format for plotting or data is empty after fetch.")


        # --- SANKEY CHART: Energy Flow ---
        st.subheader("🔗 Daily Energy Flow (Sankey Diagram)")

        st.info("""
            This Sankey diagram visualizes a conceptual energy flow based on your specified components and formula.
            Flows originate from individual BCQ Generators and WESM, combine into a 'Combined Input' node scaled per your formula,
            and then distribute proportionally to individual MQ Units.
            Note: The link values are derived from the daily sums and your scaling formula, not direct individual flows.
        """)

        # Calculate daily sums for individual components and totals
        daily_sums = {}
        # Use raw column names from the dictionaries as keys to access DataFrame columns
        all_component_cols_raw = list(BCQ_SOURCE_MAP.keys()) + list(MQ_DEST_MAP.keys()) + ['Total_BCQ', 'Total_MQ', 'WESM']

        for col in all_component_cols_raw:
            # Check if column exists and is numeric before summing
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                 daily_sums[col] = data[col].sum()
            else:
                 daily_sums[col] = 0 # Default to 0 if column is missing or not numeric


        daily_total_bcq_sum = daily_sums.get('Total_BCQ', 0)
        daily_total_mq_sum = daily_sums.get('Total_MQ', 0)
        daily_wesm_sum = daily_sums.get('WESM', 0)

        # Calculate the value of the RHS of the user's equation for context
        rhs_equation_value = (daily_total_bcq_sum * 1000) - daily_wesm_sum

        st.write(f"Daily Total MQ (Sum of Destinations): {daily_total_mq_sum:,.2f} kWh")
        st.write(f"Daily Total BCQ (Sum of Sources): {daily_total_bcq_sum:,.2f} kWh")
        st.write(f"Daily Total WESM: {daily_wesm_sum:,.2f} kWh")
        st.write(f"Value of RHS of Equation (Total_BCQ * 1000 - WESM): {rhs_equation_value:,.2f} kWh")
        st.write(f"Difference (Daily Total MQ - RHS): {(daily_total_mq_sum - rhs_equation_value):,.2f} kWh")


        # Check if there is any data in the components to build the Sankey chart
        any_component_data = False
        for col in list(BCQ_SOURCE_MAP.keys()) + list(MQ_DEST_MAP.keys()):
            if daily_sums.get(col, 0) != 0:
                 any_component_data = True
                 break
        # Also check if WESM contributes
        if daily_wesm_sum != 0:
             any_component_data = True


        if not any_component_data:
             st.info(f"No relevant component data available to build the Sankey diagram for {selected_date_str}.")
        else:

            # Define nodes for the Sankey diagram
            node_labels = []
            # Add BCQ source nodes that have non-zero sums
            bcq_source_nodes_present = [name for col, name in BCQ_SOURCE_MAP.items() if daily_sums.get(col, 0) != 0]
            node_labels.extend(bcq_source_nodes_present)

            # Add WESM node if WESM sum is not zero
            if daily_wesm_sum != 0:
                 node_labels.append('WESM')

            # Add intermediate nodes - only if there's flow into or out of them
            include_total_bcq_input = daily_total_bcq_sum != 0 or any(daily_sums.get(col, 0) != 0 for col in BCQ_SOURCE_MAP.keys())
            include_total_wesm_input = daily_wesm_sum != 0
            scaled_total_bcq = daily_total_bcq_sum * 1000
            scaled_wesm = daily_wesm_sum * -1
            combined_input_sum = scaled_total_bcq + scaled_wesm
            include_combined_input = combined_input_sum != 0 or any(daily_sums.get(col, 0) != 0 for col in MQ_DEST_MAP.keys()) # Also if destinations exist

            if include_total_bcq_input:
                 node_labels.append('Total BCQ Input')
            if include_total_wesm_input:
                 node_labels.append('Total WESM Input')
            if include_combined_input:
                 node_labels.append('Combined Input')


            # Add MQ destination nodes that have non-zero sums
            mq_dest_nodes_present = [name for col, name in MQ_DEST_MAP.items() if daily_sums.get(col, 0) != 0]
            node_labels.extend(mq_dest_nodes_present)


            # Create a mapping from node label to index
            label_to_index = {label: i for i, label in enumerate(node_labels)}

            # --- DEBUGGING: Display Node Labels and Index Mapping ---
            st.write("Sankey Node Labels:", node_labels)
            st.write("Sankey Label to Index Mapping:", label_to_index)


            # Define links for the Sankey diagram
            sources = []
            targets = []
            values = []
            link_labels = [] # Optional: labels for links

            # Links from individual BCQ sources to 'Total BCQ Input'
            if 'Total BCQ Input' in label_to_index: # Only create links if the target node exists
                for col, name in BCQ_SOURCE_MAP.items():
                     value = daily_sums.get(col, 0)
                     if value != 0 and name in label_to_index: # Ensure source node exists and has value
                          sources.append(label_to_index[name])
                          targets.append(label_to_index['Total BCQ Input'])
                          values.append(abs(value)) # Use absolute value for link thickness
                          link_labels.append(f'{name} to Total BCQ Input: {value:,.2f}')


            # Link from 'WESM' to 'Total WESM Input'
            if daily_wesm_sum != 0 and 'WESM' in label_to_index and 'Total WESM Input' in label_to_index:
                 sources.append(label_to_index['WESM'])
                 targets.append(label_to_index['Total WESM Input'])
                 values.append(abs(daily_wesm_sum)) # Use absolute value
                 link_labels.append(f'WESM to Total WESM Input: {daily_wesm_sum:,.2f}')

            # Link from 'Total BCQ Input' to 'Combined Input' (scaled)
            # Check if both source and target nodes exist and the scaled value is non-zero
            if 'Total BCQ Input' in label_to_index and 'Combined Input' in label_to_index and scaled_total_bcq != 0:
                 sources.append(label_to_index['Total BCQ Input'])
                 targets.append(label_to_index['Combined Input'])
                 values.append(abs(scaled_total_bcq)) # Use absolute value
                 link_labels.append(f'Total BCQ Input to Combined Input (x1000): {scaled_total_bcq:,.2f}')


            # Link from 'Total WESM Input' to 'Combined Input' (scaled)
            # Check if both source and target nodes exist and the scaled value is non-zero
            if 'Total WESM Input' in label_to_index and 'Combined Input' in label_to_index and scaled_wesm != 0:
                 sources.append(label_to_index['Total WESM Input'])
                 targets.append(label_to_index['Combined Input'])
                 values.append(abs(scaled_wesm)) # Use absolute value
                 link_labels.append(f'Total WESM Input to Combined Input (x-1): {scaled_wesm:,.2f}')

            # Links from 'Combined Input' to individual MQ destinations (proportional distribution)
            if 'Combined Input' in label_to_index and combined_input_sum != 0: # Check if source node exists and there's flow
                 # Calculate the sum of daily MQ destination sums for proportional distribution
                 total_mq_dest_sum = sum(daily_sums.get(col, 0) for col in MQ_DEST_MAP.keys())

                 if total_mq_dest_sum != 0:
                      for col, name in MQ_DEST_MAP.items():
                           daily_mq_dest_sum = daily_sums.get(col, 0)
                           # Only create a link to this destination if it exists as a node and has a non-zero sum
                           if daily_mq_dest_sum != 0 and name in label_to_index:
                                # Calculate the proportional value flowing to this destination
                                # Use the actual combined_input_sum (can be positive or negative) for proportional calculation
                                proportional_value = (daily_mq_dest_sum / total_mq_dest_sum) * combined_input_sum
                                if proportional_value != 0: # Only create link if there's flow
                                     sources.append(label_to_index['Combined Input'])
                                     targets.append(label_to_index[name])
                                     values.append(abs(proportional_value)) # Use absolute value for thickness
                                     link_labels.append(f'Combined Input to {name}: {proportional_value:,.2f}')
                 # else: This case is handled by the outer 'if not any_component_data:' check

            # --- DEBUGGING: Display Link Data ---
            st.write("Sankey Sources:", sources)
            st.write("Sankey Targets:", targets)
            st.write("Sankey Values:", values)
            st.write("Sankey Link Labels:", link_labels)


            # Create the Sankey diagram using Plotly ONLY if there are links
            if sources and targets and values and len(sources) == len(targets) == len(values) and len(sources) > 0: # Added check for empty lists
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=node_labels,
                        # color="blue" # Optional: set node color
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        label=link_labels, # Display values on hover
                        # color="rgba(0,0,255,0.1)" # Optional: set link color
                ))])

                fig.update_layout(
                    title_text=f"Daily Energy Flow for {selected_date_str}",
                    font_size=10,
                    height=600 # Adjust height as needed
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No valid links generated for the Sankey diagram. Check data and calculations.")


    else:
        st.warning(f"No data available for selected date: {selected_date_str}.")
