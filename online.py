import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, date
import altair as alt
import numpy as np # Import numpy for handling potential NaN sums
# import altair_ally # Keep commented for now

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
# Define the columns for individual MQ units and BCQ generators and their desired display aliases
MQ_UNIT_COLUMNS = {
    "14BGN_T1L1_KIDCOTE01_NET": 'M1,M6,M8',
    "14BGN_T1L1_KIDCOTE02_NET": 'M2',
    "14BGN_T1L1_KIDCOTE03_NET": 'M3',
    "14BGN_T1L1_KIDCOTE04_NET": 'M4',
    "14BGN_T2L1_KIDCOTE05_NET": 'M5',
    "14BGN_T1L1_KIDCOTE08_NET": 'M7',
    "14BGN_T1L1_KIDCOTE10_NET": 'M9',
    "14BGN_T1L1_KIDCSCV01_DEL": 'KIDCSCV01', # Assuming these names are in the DB
    "14BGN_T1L1_KIDCSCV02_DEL": 'KIDCSCV02', # Assuming these names are in the DB
}

BCQ_GENERATOR_COLUMNS = {
    "FDC Misamis Power Corporation  (FDC)": 'FDC',
    "GNPower Kauswagan Ltd. Co. (GNPKLCO)": 'GNPK',
    "Power Sector Assets & Liabilities Management Corporation (PSALMGMIN)": 'PSALM',
    "Sarangani Energy Corporation (SEC)": 'SEC',
    "Therma South, Inc. (TSI)": 'TSI',
    "Malita Power Inc. (SMCPC)": 'MPI',
}

@st.cache_data(ttl=3600) # Cache available dates for an hour
def fetch_available_dates():
    """Fetches a list of unique dates available in the database."""
    try:
        engine = get_sqlalchemy_engine()
        query = """
            SELECT DISTINCT CAST("Date" AS DATE) AS "Date"
            FROM "MQ_Hourly"
            ORDER BY "Date";
        """
        dates_df = pd.read_sql(query, engine, parse_dates=["Date"])
        available_dates = dates_df["Date"].dt.date.tolist()
        return available_dates
    except Exception as e:
        st.error(f"Error fetching available dates: {e}")
        return []


@st.cache_data(ttl=600) # Cache hourly data for 10 minutes
def fetch_data(selected_date_str: str): # Added type hint for caching key
    """Fetches hourly MQ, BCQ, Prices, and individual generator/unit data for a selected date."""
    try:
        engine = get_sqlalchemy_engine()

        # Construct the list of columns to select
        cols_to_select = [
            'mq."Date"', 'mq."Time"', 'mq."Total_MQ"', 'bcq."Total_BCQ"', 'p."Prices"'
        ]
        # Add individual MQ unit columns
        for col_name in MQ_UNIT_COLUMNS.keys():
             # Add quotes around column names just in case
             cols_to_select.append(f'mq."{col_name}"')

        # Add individual BCQ generator columns, using aliases in the query
        bcq_select_aliases = []
        for col_name, alias in BCQ_GENERATOR_COLUMNS.items():
             # Quote original column name and alias it
             cols_to_select.append(f'bcq."{col_name}" AS "{alias}"')
             bcq_select_aliases.append(alias)


        query = f"""
            SELECT {', '.join(cols_to_select)}
            FROM "MQ_Hourly" AS mq
            JOIN "BCQ_Hourly" AS bcq ON mq."Date" = bcq."Date" AND mq."Time" = bcq."Time"
            JOIN "Prices_Hourly" AS p ON mq."Date" = p."Date" AND mq."Time" = p."Time"
            WHERE mq."Date" = %s
            ORDER BY
                mq."Time"
        """
        df = pd.read_sql(query, engine, params=[(selected_date_str,)], parse_dates=["Date", "Time"])

        # Ensure Time is treated as a datetime object
        if not df.empty and 'Date' in df.columns and 'Time' in df.columns:
             try:
                 if not pd.api.types.is_datetime64_any_dtype(df['Time']):
                     date_str_col = df['Date'].dt.strftime('%Y-%m-%d') if pd.api.types.is_datetime64_any_dtype(df['Date']) else df['Date'].astype(str)
                     time_str_col = df['Time'].dt.strftime('%H:%M:%S') if pd.api.types.is_datetime64_any_dtype(df['Time']) else df['Time'].astype(str)
                     df['Datetime'] = pd.to_datetime(date_str_col + ' ' + time_str_col, errors='coerce')
                     df.dropna(subset=['Datetime'], inplace=True)
                     df['Time'] = df['Datetime']

                 if 'Date' in df.columns:
                     df = df.drop(columns=['Date'])

             except Exception as e:
                 st.error(f"Error converting 'Time' column to datetime after fetch: {e}.")
                 return pd.DataFrame()


        # Ensure all relevant columns are numeric after fetching/parsing
        all_value_cols = ["Total_MQ", "Total_BCQ", "Prices"] + list(MQ_UNIT_COLUMNS.keys()) + list(BCQ_GENERATOR_COLUMNS.values()) # Use values() for aliases here
        for col in all_value_cols:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where essential columns like Time or Total_MQ/BCQ are missing after conversion
        essential_cols = ['Time', 'Total_MQ', 'Total_BCQ']
        if all(col in df.columns for col in essential_cols):
             df.dropna(subset=essential_cols, inplace=True)


        return df
    except Exception as e:
        st.error(f"Error fetching data, including individual generator data: {e}")
        if "column" in str(e) and "does not exist" in str(e):
             st.error("Database column not found. Ensure columns for individual MQ units and BCQ generators exist and names match in your database tables.")
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
            col1.warning("Prices data not available or not numeric.")
            col2.warning("Avg Price data not available or not numeric.")


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


        # --- HOURLY LINE/BAR CHART ---
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
                        scale=alt.Scale(domain=['Total_MQ', 'Total_BCQ'], range=['#FFC20A', '#1A85FF'])
                    ),
                    tooltip=[alt.Tooltip("Time", format="%Y-%m-%d %H:%M"), "Metric", alt.Tooltip("Value", format=".2f")]
                ).properties(
                )

                chart_price_data = melted_data[melted_data["Metric"] == "Prices"].dropna(subset=['Value'])
                chart_price = alt.Chart(chart_price_data).mark_bar(color="#40B0A6").encode(
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
                st.warning(f"Data fetched but required columns for line/bar plotting are missing or not numeric: {missing_or_non_numeric_cols}.")

        else:
            st.warning("Time column is not in the expected datetime format for plotting or data is empty after fetch.")


        # --- SANKEY CHART: Flow from BCQ Generators to MQ Units ---
        st.subheader("📊 Daily Energy Flow (Conceptual Sankey Chart)")

        st.info("This Sankey chart visualizes a conceptual flow from individual BCQ Generators (Sources) to individual MQ Units (Destinations). The total volume flowing into the destinations (MQ Units) is their daily total (Daily_Total_MQ). The flow from each source BCQ Generator is distributed proportionally to each destination MQ Unit.")

        # Calculate daily totals for all components
        daily_totals = data.sum(numeric_only=True)

        # Ensure Total_MQ, Total_BCQ, and WESM totals are available
        daily_mq_total_sum = daily_totals.get("Total_MQ", 0)
        daily_bcq_total_sum = daily_totals.get("Total_BCQ", 0)
        daily_wesm_total_sum = daily_totals.get("WESM", 0) # WESM is already BCQ - MQ

        # Calculate the value of the RHS of the user's equation for context
        rhs_equation_value = (daily_bcq_total_sum * 1000) - daily_wesm_total_sum

        st.write(f"Daily Total MQ (Destination Sum): {daily_mq_total_sum:,.2f} kWh")
        st.write(f"Daily Total BCQ (Source Sum): {daily_bcq_total_sum:,.2f} kWh")
        st.write(f"Daily Total WESM: {daily_wesm_total_sum:,.2f} kWh")
        st.write(f"Value of RHS of Equation (Total_BCQ * 1000 - WESM): {rhs_equation_value:,.2f} kWh")
        st.write(f"Difference (Daily Total MQ - RHS): {(daily_mq_total_sum - rhs_equation_value):,.2f} kWh")


        links_data = []
        mq_available_totals = {}
        bcq_available_totals = {}

        # Get daily totals for available MQ unit columns (destinations)
        # Use original column names from fetch_data as keys, map to aliases for display
        for col_name, alias in MQ_UNIT_COLUMNS.items():
             if col_name in daily_totals:
                 mq_available_totals[alias] = daily_totals[col_name]

        # Get daily totals for available BCQ generator columns (sources)
        # Use aliases from fetch_data query as keys
        for col_name, alias in BCQ_GENERATOR_COLUMNS.items():
             if alias in daily_totals:
                 bcq_available_totals[alias] = daily_totals[alias]


        total_available_mq_units_sum = sum(mq_available_totals.values())
        total_available_bcq_gens_sum = sum(bcq_available_totals.values())


        # Create links data for the Sankey chart
        # Flow from BCQ Sources to MQ Destinations
        # Only create links if there are positive totals in both source and target components
        if total_available_bcq_gens_sum > 0 and total_available_mq_units_sum > 0:
            # Avoid division by zero in proportionality calculation
            for bcq_alias, bcq_total in bcq_available_totals.items():
                for mq_alias, mq_total in mq_available_totals.items():
                     # Calculate flow value based on proportionality
                     # Flow(bcq_gen_j -> mq_unit_i) = Daily_Total_mq_unit_i * (Daily_Total_bcq_gen_j / Total_Available_BCQ_Gens_Sum)
                     # This ensures that the sum of flows TO each mq_unit_i is its total,
                     # and the sum of flows FROM each bcq_gen_j is proportional to its share of the total BCQ.
                     if total_available_bcq_gens_sum > 0: # Re-check to be safe
                        flow_value = mq_total * (bcq_total / total_available_bcq_gens_sum)
                     else:
                        flow_value = 0 # Should not happen if outer if passes

                     if flow_value > 0: # Only add links with positive flow
                        links_data.append({'source': bcq_alias, 'target': mq_alias, 'value': flow_value})

            links_df = pd.DataFrame(links_data)

            if not links_df.empty:
                # --- Altair Sankey Chart Implementation ---
                # Similar approach to Alluvial, using mark_trail for bands and mark_text for labels.

                # Prepare data for bands
                melted_links_source = links_df.copy()
                melted_links_source['stage'] = 0 # Source stage (BCQ)
                melted_links_source['node'] = melted_links_source['source']
                melted_links_source['band_group'] = melted_links_source['source'] + '->' + melted_links_source['target']

                melted_links_target = links_df.copy()
                melted_links_target['stage'] = 1 # Target stage (MQ)
                melted_links_target['node'] = melted_links_target['target']
                melted_links_target['band_group'] = melted_links_target['source'] + '->' + melted_links_target['target']

                # Combine source and target points for each band
                sankey_data = pd.concat([melted_links_source, melted_links_target])

                # Sort data for correct stacking and band drawing
                # Sort by stage first, then by source, then by target for consistent banding
                sankey_data = sankey_data.sort_values(by=['stage', 'source', 'target']).reset_index(drop=True)

                # Sankey bands (using mark_trail)
                bands = alt.Chart(sankey_data).mark_trail().encode(
                    x=alt.X('stage', axis=None), # X-axis represents the stage (Source/Target)
                    # Y-axis represents the stacked value. Need to stack 'value' within each stage.
                    y=alt.Y('value', stack='zero', axis=None), # Stack values at each stage
                    detail='band_group', # Group paths by the band
                    # Color the bands by the source node (BCQ Generator)
                    color=alt.Color('source', title='BCQ Generator Source'),
                    opacity=alt.Opacity('value', legend=None), # Opacity based on flow value
                    tooltip=[
                        alt.Tooltip('source', title='Source BCQ'),
                        alt.Tooltip('target', title='Destination MQ'),
                        alt.Tooltip('value', title='Flow (kWh)', format=".2f")
                    ]
                )

                # Create node labels
                # Need to get the total value per node at each stage
                # Use the mq_available_totals and bcq_available_totals for node totals directly
                node_totals_list = []
                for alias, total in bcq_available_totals.items():
                    node_totals_list.append({'node': alias, 'stage': 0, 'value': total, 'stage_name': 'Sources'})
                for alias, total in mq_available_totals.items():
                    node_totals_list.append({'node': alias, 'stage': 1, 'value': total, 'stage_name': 'Destinations'})
                node_totals_df = pd.DataFrame(node_totals_list)


                # Calculate cumulative stack position for labels
                node_totals_df['cumulative_stack'] = node_totals_df.groupby('stage')['value'].cumsum() - (node_totals_df['value'] / 2) # Center position


                # Create node labels layer
                node_label_layer = alt.Chart(node_totals_df).mark_text(
                    align='left', # Alignment will be adjusted below
                    baseline='middle',
                    dx=5 # Small offset
                ).encode(
                     x=alt.X('stage', axis=None),
                     y=alt.Y('cumulative_stack', axis=None), # Use calculated center position
                     text='node',
                     color=alt.value('black'),
                     # Tooltip for the node total itself
                     tooltip=['node', alt.Tooltip('value', title='Daily Total (kWh)', format=".2f")]
                )

                # Adjust text label alignment based on stage
                node_label_layer = node_label_layer.encode(
                    align=alt.condition(alt.datum.stage == 0, alt.value('right'), alt.value('left')),
                    dx=alt.condition(alt.datum.stage == 0, alt.value(-5), alt.value(5)) # Offset left for source, right for target
                )


                # Combine bands and node labels
                final_sankey_chart = alt.layer(bands, node_label_layer).properties(
                     title=f"Daily Energy Flow from BCQ Generators to MQ Units for {selected_date_str} (Conceptual)"
                ).interactive()


                st.altair_chart(final_sankey_chart, use_container_width=True)

            else:
                 st.info("Insufficient positive flow calculated between components to generate the Sankey chart links.")

        else:
             st.info("Insufficient data (zero totals in source or target components) to generate the Sankey chart.")


    else:
        st.warning(f"No data available for selected date: {selected_date_str}.")
