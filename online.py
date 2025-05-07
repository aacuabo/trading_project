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
    "14BGN_T1L1_KIDCSCV01_DEL": 'KIDCSCV01',
    "14BGN_T1L1_KIDCSCV02_DEL": 'KIDCSCV02',
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
        # Assuming 'MQ_Hourly' table contains all relevant dates.
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
        # Add individual MQ unit columns, using aliases that match keys in MQ_UNIT_COLUMNS
        # Assuming column names in the database are the keys in MQ_UNIT_COLUMNS
        for col_name in MQ_UNIT_COLUMNS.keys():
             cols_to_select.append(f'mq."{col_name}"')

        # Add individual BCQ generator columns, using aliases that match keys in BCQ_GENERATOR_COLUMNS
        # Need to handle potential spaces/special chars in BCQ generator names from the database
        # Using aliases in the query for easier handling in pandas
        bcq_select_aliases = []
        for col_name, alias in BCQ_GENERATOR_COLUMNS.items():
             # Quote original column name and alias it
             cols_to_select.append(f'bcq."{col_name}" AS "{alias}"')
             bcq_select_aliases.append(alias) # Keep track of the aliases used


        query = f"""
            SELECT {', '.join(cols_to_select)}
            FROM "MQ_Hourly" AS mq
            JOIN "BCQ_Hourly" AS bcq ON mq."Date" = bcq."Date" AND mq."Time" = bcq."Time"
            JOIN "Prices_Hourly" AS p ON mq."Date" = p."Date" AND mq."Time" = p."Time"
            WHERE mq."Date" = %s
            ORDER BY
                mq."Time"
        """
        # Explicitly list columns to parse as dates/times
        # Include Date and Time. Pandas should handle the aliased BCQ columns fine.
        df = pd.read_sql(query, engine, params=[(selected_date_str,)], parse_dates=["Date", "Time"])


        # Ensure Time is treated as a datetime object by combining with Date if needed
        if not df.empty and 'Date' in df.columns and 'Time' in df.columns:
             try:
                 if not pd.api.types.is_datetime64_any_dtype(df['Time']):
                     date_str_col = df['Date'].dt.strftime('%Y-%m-%d') if pd.api.types.is_datetime64_any_dtype(df['Date']) else df['Date'].astype(str)
                     time_str_col = df['Time'].dt.strftime('%H:%M:%S') if pd.api.types.is_datetime64_any_dtype(df['Time']) else df['Time'].astype(str)
                     df['Datetime'] = pd.to_datetime(date_str_col + ' ' + time_str_col, errors='coerce')
                     df.dropna(subset=['Datetime'], inplace=True)
                     df['Time'] = df['Datetime']

                 # Drop the original 'Date' column as 'Time' now includes date+time
                 if 'Date' in df.columns:
                     df = df.drop(columns=['Date'])

             except Exception as e:
                 st.error(f"Error converting 'Time' column to datetime after fetch: {e}.")
                 return pd.DataFrame()


        # Ensure all relevant columns are numeric after fetching/parsing
        # Use the aliases for BCQ columns in this check as they are aliased in the query
        all_value_cols = ["Total_MQ", "Total_BCQ", "Prices"] + list(MQ_UNIT_COLUMNS.keys()) + bcq_select_aliases
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
        # Provide more specific error message if a column might be missing
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


        # --- ALLUVIAL CHART: Flow from MQ Units to BCQ Generators ---
        st.subheader("🌊 Daily Energy Flow (Conceptual Alluvial Chart)")

        st.info("This alluvial chart visualizes a conceptual flow from individual MQ Units to individual BCQ Generators. The total volume of the flow depicted is based on the equation: Total_MQ = (Total_BCQ * 1000) - WESM. The distribution of this flow among individual links is assumed to be proportional to the daily totals of the source MQ unit and the target BCQ generator.")

        # Calculate daily totals for all components
        daily_totals = data.sum(numeric_only=True)

        # Ensure Total_MQ, Total_BCQ, and WESM totals are available
        daily_mq_total_sum = daily_totals.get("Total_MQ", 0)
        daily_bcq_total_sum = daily_totals.get("Total_BCQ", 0)
        daily_wesm_total_sum = daily_totals.get("WESM", 0) # WESM is already BCQ - MQ

        # Calculate the total flow volume based on the user's equation
        # Total_MQ = (Total_BCQ * 1000) - WESM
        # We will visualize a flow quantity equal to the Right Hand Side of the equation
        total_flow_volume = (daily_bcq_total_sum * 1000) - daily_wesm_total_sum

        links_data = []
        mq_available_totals = {}
        bcq_available_totals = {}

        # Get daily totals for available MQ unit columns using their original names from fetch_data
        for col_name, alias in MQ_UNIT_COLUMNS.items():
             if col_name in daily_totals:
                 mq_available_totals[alias] = daily_totals[col_name]

        # Get daily totals for available BCQ generator columns using their aliases from fetch_data query
        for col_name, alias in BCQ_GENERATOR_COLUMNS.items():
             # Use the alias as the key for bcq_available_totals dictionary
             if alias in daily_totals:
                 bcq_available_totals[alias] = daily_totals[alias]


        total_available_mq_units_sum = sum(mq_available_totals.values())
        total_available_bcq_gens_sum = sum(bcq_available_totals.values())

        st.write(f"Calculated Total Flow Volume for Alluvial Chart: {total_flow_volume:,.2f} kWh")


        # Create links data for the alluvial chart only if total flow volume is positive
        # and there are positive totals in both source and target components to distribute from/to.
        if total_flow_volume > 0 and total_available_mq_units_sum > 0 and total_available_bcq_gens_sum > 0:
            # Avoid division by zero in proportionality calculation
            for mq_alias, mq_total in mq_available_totals.items():
                for bcq_alias, bcq_total in bcq_available_totals.items():
                    # Calculate flow value based on proportionality and the defined total_flow_volume
                    # Flow(mq_unit_i -> bcq_gen_j) = (Daily_Total_mq_unit_i / Total of ALL available MQ units) *
                    #                                (Daily_Total_bcq_gen_j / Total of ALL available BCQ generators) *
                    #                                Total Flow Volume derived from equation
                    flow_value = (mq_total / total_available_mq_units_sum) * (bcq_total / total_available_bcq_gens_sum) * total_flow_volume

                    if flow_value > 0: # Only add links with positive flow
                        links_data.append({'source': mq_alias, 'target': bcq_alias, 'value': flow_value})

            links_df = pd.DataFrame(links_data)

            if not links_df.empty:
                # --- Altair Alluvial Chart Implementation ---
                # This requires transforming the links data into a format suitable for stacked areas/trails.
                # We need to create a data structure where each row represents a point on a band's path.
                # A common approach: melt the links data to have 'node', 'stage', 'value', and 'band_group'
                # 'band_group' is a unique identifier for each source-target link (e.g., 'M1-FDC')

                melted_links_source = links_df.copy()
                melted_links_source['stage'] = 0 # Source stage
                melted_links_source['node'] = melted_links_source['source']
                melted_links_source['band_group'] = melted_links_source['source'] + '->' + melted_links_source['target']

                melted_links_target = links_df.copy()
                melted_links_target['stage'] = 1 # Target stage
                melted_links_target['node'] = melted_links_target['target']
                melted_links_target['band_group'] = melted_links_target['source'] + '->' + melted_links_target['target']

                # Combine source and target points for each band
                alluvial_data = pd.concat([melted_links_source, melted_links_target])

                # Sort data for correct stacking and band drawing
                # Sort by stage first, then by source, then by target for consistent banding
                alluvial_data = alluvial_data.sort_values(by=['stage', 'source', 'target']).reset_index(drop=True)


                # Create the bands using mark_area or mark_trail
                # mark_area with stack='center' can create a streamgraph effect,
                # mark_trail is better suited for connecting discrete points.
                # Let's try mark_area with custom stacking to represent the flow volume.

                # Manual stacking requires calculating cumulative sums at each stage.
                # This is complex in Altair directly. Let's try mark_trail which handles paths.

                # Alluvial bands (using mark_trail)
                bands = alt.Chart(alluvial_data).mark_trail().encode(
                    x=alt.X('stage', axis=None), # X-axis represents the stage (Source/Target)
                    # Y-axis represents the stacked value. Need to stack 'value' within each stage,
                    # grouped by 'band_group', and ordered.
                    y=alt.Y('value', stack='zero', axis=None), # Stack values at each stage
                    detail='band_group', # Group paths by the band
                    # Color the bands by the source node for clarity
                    color=alt.Color('source', title='MQ Unit Source'),
                    opacity=alt.Opacity('value', legend=None), # Opacity based on flow value
                    tooltip=[
                        alt.Tooltip('source', title='Source MQ'),
                        alt.Tooltip('target', title='Target BCQ'),
                        alt.Tooltip('value', title='Flow (kWh)', format=".2f")
                    ]
                )

                # Create nodes (rectangles) for the source and target categories
                # Need a separate dataset for nodes with calculated positions
                # For simplicity, let's position nodes based on the summed stacked values

                # Calculate stacked positions for nodes
                alluvial_data['cumulative_value'] = alluvial_data.groupby(['stage', 'node'])['value'].transform('sum')
                # Get the starting position for each node's rectangle - tricky with stack='zero'
                # Let's simplify node representation by using text labels positioned manually or
                # calculating rectangle positions based on cumulative sums.

                # Create node labels
                # Need to get the total value per node at each stage
                node_totals = alluvial_data.groupby(['stage', 'node'])['value'].sum().reset_index()

                # Add stage name for display
                node_totals['stage_name'] = node_totals['stage'].replace({0: 'Source (MQ Units)', 1: 'Target (BCQ Generators)'})


                nodes = alt.Chart(node_totals).mark_text(align='center', baseline='middle', dx=alt.expr("datum.stage == 0 ? -20 : 20")).encode(
                    x=alt.X('stage', axis=None),
                    # Y position should be the center of the stacked bar for this node.
                    # This requires calculating the stack layout manually or using a transform.
                    # Let's approximate positioning for now.
                    # A better way is to use window transforms to calculate stack start/end.
                    # For a simple text label next to the node, let's just use the node name and stage.
                    # We need to calculate the y position for each node based on the cumulative sum of its value and the values below it at that stage.

                    # Calculate cumulative sum per stage and node for positioning
                    node_totals['cumulative_stack'] = node_totals.groupby('stage')['value'].cumsum() - (node_totals['value'] / 2) # Center position

                    y=alt.Y('cumulative_stack', axis=None),
                    text='node',
                    color=alt.value('black'),
                    tooltip=['node', alt.Tooltip('value', title='Daily Total (kWh)', format=".2f")]
                )

                # Title for stages (optional)
                stage_titles = alt.Chart(pd.DataFrame({'stage': [0, 1], 'title': ['MQ Units', 'BCQ Generators']})).mark_text(
                    align='center', baseline='bottom', dy=-10 # Adjust positioning
                ).encode(
                    x='stage',
                    text='title',
                    color=alt.value('black')
                )


                # Combine bands and nodes/labels
                # Need to use layer. For mark_trail + mark_text, layering works.
                # For mark_area based alluvial, layering rectangles on top of areas is complex.
                # Let's stick with the mark_trail bands and text labels.


                chart_alluvial = alt.layer(bands).properties( # Layer bands first
                     title=f"Daily Energy Flow from MQ Units to BCQ Generators for {selected_date_str}"
                ).interactive() # Add interactivity

                # Adding nodes/labels on top requires careful coordinate alignment.
                # Let's display the bands first, and add labels as an overlay if possible.

                # Create a separate layer for node labels using the node_totals DataFrame
                node_label_layer = alt.Chart(node_totals).mark_text(
                    align='left', # Align text to the left for Source, right for Target
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
                final_alluvial_chart = alt.layer(bands, node_label_layer).properties(
                     title=f"Daily Energy Flow from MQ Units to BCQ Generators for {selected_date_str}"
                ).interactive()


                st.altair_chart(final_alluvial_chart, use_container_width=True)

            else:
                 # If links_df is empty because total_flow_volume <= 0 or totals are zero
                 st.info("Insufficient data or zero/negative calculated total flow volume to generate the alluvial chart links with positive flow.")

        else:
             st.info("Insufficient data (zero total flow volume or zero totals in source/target components) to generate the alluvial chart.")


    else:
        st.warning(f"No data available for selected date: {selected_date_str}.")
