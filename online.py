import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, date
import altair as alt
import plotly.graph_objects as go # Added for Sankey chart

# Set Streamlit page configuration
st.set_page_config(layout="wide") # Use wide layout for better display

# --- DATABASE CONFIGURATION ---
@st.cache_resource # Cache the database engine creation
def get_sqlalchemy_engine():
    """Establishes and returns a SQLAlchemy database engine using Streamlit secrets."""
    try:
        user = st.secrets["database"]["user"]
        password = st.secrets["database"]["password"]
        host = st.secrets["database"]["host"]
        db = st.secrets["database"]["db"]
        port = int(st.secrets["database"]["port"])
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
        engine = create_engine(url, pool_pre_ping=True)
        return engine
    except KeyError as e:
        st.error(f"Error loading database credentials: {e}. Make sure your .streamlit/secrets.toml file is correctly configured with [database] section and keys: user, password, host, db, port.")
        st.stop()
    except ValueError:
         st.error("Error: Database port in secrets.toml is not a valid integer.")
         st.stop()
    except Exception as e:
        st.error(f"Error creating database engine: {e}")
        st.stop()


# --- LOAD DATA ---
@st.cache_data(ttl=3600)
def fetch_available_dates():
    """Fetches a list of unique dates available in the database."""
    try:
        engine = get_sqlalchemy_engine()
        query = """
            SELECT DISTINCT "Date"
            FROM "MQ_Hourly"
            ORDER BY "Date";
        """
        dates_df = pd.read_sql(query, engine, parse_dates=["Date"])
        available_dates = dates_df["Date"].dt.date.tolist()
        return available_dates
    except Exception as e:
        st.error(f"Error fetching available dates: {e}")
        return []


@st.cache_data(ttl=600)
def fetch_data(selected_date_str: str):
    """Fetches hourly MQ, BCQ, and Prices data for a selected date."""
    try:
        engine = get_sqlalchemy_engine()
        query = """
            SELECT mq."Time", mq."Total_MQ", bcq."Total_BCQ", p."Prices"
            FROM "MQ_Hourly" AS mq
            JOIN "BCQ_Hourly" AS bcq ON mq."Date" = bcq."Date" AND mq."Time" = bcq."Time"
            JOIN "Prices_Hourly" AS p ON mq."Date" = p."Date" AND mq."Time" = p."Time"
            WHERE mq."Date" = %s
            ORDER BY mq."Time";
        """
        # parse_dates cannot parse 'Time' directly if it's just a time string without a date
        # It's better to combine date and time after fetching if 'Time' is just HH:MM:SS
        df = pd.read_sql(query, engine, params=[(selected_date_str,)])

        if not df.empty and 'Time' in df.columns:
            try:
                # Ensure 'Time' is string, extract time part if it's datetime.time, then combine
                if pd.api.types.is_datetime64_any_dtype(df['Time']): # If already full datetime
                    df['Datetime'] = df['Time']
                elif pd.api.types.is_string_dtype(df['Time']):
                    df['Datetime'] = pd.to_datetime(selected_date_str + ' ' + df['Time'], errors='coerce')
                else: # Attempt to convert to string then combine (e.g. if it's datetime.time object)
                    df['Time_str'] = df['Time'].astype(str).str.split().str[-1] # Get HH:MM:SS part
                    df['Datetime'] = pd.to_datetime(selected_date_str + ' ' + df['Time_str'], errors='coerce')

                df.dropna(subset=['Datetime'], inplace=True)
                df['Time'] = df['Datetime'] # Replace original 'Time' with full datetime
                df.drop(columns=['Datetime'], inplace=True, errors='ignore') # Clean up
                df.drop(columns=['Time_str'], inplace=True, errors='ignore') # Clean up
            except Exception as e:
                st.error(f"Error converting 'Time' column to datetime after fetch: {e}. Check format.")
                return pd.DataFrame()
        else:
            if 'Time' not in df.columns: st.warning("Time column missing in fetched data.")


        for col in ["Total_MQ", "Total_BCQ", "Prices"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# --- SANKEY CHART HELPER FUNCTIONS ---
# Define mappings as per comments in the prompt
GENERATOR_LONG_TO_SHORT_MAP = {
    "FDC Misamis Power Corporation (FDC)": 'FDC',
    "GNPower Kauswagan Ltd. Co. (GNPKLCO)": 'GNPK',
    "Power Sector Assets & Liabilities Management Corporation (PSALMGMIN)": 'PSALM',
    "Sarangani Energy Corporation (SEC)": 'SEC',
    "Therma South, Inc. (TSI)": 'TSI',
    "Malita Power Inc. (SMCPC)": 'MPI'
}

DESTINATION_LONG_TO_SHORT_MAP = {
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

@st.cache_data(ttl=600)
def fetch_sankey_generator_contributions(selected_date_str: str, engine, gen_short_to_long_map: dict):
    """
    Fetches daily total contributions for each specified generator.
    NOTE: This is a placeholder. You MUST implement the actual database query.
    The query should sum up the daily energy (e.g., in kWh) for each generator.
    The prompt mentioned '* 1000' for generators. If your DB stores in MWh,
    you'd multiply by 1000 here to get kWh. This placeholder assumes kWh.
    """
    st.warning("Sankey Generator Data: Using DUMMY values. Implement actual DB query in `Workspace_sankey_generator_contributions`.")
    contributions = {}
    # Example: Query a table 'Generator_Daily_Output'
    # query = """
    # SELECT "GeneratorName", SUM("Output_kWh") as "TotalOutput"
    # FROM "Generator_Daily_Output"
    # WHERE "Date" = %s AND "GeneratorName" IN %s
    # GROUP BY "GeneratorName";
    # """
    # df_gen = pd.read_sql(query, engine, params=[(selected_date_str, tuple(gen_short_to_long_map.keys()))])
    # for _, row in df_gen.iterrows():
    # contributions[gen_short_to_long_map[row["GeneratorName"]]] = row["TotalOutput"] * 1000 # If scaling needed

    # Placeholder: Distribute a dummy total, or assign fixed values
    dummy_total_generation = 500000 #kWh
    num_generators = len(gen_short_to_long_map)
    if num_generators > 0:
        for i, short_name in enumerate(gen_short_to_long_map.values()):
            # Assign pseudo-random looking values for better visual
            contributions[short_name] = (dummy_total_generation / num_generators) * (1 + (i % 3 - 1) * 0.2)
    return contributions


@st.cache_data(ttl=600)
def fetch_sankey_destination_consumption(selected_date_str: str, engine, dest_short_to_long_map: dict, total_mq_to_distribute: float):
    """
    Fetches or calculates the consumption for each specified destination.
    NOTE: This is a placeholder. Ideally, you query actual consumption data.
    If not available, it distributes total_mq_to_distribute among destinations.
    """
    st.warning("Sankey Destination Data: Using DUMMY proportional distribution. Implement DB query in `Workspace_sankey_destination_consumption`.")
    consumption = {}
    # Example: Query a table 'Destination_Daily_Consumption'
    # query = """
    # SELECT "DestinationNodeID", SUM("Consumption_kWh") as "TotalConsumption"
    # FROM "Destination_Daily_Consumption"
    # WHERE "Date" = %s AND "DestinationNodeID" IN %s
    # GROUP BY "DestinationNodeID";
    # """
    # df_dest = pd.read_sql(query, engine, params=[(selected_date_str, tuple(dest_short_to_long_map.keys()))])
    # for _, row in df_dest.iterrows():
    #     consumption[dest_short_to_long_map[row["DestinationNodeID"]]] = row["TotalConsumption"]

    # Placeholder: Distribute total_mq_to_distribute proportionally (equally here)
    num_destinations = len(dest_short_to_long_map)
    if num_destinations > 0:
        for i, short_name in enumerate(dest_short_to_long_map.values()):
            consumption[short_name] = (total_mq_to_distribute / num_destinations) * (1 + (i % 3 - 1) * 0.1) # Slight variation
        # Normalize to ensure sum matches total_mq_to_distribute if using variations
        current_sum = sum(consumption.values())
        if current_sum > 0 : # Avoid division by zero
            scaling_factor = total_mq_to_distribute / current_sum
            for short_name in consumption:
                consumption[short_name] *= scaling_factor
    return consumption


# --- STREAMLIT UI ---
st.title("📊 Daily Energy Trading Dashboard")

spacer_left, main_content, spacer_right = st.columns([0.5, 4, 0.5]) # Adjusted spacer for potentially wider content

with main_content:
    available_dates = fetch_available_dates()

    if not available_dates:
        st.error("No available dates found. Check database connection and data.")
        st.stop()

    min_available_date = min(available_dates)
    max_available_date = max(available_dates)
    default_date = max_available_date # Default to the latest available date

    selected_date = st.date_input(
        "Select date",
        value=default_date,
        min_value=min_available_date,
        max_value=max_available_date,
    )

    if selected_date not in available_dates:
        st.warning(f"Data may not be available for the exact date selected: {selected_date}. Displaying data for the closest available date or period if applicable.")

    selected_date_str = selected_date.strftime('%Y-%m-%d')
    data = fetch_data(selected_date_str)

    if not data.empty:
        st.subheader("Daily Summary Metrics")
        col1, col2, col3 = st.columns(3)

        if "Prices" in data.columns and not data["Prices"].empty and pd.api.types.is_numeric_dtype(data["Prices"]):
            max_price = data["Prices"].max(skipna=True)
            avg_price = data["Prices"].mean(skipna=True)
            col1.metric(label="Maximum Price (PHP/kWh)", value=f"{max_price:,.2f}" if pd.notna(max_price) else "N/A")
            col2.metric(label="Average Price (PHP/kWh)", value=f"{avg_price:,.2f}" if pd.notna(avg_price) else "N/A")
        else:
            col1.warning("Prices data not available/numeric.")
            col2.warning("Avg Price data not available/numeric.")

        if "Total_MQ" in data.columns and "Time" in data.columns and \
           not data["Total_MQ"].empty and pd.api.types.is_numeric_dtype(data["Total_MQ"]) and \
           not data["Total_MQ"].isnull().all():
            max_mq_value = data["Total_MQ"].max(skipna=True)
            if pd.notna(max_mq_value):
                max_mq_row_index = data["Total_MQ"].idxmax()
                max_mq_time = data.loc[max_mq_row_index, "Time"]
                max_mq_time_str = max_mq_time.strftime("%H:%M") if pd.api.types.is_datetime64_any_dtype(max_mq_time) else str(max_mq_time)
                col3.metric(label="Maximum Total MQ (kWh)", value=f"{max_mq_value:,.2f}")
                col3.write(f"at {max_mq_time_str}")
            else:
                col3.info("Total_MQ data is all NaN.")
        else:
            col3.warning("Max MQ/Time data not available/numeric or all null.")


        if all(col in data.columns for col in ["Total_BCQ", "Total_MQ"]) and \
           pd.api.types.is_numeric_dtype(data["Total_BCQ"]) and pd.api.types.is_numeric_dtype(data["Total_MQ"]):
            data['WESM'] = data['Total_BCQ'] - data['Total_MQ']
        else:
            st.warning("WESM column not calculated: Total_BCQ or Total_MQ are missing or not numeric.")
            data['WESM'] = pd.NA # Ensure column exists even if calculation fails for safety

        st.subheader("Hourly Summary")
        st.dataframe(data)

        st.subheader("📈 Energy Metrics Over Time (Interactive)")
        if 'Time' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Time']):
            columns_to_melt = ["Total_MQ", "Total_BCQ", "Prices"]
            existing_cols_to_melt = [col for col in columns_to_melt if col in data.columns and not data[col].isnull().all()]

            if existing_cols_to_melt:
                melted_data = data.melt(
                    id_vars=["Time"],
                    value_vars=existing_cols_to_melt,
                    var_name="Metric",
                    value_name="Value"
                )
                melted_data.dropna(subset=['Value'], inplace=True) # Drop rows where 'Value' became NaN

                # Chart for MQ and BCQ
                energy_metrics = [m for m in ["Total_MQ", "Total_BCQ"] if m in existing_cols_to_melt]
                if energy_metrics:
                    chart_energy = alt.Chart(melted_data[melted_data["Metric"].isin(energy_metrics)]).mark_line(point=True).encode(
                        x=alt.X("Time:T", axis=alt.Axis(title="Time", format="%H:%M")),
                        y=alt.Y("Value:Q", title="Energy (kWh)", axis=alt.Axis(titleColor="#1A85FF"), scale=alt.Scale(zero=True)),
                        color=alt.Color("Metric:N", legend=alt.Legend(title="Metric", orient='bottom'),
                                        scale=alt.Scale(domain=['Total_MQ', 'Total_BCQ'], range=['#FFC20A', '#1A85FF'])),
                        tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", "Value:Q"]
                    ).properties(title="Energy Metrics")
                else: chart_energy = alt.Chart(pd.DataFrame()).mark_text() # Empty chart

                # Chart for Prices
                if "Prices" in existing_cols_to_melt:
                    chart_price = alt.Chart(melted_data[melted_data["Metric"] == "Prices"]).mark_bar(color="#40B0A6").encode(
                        x=alt.X("Time:T", axis=alt.Axis(title="Time", format="%H:%M")), # Keep x-axis title for bars if layered
                        y=alt.Y("Value:Q", title="Price (PHP/kWh)", axis=alt.Axis(titleColor="#40B0A6"), scale=alt.Scale(zero=True)),
                        tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", "Value:Q"]
                    ).properties(title="Prices")
                else: chart_price = alt.Chart(pd.DataFrame()).mark_text() # Empty chart

                # Combine charts
                if energy_metrics and "Prices" in existing_cols_to_melt:
                    combined_chart = alt.layer(chart_price, chart_energy).resolve_scale(y='independent').properties(
                        title=f"Energy Metrics and Prices for {selected_date_str}"
                    ).interactive()
                elif energy_metrics: # Only energy chart
                    combined_chart = chart_energy.properties(title=f"Energy Metrics for {selected_date_str}").interactive()
                elif "Prices" in existing_cols_to_melt: # Only price chart
                    combined_chart = chart_price.properties(title=f"Prices for {selected_date_str}").interactive()
                else: # No data to plot
                    combined_chart = alt.Chart(pd.DataFrame()).mark_text(text="No data to plot for selected metrics.").properties(title="No Data")

                st.altair_chart(combined_chart, use_container_width=True)
            else:
                st.warning(f"Required columns for plotting are missing or all null in data for {selected_date_str}.")
        else:
            st.warning("Time column not in expected datetime format or data is empty.")


        # --- SANKEY CHART ---
        st.subheader("⚡ Daily Energy Flow Sankey Chart")
        if 'Total_MQ' in data.columns and 'WESM' in data.columns and \
           pd.api.types.is_numeric_dtype(data['Total_MQ']) and \
           pd.api.types.is_numeric_dtype(data['WESM']) and \
           not data['Total_MQ'].isnull().all():

            engine = get_sqlalchemy_engine()
            sankey_node_labels = []
            node_indices = {} # To map label to index
            sankey_sources_indices = []
            sankey_targets_indices = []
            sankey_values = []
            node_colors = [] # For Plotly node colors

            def add_node(label, color="grey"): # Helper to add unique nodes
                if label not in node_indices:
                    node_indices[label] = len(sankey_node_labels)
                    sankey_node_labels.append(label)
                    node_colors.append(color) # Store color for this node index
                return node_indices[label]

            # 1. Middle Node: Total MQ
            total_mq_sum = data['Total_MQ'].sum()
            if pd.isna(total_mq_sum) or total_mq_sum == 0:
                st.info(f"Total MQ is zero or N/A for {selected_date_str}. Cannot generate Sankey chart.")
                # To prevent further execution in this block if total_mq_sum is not valid
                display_sankey = False
            else:
                display_sankey = True
                middle_node_label = f"Total Daily MQ ({total_mq_sum:,.0f} kWh)"
                middle_node_idx = add_node(middle_node_label, "orange")

                # 2. Source Nodes (Generators & WESM)
                # Generators
                gen_short_to_long_map_inv = {v: k for k, v in GENERATOR_LONG_TO_SHORT_MAP.items()} # for dummy data fetch
                generator_contributions = fetch_sankey_generator_contributions(selected_date_str, engine, GENERATOR_LONG_TO_SHORT_MAP)

                for short_name, value in generator_contributions.items():
                    if value > 0: # Only add if there's a positive contribution
                        # The prompt's '* 1000' for generators: apply if fetched data is e.g. MWh
                        # scaled_value = value * 1000 # Apply scaling if necessary
                        scaled_value = value # Assuming fetched value is already in desired unit (e.g. kWh)
                        gen_node_label = f"{short_name} ({scaled_value:,.0f} kWh)"
                        gen_node_idx = add_node(gen_node_label, "blue")
                        sankey_sources_indices.append(gen_node_idx)
                        sankey_targets_indices.append(middle_node_idx)
                        sankey_values.append(scaled_value)

                # WESM Contribution
                wesm_daily_sum = data['WESM'].sum()
                # Prompt: "Total WESM (from chart) * -1" as a SOURCE.
                # Interpretation 1 (Strict): value = wesm_daily_sum * -1. If positive, use it.
                # wesm_sankey_val_strict = wesm_daily_sum * -1
                # if wesm_sankey_val_strict > 0:
                #     wesm_label = f"WESM (calc as export: {wesm_sankey_val_strict:,.0f} kWh)"
                #     wesm_node_idx = add_node(wesm_label, "red")
                #     sankey_sources_indices.append(wesm_node_idx)
                #     sankey_targets_indices.append(middle_node_idx)
                #     sankey_values.append(wesm_sankey_val_strict)

                # Interpretation 2 (Standard for WESM as Source - Net Import):
                if wesm_daily_sum > 0: # Net import from WESM
                    wesm_label = f"WESM Net Import ({wesm_daily_sum:,.0f} kWh)"
                    wesm_node_idx = add_node(wesm_label, "red")
                    sankey_sources_indices.append(wesm_node_idx)
                    sankey_targets_indices.append(middle_node_idx)
                    sankey_values.append(wesm_daily_sum)
                # If WESM is a net export (wesm_daily_sum < 0), it could be a destination.
                # The prompt only lists it as a source, so we'll only consider net imports here.

                # 3. Destination Nodes
                # The sum of destination values should ideally equal total_mq_sum.
                # fetch_sankey_destination_consumption should handle fetching/distributing this.
                dest_short_to_long_map_inv = {v: k for k, v in DESTINATION_LONG_TO_SHORT_MAP.items()} # for dummy data
                destination_consumptions = fetch_sankey_destination_consumption(
                    selected_date_str, engine, DESTINATION_LONG_TO_SHORT_MAP, total_mq_sum
                )

                for short_name, value in destination_consumptions.items():
                    if value > 0:
                        dest_node_label = f"{short_name} ({value:,.0f} kWh)"
                        dest_node_idx = add_node(dest_node_label, "green")
                        sankey_sources_indices.append(middle_node_idx) # Source is middle node
                        sankey_targets_indices.append(dest_node_idx)
                        sankey_values.append(value)

            # Check for data sufficiency for Sankey
            if not sankey_values or sum(sankey_values) == 0 or not display_sankey:
                st.info(f"Not enough data or zero total MQ to draw Sankey chart for {selected_date_str}.")
            else:
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=25, # Increased padding
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=sankey_node_labels,
                        color=node_colors # Use the dynamic list of colors
                    ),
                    link=dict(
                        source=sankey_sources_indices,
                        target=sankey_targets_indices,
                        value=sankey_values,
                    )
                )])
                fig.update_layout(
                    title_text=f"Energy Flow for {selected_date_str}",
                    font_size=10,
                    height=600 # Adjust height if needed
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Cannot generate Sankey: MQ or WESM data missing/invalid for {selected_date_str}.")

    else: # if data.empty
        st.warning(f"No data available for selected date: {selected_date_str}.")
