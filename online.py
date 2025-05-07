import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker # Not strictly needed if only Altair/Plotly used for this part
import altair as alt
import plotly.graph_objects as go # Import Plotly for Sankey

# --- CONSTANTS FOR SANKEY CHART ---
# These map database column names to display names or internal keys for the Sankey chart.
# IMPORTANT: The KEYS in these dictionaries MUST EXACTLY MATCH your database column names.
BCQ_SOURCE_MAP = {
    "FDC Misamis Power Corporation (FDC)": 'FDC',
    "GNPower Kauswagan Ltd. Co. (GNPKLCO)": 'GNPK',
    "Power Sector Assets & Liabilities Management Corporation (PSALMGMIN)": 'PSALM',
    "Sarangani Energy Corporation (SEC)": 'SEC',
    "Therma South, Inc. (TSI)": 'TSI',
    "Malita Power Inc. (SMCPC)": 'MPI',
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

# --- DATABASE CONFIGURATION ---
def get_sqlalchemy_engine():
    try:
        user = st.secrets["database"]["user"]
        password = st.secrets["database"]["password"]
        host = st.secrets["database"]["host"]
        db = st.secrets["database"]["db"]
        port = int(st.secrets["database"]["port"])
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
        return create_engine(url)
    except KeyError as e:
        st.error(f"Error loading database credentials: {e}. Make sure your .streamlit/secrets.toml file is correctly configured.")
        st.stop()
    except ValueError:
        st.error("Error: Database port in secrets.toml is not a valid integer.")
        st.stop()
    except Exception as e:
        st.error(f"Error creating database engine: {e}")
        st.stop()

# --- LOAD DATA ---
def fetch_data(selected_date_str):
    try:
        engine = get_sqlalchemy_engine()

        # Prepare column selections for SQL query, ensuring proper quoting
        # These are the actual column names from your database
        bcq_db_columns = [f'bcq."{col}"' for col in BCQ_SOURCE_MAP.keys()]
        mq_db_columns = [f'mq."{col}"' for col in MQ_DEST_MAP.keys()]

        bcq_columns_sql = ", ".join(bcq_db_columns)
        mq_columns_sql = ", ".join(mq_db_columns)

        # Construct the query
        # It's crucial that all listed column names exist in their respective tables.
        query = f"""
            SELECT
                mq."Time",
                mq."Total_MQ",
                {mq_columns_sql + "," if mq_columns_sql else ""}
                bcq."Total_BCQ",
                {bcq_columns_sql + "," if bcq_columns_sql else ""}
                p."Prices"
            FROM "MQ_Hourly" AS mq
            JOIN "BCQ_Hourly" AS bcq ON mq."Date" = bcq."Date" AND mq."Time" = bcq."Time"
            JOIN "Prices_Hourly" AS p ON mq."Date" = p."Date" AND mq."Time" = p."Time"
            WHERE mq."Date" = %s
            ORDER BY
                mq."Time"
        """
        # Remove trailing comma if any of the column groups were empty (though unlikely here)
        query = query.replace(",                p", " p")


        df = pd.read_sql(query, engine, params=[(selected_date_str,)])

        if not df.empty and 'Time' in df.columns:
            df['Time'] = pd.to_datetime(selected_date_str + ' ' + df['Time'].astype(str))
        
        # Ensure all expected numeric columns are present and convert to numeric, coercing errors
        # This helps prevent issues if a column is unexpectedly missing or non-numeric
        expected_numeric_cols = ['Total_MQ', 'Total_BCQ', 'Prices'] + \
                                list(BCQ_SOURCE_MAP.keys()) + \
                                list(MQ_DEST_MAP.keys())
        for col in expected_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # else: # Optional: Warn if an expected column for Sankey is missing
                # st.warning(f"Expected column '{col}' not found in fetched data. Sankey chart might be incomplete.")
        
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        # st.error(f"Last attempted SQL query: {query}") # For debugging, if needed
        return pd.DataFrame()

# --- PREPARE SANKEY DATA ---
def prepare_sankey_data(daily_data_sum, selected_date_str_title):
    """
    Prepares data for the Sankey chart based on daily sums.
    daily_data_sum: A Pandas Series containing summed values for the day.
    selected_date_str_title: The date string for the chart title.
    """
    labels = []
    source_indices = []
    target_indices = []
    values = []
    node_colors = [] # For node colors
    node_map = {}

    def add_node(name, color='rgba(100,100,200,0.8)'): # Default node color
        if name not in node_map:
            node_map[name] = len(labels)
            labels.append(name)
            node_colors.append(color)
        return node_map[name]

    def add_link(src_name, tgt_name, value, src_node_color=None, tgt_node_color=None):
        # Ensure value is a number and positive, otherwise skip
        if not isinstance(value, (int, float)) or pd.isna(value) or value <= 0:
            return
        
        src_idx = add_node(src_name, color=src_node_color if src_node_color else 'rgba(0, 128, 255, 0.7)') # Default source color
        tgt_idx = add_node(tgt_name, color=tgt_node_color if tgt_node_color else 'rgba(255, 128, 0, 0.7)') # Default target color
        
        source_indices.append(src_idx)
        target_indices.append(tgt_idx)
        values.append(value)

    # --- Define Nodes and Links ---
    middle_node_label = "Total BCQ (Daily Sum)"
    middle_node_color = 'rgba(128, 128, 128, 0.7)' # Grey for middle node
    add_node(middle_node_label, color=middle_node_color)


    # 1a. WESM to Total BCQ
    # WESM value = (Total_BCQ_column_sum * -1) + (1000 * sum of values from BCQ_SOURCE_MAP columns)
    sum_of_mapped_bcq_values = 0
    for col_db_name in BCQ_SOURCE_MAP.keys():
        sum_of_mapped_bcq_values += daily_data_sum.get(col_db_name, 0)
    
    total_bcq_val_for_wesm_calc = daily_data_sum.get("Total_BCQ", 0)
    wesm_source_value = (total_bcq_val_for_wesm_calc * -1) + (1000 * sum_of_mapped_bcq_values)
    
    add_link("WESM", middle_node_label, wesm_source_value, src_node_color='rgba(66, 135, 245, 0.7)') # Blue-ish for WESM

    # 1b. Individual BCQ Sources (from BCQ_SOURCE_MAP) to Total BCQ
    # Value for these links is 1000 * daily_data_sum[col_name]
    generator_node_color = 'rgba(76, 175, 80, 0.7)' # Green-ish for generators
    for col_db_name, display_name in BCQ_SOURCE_MAP.items():
        value = daily_data_sum.get(col_db_name, 0) * 1000
        add_link(display_name, middle_node_label, value, src_node_color=generator_node_color, tgt_node_color=middle_node_color)

    # 2. "Total BCQ" (Middle Node) to MQ Destinations
    # Value for these links is daily_data_sum[mq_col_db_name]
    destination_node_color = 'rgba(255, 87, 34, 0.7)' # Orange-ish for destinations
    for mq_col_db_name, mq_display_name in MQ_DEST_MAP.items():
        value = daily_data_sum.get(mq_col_db_name, 0)
        add_link(middle_node_label, mq_display_name, value, src_node_color=middle_node_color, tgt_node_color=destination_node_color)
            
    if not values: # No valid links created
        return None

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=25, # Increased padding
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=node_colors # Apply custom node colors
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=values,
            # color='rgba(200,200,200,0.4)' # Optional: uniform link color
        ))])

    fig.update_layout(
        title_text=f"Daily Energy Flow: {selected_date_str_title}",
        font_size=10,
        height=600 # Adjust height as needed
    )
    return fig

# --- STREAMLIT UI ---
st.set_page_config(layout="wide") # Use wide layout for better dashboard view
st.title("📊 Daily Energy Trading Dashboard")

selected_date = st.date_input("Select date", datetime.today())
selected_date_str = selected_date.strftime('%Y-%m-%d')

# --- FETCH AND DISPLAY DATA ---
data = fetch_data(selected_date_str)

if not data.empty:
    st.subheader(f"Hourly Summary for {selected_date_str}")
    # For debugging, you might want to see all columns:
    # st.dataframe(data)

    # --- ALTAIR PLOT (INTERACTIVE TIME SERIES) ---
    st.subheader("📈 Energy Metrics Over Time (Interactive)")
    if 'Time' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Time']):
        # Ensure 'Prices' is numeric for Altair plot, if not already handled by fetch_data
        data_for_altair = data.copy()
        data_for_altair['Prices'] = pd.to_numeric(data_for_altair['Prices'], errors='coerce')
        
        melted_data = data_for_altair.melt(
            "Time",
            ["Total_MQ", "Total_BCQ", "Prices"],
            "Metric",
            "Value"
        )
        
        # Filter out NaN values that can cause issues in Altair plots, especially with connected lines
        melted_data = melted_data.dropna(subset=['Value'])

        chart_energy = alt.Chart(melted_data[melted_data["Metric"].isin(["Total_MQ", "Total_BCQ"])]).mark_line(point=True).encode(
            x=alt.X("Time:T", axis=alt.Axis(title="Time", format="%H:%M")),
            y=alt.Y("Value:Q", title="Energy (kWh)", axis=alt.Axis(titleColor="steelblue"), scale=alt.Scale(zero=True)),
            color=alt.Color("Metric:N", legend=alt.Legend(title="Energy Metric"), scale=alt.Scale(scheme='category10')),
            tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", "Value:Q"]
        ).properties(title="Energy Consumption/Nomination")

        chart_price = alt.Chart(melted_data[melted_data["Metric"] == "Prices"]).mark_bar(color="#40B0A6", opacity=0.7).encode(
            x=alt.X("Time:T", axis=alt.Axis(title="")),
            y=alt.Y("Value:Q", title="Price (PHP/kWh)", axis=alt.Axis(titleColor="#40B0A6"), scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", "Value:Q"]
        ).properties(title="Market Prices")
        
        combined_chart = alt.layer(chart_price, chart_energy).resolve_scale(
            y='independent'
        ).properties(
            title=f"Energy Metrics and Prices for {selected_date_str}"
        )
        st.altair_chart(combined_chart, use_container_width=True)
    else:
        st.warning("Time column is not in the expected datetime format for plotting or is empty.")

    # --- SANKEY CHART ---
    st.subheader(f"🌊 Sankey Diagram: Daily Energy Flow for {selected_date_str}")
    
    # Prepare data for Sankey: sum all relevant numeric columns for the day.
    # 'Time' and 'Prices' are excluded as they are handled differently or not part of the flow logic.
    sankey_relevant_cols = ['Total_MQ', 'Total_BCQ'] + \
                           list(BCQ_SOURCE_MAP.keys()) + \
                           list(MQ_DEST_MAP.keys())
    
    # Select only relevant columns and ensure they are numeric before summing
    sankey_input_df = data[sankey_relevant_cols].copy()
    for col in sankey_relevant_cols:
        if col in sankey_input_df.columns: # Column might be missing if fetch_data had issues
             sankey_input_df[col] = pd.to_numeric(sankey_input_df[col], errors='coerce').fillna(0) # Fill NaNs with 0 before sum
        else:
            sankey_input_df[col] = 0 # Add column with 0 if it was missing, to prevent KeyError in .sum()
            
    daily_sums = sankey_input_df.sum()

    if not daily_sums.empty and daily_sums.sum() > 0 : # Check if there's any data to plot
        sankey_fig = prepare_sankey_data(daily_sums, selected_date_str)
        if sankey_fig:
            st.plotly_chart(sankey_fig, use_container_width=True)
        else:
            st.info(f"No data or only zero/negative values available to display in the Sankey diagram for {selected_date_str}.")
    else:
        st.warning(f"Could not compute valid daily sums for the Sankey diagram for {selected_date_str}. Please check data availability and integrity.")

else:
    st.warning(f"No data available for selected date: {selected_date_str}.")
