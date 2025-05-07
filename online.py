import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
# import matplotlib.pyplot as plt # Kept if other matplotlib plots exist
# import matplotlib.ticker as ticker # Kept if other matplotlib plots exist
import altair as alt
import plotly.graph_objects as go # Import Plotly

# --- Configuration for Sankey Chart ---
# These dictionaries map the descriptive source/destination names (assumed to be column names in your DB)
# to shorter aliases. The Sankey will use the keys as node labels and for data fetching.
BCQ_SOURCE_MAP = {
    "FDC Misamis Power Corporation  (FDC)": 'FDC',
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
        st.error(f"Error loading database credentials: {e}. Make sure your .streamlit/secrets.toml file is correctly configured with [database] section and keys: user, password, host, db, port.")
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
        query = """
            SELECT mq."Time", mq."Total_MQ", bcq."Total_BCQ", p."Prices"
            FROM "MQ_Hourly" AS mq
            JOIN "BCQ_Hourly" AS bcq ON mq."Date" = bcq."Date" AND mq."Time" = bcq."Time"
            JOIN "Prices_Hourly" AS p ON mq."Date" = p."Date" AND mq."Time" = p."Time"
            WHERE mq."Date" = %s
            ORDER BY mq."Time"
        """
        df = pd.read_sql(query, engine, params=[(selected_date_str,)])
        if not df.empty and 'Time' in df.columns:
            df['Time'] = pd.to_datetime(selected_date_str + ' ' + df['Time'].astype(str))
        return df
    except Exception as e:
        st.error(f"Error fetching main data: {e}")
        return pd.DataFrame()

def fetch_sankey_df(selected_date_str, engine):
    """
    Fetches and prepares data specifically for the Sankey chart.
    It sums the hourly data for the selected date.
    """
    bcq_source_columns = list(BCQ_SOURCE_MAP.keys())
    mq_dest_columns = list(MQ_DEST_MAP.keys())

    # Columns to select from BCQ_Hourly
    # We need "Total_BCQ" for WESM calculation and individual source columns
    bcq_cols_sql = ['bcq."Total_BCQ"'] + [f'bcq."{col}"' for col in bcq_source_columns]

    # Columns to select from MQ_Hourly
    mq_cols_sql = [f'mq."{col}"' for col in mq_dest_columns]

    # Combine all column selections
    # mq."Time" is included to maintain structure similar to fetch_data, though not directly used in sums
    select_columns_str = ", ".join(['mq."Time"'] + bcq_cols_sql + mq_cols_sql)

    query = f"""
        SELECT {select_columns_str}
        FROM "MQ_Hourly" AS mq
        INNER JOIN "BCQ_Hourly" AS bcq ON mq."Date" = bcq."Date" AND mq."Time" = bcq."Time"
        WHERE mq."Date" = %s
    """
    try:
        df_hourly = pd.read_sql(query, engine, params=[(selected_date_str,)])

        if df_hourly.empty:
            return pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64'), 0.0

        # Calculate daily sums
        daily_total_bcq_sum = df_hourly["Total_BCQ"].sum()
        
        # Sums for BCQ sources
        daily_bcq_source_sums = pd.Series(dtype='float64')
        for col in bcq_source_columns:
            if col in df_hourly.columns:
                daily_bcq_source_sums[col] = df_hourly[col].sum()
            else:
                st.warning(f"BCQ source column '{col}' not found in fetched data for Sankey.")
                daily_bcq_source_sums[col] = 0


        # Sums for MQ destinations
        daily_mq_dest_sums = pd.Series(dtype='float64')
        for col in mq_dest_columns:
            if col in df_hourly.columns:
                daily_mq_dest_sums[col] = df_hourly[col].sum()
            else:
                st.warning(f"MQ destination column '{col}' not found in fetched data for Sankey.")
                daily_mq_dest_sums[col] = 0
        
        return df_hourly, daily_bcq_source_sums, daily_mq_dest_sums, daily_total_bcq_sum

    except Exception as e:
        st.error(f"Error fetching Sankey data: {e}")
        return pd.DataFrame(), pd.Series(dtype='float64'), pd.Series(dtype='float64'), 0.0


# --- STREAMLIT UI ---
st.title("📊 Daily Energy Trading Dashboard")
selected_date = st.date_input("Select date", datetime.today())
selected_date_str = selected_date.strftime('%Y-%m-%d')

# --- FETCH AND DISPLAY MAIN DATA ---
data = fetch_data(selected_date_str)

if not data.empty:
    st.subheader("Hourly Summary")
    # st.dataframe(data) # Optionally display raw data

    st.subheader("📈 Energy Metrics Over Time (Interactive)")
    if 'Time' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Time']):
        melted_data = data.melt("Time", ["Total_MQ", "Total_BCQ", "Prices"], "Metric", "Value")
        chart_energy = alt.Chart(melted_data[melted_data["Metric"].isin(["Total_MQ", "Total_BCQ"])]).mark_line(point=True).encode(
            x=alt.X("Time", axis=alt.Axis(title="Time", format="%H:%M")),
            y=alt.Y("Value", title="Energy (kWh)", axis=alt.Axis(titleColor="tab:blue"), scale=alt.Scale(zero=True)),
            color=alt.Color("Metric", legend=alt.Legend(title="Metric Type"), scale=alt.Scale(scheme='category10')),
            tooltip=[alt.Tooltip("Time", format="%Y-%m-%d %H:%M"), "Metric", "Value"]
        ).properties(title="Energy Metrics")

        chart_price = alt.Chart(melted_data[melted_data["Metric"] == "Prices"]).mark_bar(color="#40B0A6").encode(
            x=alt.X("Time", axis=alt.Axis(title="")),
            y=alt.Y("Value", title="Price (PHP/kWh)", axis=alt.Axis(titleColor="tab:red"), scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip("Time", format="%Y-%m-%d %H:%M"), "Metric", "Value"]
        ).properties(title="Prices")

        combined_chart = alt.layer(chart_price, chart_energy).resolve_scale(y='independent').properties(
            title=f"Energy Metrics and Prices for {selected_date_str}"
        )
        st.altair_chart(combined_chart, use_container_width=True)
    else:
        st.warning("Time column is not in the expected datetime format for plotting or is empty.")
else:
    st.warning(f"No main data available for selected date: {selected_date_str}.")


# --- SANKEY CHART SECTION ---
st.subheader("🌊 Daily Energy Flow (Sankey Chart)")
sankey_engine = get_sqlalchemy_engine() # Get engine for sankey data
sankey_hourly_data, daily_bcq_source_sums, daily_mq_dest_sums, daily_total_bcq_val = fetch_sankey_df(selected_date_str, sankey_engine)

if not sankey_hourly_data.empty:
    # Calculate WESM flow based on user's formula
    sum_of_daily_sums_of_mapped_bcq_generators = daily_bcq_source_sums.sum()
    wesm_flow_value = (daily_total_bcq_val * -1) + (1000 * sum_of_daily_sums_of_mapped_bcq_generators)
    # Sankey values must be non-negative
    wesm_flow_value_sankey = max(0, wesm_flow_value)

    # Prepare labels for Sankey nodes
    source_labels = list(BCQ_SOURCE_MAP.keys())
    dest_labels = list(MQ_DEST_MAP.keys())
    central_node_label = f"Total Daily BCQ ({daily_total_bcq_val:,.0f} kWh)" # Label for the central node

    all_labels_list = source_labels + ["WESM"] + [central_node_label] + dest_labels
    
    # Create a unique list of labels and a mapping to indices
    unique_labels = sorted(list(set(all_labels_list)))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    sankey_sources_indices = []
    sankey_targets_indices = []
    sankey_values = []
    link_colors = []

    # Define some base colors for entity types
    gen_color = 'rgba(50, 100, 200, 0.6)'
    wesm_color = 'rgba(255, 150, 50, 0.6)'
    central_color = 'rgba(100, 180, 100, 0.6)'
    dest_color = 'rgba(200, 80, 80, 0.6)'
    
    node_colors = ['gray'] * len(unique_labels) # Default

    # Assign colors and build links: BCQ Sources -> Central Node
    for src_label in source_labels:
        val = daily_bcq_source_sums.get(src_label, 0)
        if val > 0.001: # Add a small threshold to avoid tiny links
            sankey_sources_indices.append(label_to_index[src_label])
            sankey_targets_indices.append(label_to_index[central_node_label])
            sankey_values.append(val)
            link_colors.append(gen_color)
            if src_label in label_to_index: node_colors[label_to_index[src_label]] = gen_color


    # WESM -> Central Node
    if wesm_flow_value_sankey > 0.001:
        sankey_sources_indices.append(label_to_index["WESM"])
        sankey_targets_indices.append(label_to_index[central_node_label])
        sankey_values.append(wesm_flow_value_sankey)
        link_colors.append(wesm_color)
        if "WESM" in label_to_index: node_colors[label_to_index["WESM"]] = wesm_color

    if central_node_label in label_to_index: node_colors[label_to_index[central_node_label]] = central_color

    # Central Node -> MQ Destinations
    for dest_label in dest_labels:
        val = daily_mq_dest_sums.get(dest_label, 0)
        if val > 0.001:
            sankey_sources_indices.append(label_to_index[central_node_label])
            sankey_targets_indices.append(label_to_index[dest_label])
            sankey_values.append(val)
            link_colors.append(central_color) # Links originating from central node can share its color or a neutral one
            if dest_label in label_to_index: node_colors[label_to_index[dest_label]] = dest_color


    if not sankey_sources_indices:
        st.warning(f"No data to display in Sankey chart for {selected_date_str}. All flow values are too small or zero.")
    else:
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap', # Snap nodes to fixed positions
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="black", width=0.5),
                label=unique_labels,
                color=node_colors # Assign colors to nodes
            ),
            link=dict(
                source=sankey_sources_indices,
                target=sankey_targets_indices,
                value=sankey_values,
                color=link_colors # Assign colors to links
            ))])

        fig.update_layout(
            title_text=f"Energy Flow for {selected_date_str}",
            font_size=10,
            height=600 # Adjust height as needed
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning(f"No data available for Sankey chart for selected date: {selected_date_str}.")
