import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, date, time 
import altair as alt
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple, Any, Optional 

# --- Configuration Constants ---
# Database Column Names
COL_DATE = "Date"
COL_TIME = "Time"
COL_TOTAL_MQ = "Total_MQ"
COL_TOTAL_BCQ = "Total_BCQ"
COL_PRICES = "Prices"
COL_WESM = "WESM" # Derived column
COL_HOUR = "Hour" # Derived column for processing
COL_HOUR_STR = "Hour_Str" # Derived string representation of hour
COL_ABS_WESM = "Abs_WESM" # For WESM chart y-axis

# Generator and Destination Mapping Keys (Long Names from DB)
FDC_MISAMIS_POWER = "FDC_Misamis_Power_Corporation__FDC"
GNPOWER_KAUSWAGAN = "GNPower_Kauswagan_Ltd._Co._GNPKLCO"
PSALM_POWER = "Power_Sector_Assets_and_Liabilities_Management_Corporation_PSAL"
SARANGANI_ENERGY = "Sarangani_Energy_Corporation_SEC"
THERMA_SOUTH = "Therma_South,_Inc._TSI"
MALITA_POWER = "Malita_Power_Inc._SMCPC"

# Destination Node Names (Example, adjust as per your DB)
DEST_M1_M6_M8 = "14BGN_T1L1_KIDCOTE01_NET"
DEST_M2 = "14BGN_T1L1_KIDCOTE02_NET"
DEST_M3 = "14BGN_T1L1_KIDCOTE03_NET"
DEST_M4 = "14BGN_T1L1_KIDCOTE04_NET"
DEST_M5 = "14BGN_T2L1_KIDCOTE05_NET"
DEST_M7 = "14BGN_T1L1_KIDCOTE08_NET"
DEST_M9 = "14BGN_T1L1_KIDCOTE10_NET"
DEST_KIDCSCV01 = "14BGN_T1L1_KIDCSCV01_DEL"
DEST_KIDCSCV02 = "14BGN_T1L1_KIDCSCV02_DEL"


# Set Streamlit page configuration
st.set_page_config(layout="centered", page_title="Energy Trading Dashboard")

# --- DATABASE CONFIGURATION ---
@st.cache_resource
def get_sqlalchemy_engine():
    """Establishes and returns a SQLAlchemy database engine using Streamlit secrets."""
    try:
        # Access database credentials from Streamlit secrets
        user = st.secrets["database"]["user"]
        password = st.secrets["database"]["password"]
        host = st.secrets["database"]["host"]
        db_name = st.secrets["database"]["db"] 
        port = int(st.secrets["database"]["port"])
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 15})
        # Test connection
        with engine.connect() as conn: 
            pass 
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
@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_available_dates() -> List[date]:
    """Fetches a list of unique dates available in the database."""
    try:
        engine = get_sqlalchemy_engine()
        query = f"""
            SELECT DISTINCT "{COL_DATE}"
            FROM "MQ_Hourly"
            ORDER BY "{COL_DATE}";
        """
        dates_df = pd.read_sql(query, engine, parse_dates=[COL_DATE])
        available_dates = dates_df[COL_DATE].dt.date.tolist()
        return available_dates
    except Exception as e:
        st.error(f"Error fetching available dates: {e}")
        return []

@st.cache_data(ttl=600) # Cache data for 10 minutes
def fetch_data_for_range(start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """Fetches hourly MQ, BCQ, and Prices data for a selected date range."""
    try:
        engine = get_sqlalchemy_engine()
        query = f"""
            SELECT mq."{COL_DATE}", mq."{COL_TIME}", mq."{COL_TOTAL_MQ}", bcq."{COL_TOTAL_BCQ}", p."{COL_PRICES}"
            FROM "MQ_Hourly" AS mq
            LEFT JOIN "BCQ_Hourly" AS bcq ON mq."{COL_DATE}" = bcq."{COL_DATE}" AND mq."{COL_TIME}" = bcq."{COL_TIME}"
            LEFT JOIN "Prices_Hourly" AS p ON mq."{COL_DATE}" = p."{COL_DATE}" AND mq."{COL_TIME}" = p."{COL_TIME}"
            WHERE mq."{COL_DATE}" BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY mq."{COL_DATE}", mq."{COL_TIME}";
        """
        df = pd.read_sql(query, engine, params={"start_date": start_date_str, "end_date": end_date_str})

        if df.empty:
            return pd.DataFrame()

        df[COL_DATE] = pd.to_datetime(df[COL_DATE])

        # Convert COL_TIME to time objects
        if COL_TIME in df.columns:
            try:
                if not pd.api.types.is_string_dtype(df[COL_TIME]):
                    df[COL_TIME] = df[COL_TIME].astype(str)
                df[COL_HOUR] = pd.to_datetime(df[COL_TIME].str.strip(), format='%H:%M:%S', errors='coerce').dt.time
            except Exception as e:
                st.warning(f"Warning converting '{COL_TIME}' column to time objects: {e}")
                df[COL_HOUR] = pd.NaT

        # Convert relevant columns to numeric, coercing errors
        for col in [COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_PRICES]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate WESM (Wholesale Electricity Spot Market)
        if all(c in df.columns for c in [COL_TOTAL_BCQ, COL_TOTAL_MQ]) and \
           pd.api.types.is_numeric_dtype(df[COL_TOTAL_BCQ]) and \
           pd.api.types.is_numeric_dtype(df[COL_TOTAL_MQ]):
            df[COL_WESM] = df[COL_TOTAL_BCQ] - df[COL_TOTAL_MQ]
        else:
            df[COL_WESM] = pd.NA
        return df
    except Exception as e:
        st.error(f"Error fetching data for range: {e}")
        return pd.DataFrame()

# --- SANKEY CHART HELPER FUNCTIONS ---

# Mapping for generator long names to short names for display
GENERATOR_LONG_TO_SHORT_MAP = {
    FDC_MISAMIS_POWER: 'FDC',
    GNPOWER_KAUSWAGAN: 'GNPK',
    PSALM_POWER: 'PSALM',
    SARANGANI_ENERGY: 'SEC',
    THERMA_SOUTH: 'TSI',
    MALITA_POWER: 'MPI'
}
# Columns that need to be scaled (e.g., from MWh to kWh if original data is MWh)
GENERATOR_COLUMNS_TO_SCALE = list(GENERATOR_LONG_TO_SHORT_MAP.keys()) 

# Mapping for destination long names to short names for display
DESTINATION_LONG_TO_SHORT_MAP = {
    DEST_M1_M6_M8: 'M1/M6/M8',
    DEST_M2: 'M2',
    DEST_M3: 'M3',
    DEST_M4: 'M4',
    DEST_M5: 'M5',
    DEST_M7: 'M7',
    DEST_M9: 'M9',
    DEST_KIDCSCV01: 'KIDCSCV01_DEL',
    DEST_KIDCSCV02: 'KIDCSCV02_DEL'
}

# For TOTAL Sankey
@st.cache_data(ttl=600)
def fetch_sankey_generator_contributions_total(start_date_str: str, end_date_str: str, selected_day_indices: List[int]) -> Dict[str, float]:
    """
    Fetches total generator contributions for the Sankey chart within a date range and selected days.
    """
    contributions = {short_name: 0.0 for short_name in GENERATOR_LONG_TO_SHORT_MAP.values()}
    if not GENERATOR_LONG_TO_SHORT_MAP: return contributions
    try:
        engine = get_sqlalchemy_engine()
        select_cols_for_fetch = '", "'.join(GENERATOR_LONG_TO_SHORT_MAP.keys())
        query = f""" SELECT "{COL_DATE}", "{select_cols_for_fetch}" FROM "BCQ_Hourly" WHERE "{COL_DATE}" BETWEEN %(start_date)s AND %(end_date)s; """
        params = {"start_date": start_date_str, "end_date": end_date_str}
        period_data_df = pd.read_sql(query, engine, params=params)
        if period_data_df.empty: return contributions
        period_data_df[COL_DATE] = pd.to_datetime(period_data_df[COL_DATE])
        filtered_df = period_data_df[period_data_df[COL_DATE].dt.dayofweek.isin(selected_day_indices)].copy()
        if filtered_df.empty: return contributions
        for long_name, short_name in GENERATOR_LONG_TO_SHORT_MAP.items():
            if long_name in filtered_df.columns:
                total_value = pd.to_numeric(filtered_df[long_name], errors='coerce').sum() 
                if pd.notna(total_value):
                    if long_name in GENERATOR_COLUMNS_TO_SCALE: # Apply scaling if needed
                        total_value *= 1000 
                    contributions[short_name] = float(total_value) if total_value > 0 else 0.0
        return contributions
    except Exception as e:
        st.error(f"Error fetching total Sankey generator contributions: {e}")
        return contributions

@st.cache_data(ttl=600)
def fetch_sankey_destination_consumption_total(start_date_str: str, end_date_str: str, selected_day_indices: List[int]) -> Dict[str, float]:
    """
    Fetches total destination consumption for the Sankey chart within a date range and selected days.
    """
    consumption = {short_name: 0.0 for short_name in DESTINATION_LONG_TO_SHORT_MAP.values()}
    if not DESTINATION_LONG_TO_SHORT_MAP: return consumption
    try:
        engine = get_sqlalchemy_engine()
        select_cols_for_fetch = '", "'.join(DESTINATION_LONG_TO_SHORT_MAP.keys())
        query = f""" SELECT "{COL_DATE}", "{select_cols_for_fetch}" FROM "MQ_Hourly" WHERE "{COL_DATE}" BETWEEN %(start_date)s AND %(end_date)s; """
        params = {"start_date": start_date_str, "end_date": end_date_str}
        period_data_df = pd.read_sql(query, engine, params=params)
        if period_data_df.empty: return consumption
        period_data_df[COL_DATE] = pd.to_datetime(period_data_df[COL_DATE])
        filtered_df = period_data_df[period_data_df[COL_DATE].dt.dayofweek.isin(selected_day_indices)].copy()
        if filtered_df.empty: return consumption
        for long_name, short_name in DESTINATION_LONG_TO_SHORT_MAP.items():
            if long_name in filtered_df.columns:
                total_value = pd.to_numeric(filtered_df[long_name], errors='coerce').sum() 
                if pd.notna(total_value):
                    consumption[short_name] = float(total_value) if total_value > 0 else 0.0
        return consumption
    except Exception as e:
        st.error(f"Error fetching total Sankey destination consumption: {e}")
        return consumption

# For AVERAGED HOURLY Sankey
@st.cache_data(ttl=600)
def fetch_sankey_generator_contributions_averaged(start_date_str: str, end_date_str: str, selected_day_indices: List[int], interval_time_db_format: str) -> Dict[str, float]:
    """
    Fetches averaged generator contributions for the Sankey chart for a specific hour
    within a date range and selected days.
    """
    contributions = {short_name: 0.0 for short_name in GENERATOR_LONG_TO_SHORT_MAP.values()}
    if not GENERATOR_LONG_TO_SHORT_MAP: return contributions
    try:
        engine = get_sqlalchemy_engine()
        query_columns_list = [f'"{col_name}"' for col_name in GENERATOR_LONG_TO_SHORT_MAP.keys()]
        query_columns_str = ', '.join(query_columns_list)
        query = f""" SELECT "{COL_DATE}", {query_columns_str} FROM "BCQ_Hourly" WHERE "{COL_DATE}" BETWEEN %(start_date)s AND %(end_date)s AND "{COL_TIME}" = %(interval_time)s; """
        params = {"start_date": start_date_str, "end_date": end_date_str, "interval_time": interval_time_db_format}
        range_interval_data_df = pd.read_sql(query, engine, params=params)
        if range_interval_data_df.empty: return contributions
        range_interval_data_df[COL_DATE] = pd.to_datetime(range_interval_data_df[COL_DATE])
        filtered_df = range_interval_data_df[range_interval_data_df[COL_DATE].dt.dayofweek.isin(selected_day_indices)].copy()
        if filtered_df.empty: return contributions
        for long_name, short_name in GENERATOR_LONG_TO_SHORT_MAP.items():
            if long_name in filtered_df.columns:
                avg_value = pd.to_numeric(filtered_df[long_name], errors='coerce').mean() 
                if pd.notna(avg_value):
                    if long_name in GENERATOR_COLUMNS_TO_SCALE: # Apply scaling if needed
                        avg_value *= 1000 
                    contributions[short_name] = float(avg_value) if avg_value > 0 else 0.0
        return contributions
    except Exception as e:
        st.error(f"Error fetching averaged Sankey generator contributions: {e}")
        return contributions

@st.cache_data(ttl=600)
def fetch_sankey_destination_consumption_averaged(start_date_str: str, end_date_str: str, selected_day_indices: List[int], interval_time_db_format: str) -> Dict[str, float]:
    """
    Fetches averaged destination consumption for the Sankey chart for a specific hour
    within a date range and selected days.
    """
    consumption = {short_name: 0.0 for short_name in DESTINATION_LONG_TO_SHORT_MAP.values()}
    if not DESTINATION_LONG_TO_SHORT_MAP: return consumption
    try:
        engine = get_sqlalchemy_engine()
        query_columns_list = [f'"{col_name}"' for col_name in DESTINATION_LONG_TO_SHORT_MAP.keys()]
        query_columns_str = ', '.join(query_columns_list)
        query = f""" SELECT "{COL_DATE}", {query_columns_str} FROM "MQ_Hourly" WHERE "{COL_DATE}" BETWEEN %(start_date)s AND %(end_date)s AND "{COL_TIME}" = %(interval_time)s; """
        params = {"start_date": start_date_str, "end_date": end_date_str, "interval_time": interval_time_db_format}
        range_interval_data_df = pd.read_sql(query, engine, params=params)
        if range_interval_data_df.empty: return consumption
        range_interval_data_df[COL_DATE] = pd.to_datetime(range_interval_data_df[COL_DATE])
        filtered_df = range_interval_data_df[range_interval_data_df[COL_DATE].dt.dayofweek.isin(selected_day_indices)].copy()
        if filtered_df.empty: return consumption
        for long_name, short_name in DESTINATION_LONG_TO_SHORT_MAP.items():
            if long_name in filtered_df.columns:
                avg_value = pd.to_numeric(filtered_df[long_name], errors='coerce').mean() 
                if pd.notna(avg_value):
                    consumption[short_name] = float(avg_value) if avg_value > 0 else 0.0
        return consumption
    except Exception as e:
        st.error(f"Error fetching averaged Sankey destination consumption: {e}")
        return consumption

def create_sankey_nodes_links(
    contributions_data: Dict[str, float],
    consumption_data: Dict[str, float],
    flow_metric_val: float, # Total MQ for total Sankey, Avg MQ for hourly Sankey
    junction_label_prefix: str = "Supply"
) -> Tuple[Optional[List[str]], Optional[Dict[str, int]], Optional[List[int]], Optional[List[int]], Optional[List[float]], Optional[List[str]]]:
    """
    Helper function to create nodes and links for Sankey chart. Generic for total or average.
    Calculates WESM import/export and formats labels.
    """
    sum_contributions = float(sum(v for v in contributions_data.values() if pd.notna(v)))
    
    # WESM value for Sankey logic (positive means export, negative means import)
    wesm_value_for_sankey = float(sum_contributions - flow_metric_val)

    if sum_contributions < 0.01 and flow_metric_val < 0.01:
        return None, None, None, None, None, None # Not enough data

    sankey_node_labels: List[str] = []
    node_indices: Dict[str, int] = {}
    sankey_sources_indices: List[int] = []
    sankey_targets_indices: List[int] = []
    sankey_values: List[float] = []
    node_colors_list: List[str] = [] 
    
    # Define a color palette for different node types
    COLOR_PALETTE = {
        "junction": "#E69F00", # Orange
        "generator": "#0072B2", # Blue
        "wesm_import": "#009E73", # Green (for flow into the system)
        "load": "#D55E00", # Red-orange
        "wesm_export": "#CC79A7" # Pink (for flow out of the system)
    }
    
    def add_node(label: str, color: str) -> int:
        """Adds a node to the Sankey diagram if it doesn't already exist."""
        if label not in node_indices:
            node_indices[label] = len(sankey_node_labels)
            sankey_node_labels.append(label)
            node_colors_list.append(color)
        return node_indices[label]

    # Calculate total flow through the central junction node
    total_flow_through_junction = float(sum_contributions)
    if wesm_value_for_sankey < 0: # If there's a net import from WESM, add it to the total supply
        total_flow_through_junction += float(abs(wesm_value_for_sankey))
    
    # Create the central junction node (e.g., "Total Supply" or "Avg Supply at 10:00")
    junction_node_label = f"{junction_label_prefix} ({total_flow_through_junction:,.0f} kWh)"
    junction_node_idx = add_node(junction_node_label, COLOR_PALETTE["junction"])

    # Add generator nodes and links to the junction
    for short_name, value in contributions_data.items():
        value = float(value)
        if value > 0.01: # Only add if value is significant
            percentage = (value / sum_contributions * 100) if sum_contributions > 0 else 0
            gen_node_label = f"{short_name} ({value:,.0f} kWh, {percentage:.1f}%)"
            gen_node_idx = add_node(gen_node_label, COLOR_PALETTE["generator"])
            sankey_sources_indices.append(gen_node_idx)
            sankey_targets_indices.append(junction_node_idx)
            sankey_values.append(value)

    # Add WESM Import node if applicable
    if wesm_value_for_sankey < 0: 
        import_value = float(abs(wesm_value_for_sankey))
        if import_value > 0.01:
            percentage = (import_value / total_flow_through_junction * 100) if total_flow_through_junction > 0 else 0
            wesm_import_label = f"WESM Import ({import_value:,.0f} kWh, {percentage:.1f}%)"
            wesm_import_node_idx = add_node(wesm_import_label, COLOR_PALETTE["wesm_import"])
            sankey_sources_indices.append(wesm_import_node_idx)
            sankey_targets_indices.append(junction_node_idx)
            sankey_values.append(import_value)

    # Add destination (load) nodes and links from the junction
    sum_destination_consumptions = float(sum(v for v in consumption_data.values() if pd.notna(v) and v > 0.01))
    if sum_destination_consumptions > 0.01: 
        for short_name, value in consumption_data.items():
            value = float(value)
            if value > 0.01:
                percentage = (value / flow_metric_val * 100) if flow_metric_val > 0 else 0
                dest_node_label = f"{short_name} ({value:,.0f} kWh, {percentage:.1f}%)"
                dest_node_idx = add_node(dest_node_label, COLOR_PALETTE["load"])
                sankey_sources_indices.append(junction_node_idx)
                sankey_targets_indices.append(dest_node_idx)
                sankey_values.append(value)
    
    # Add WESM Export node if applicable
    if wesm_value_for_sankey > 0: 
        export_value = float(wesm_value_for_sankey)
        if export_value > 0.01:
            percentage = (export_value / total_flow_through_junction * 100) if total_flow_through_junction > 0 else 0
            wesm_export_label = f"WESM Export ({export_value:,.0f} kWh, {percentage:.1f}%)"
            wesm_export_node_idx = add_node(wesm_export_label, COLOR_PALETTE["wesm_export"])
            sankey_sources_indices.append(junction_node_idx)
            sankey_targets_indices.append(wesm_export_node_idx)
            sankey_values.append(export_value)
            
    # Return None if no significant flows are found
    if not sankey_values or sum(sankey_values) < 0.1: 
        return None, None, None, None, None, None
    
    return sankey_node_labels, node_indices, sankey_sources_indices, sankey_targets_indices, [float(v) for v in sankey_values], node_colors_list


def create_sankey_figure(title: str, node_labels, sources, targets, values, node_colors) -> go.Figure:
    """Helper to create the Plotly Sankey figure object."""
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap", # Arranges nodes to minimize link crossings
        node=dict(
            pad=20, # Padding between nodes
            thickness=15, # Thickness of nodes
            line=dict(color="#A9A9A9", width=0.5), # Node border
            label=node_labels, 
            color=node_colors
        ),
        link=dict(
            source=sources, 
            target=targets, 
            value=values, 
            hovertemplate='%{source.label} → %{target.label}: %{value:,.0f} kWh<extra></extra>' # Hover text format
        )
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        font=dict(family="Arial, sans-serif", size=12), 
        plot_bgcolor='rgba(0,0,0,0)', # Transparent background
        paper_bgcolor='rgba(0,0,0,0)', # Transparent paper background
        height=600, # Fixed height for consistency
        margin=dict(l=20, r=20, t=50, b=20) # Margins
    )
    return fig

# CSS to center align KPI metrics
kpi_alignment_css = """
<style>
    [data-testid="stMetricValue"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    [data-testid="stMetric"] {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        margin: auto;
    }
</style>
"""
st.markdown(kpi_alignment_css, unsafe_allow_html=True)


def app_content():
    """Main function to render the Streamlit application content."""
    st.title("📊 Daily Energy Trading Dashboard") 
    
    # CSS to adjust font size of KPI labels and values
    kpi_font_css = """
    <style>
        div[data-testid="stMetric"] > label[data-testid="stMetricLabel"] > div {
            font-size: 0.85em !important; 
        }
        div[data-testid="stMetric"] > div[data-testid="stMetricValue"] > div {
            font-size: 1.1em !important; 
        }
    </style>
    """
    st.markdown(kpi_font_css, unsafe_allow_html=True)

    # Layout for the main content area
    spacer_left, main_content, spacer_right = st.columns([0.1, 5.8, 0.1]) 

    with main_content:
        available_dates = fetch_available_dates()
        if not available_dates:
            st.error("No available dates found in the database. Please check the data source and connection.")
            return # Use return instead of st.stop() for better app flow
        
        min_avail_date, max_avail_date = min(available_dates), max(available_dates)
        
        # Set default date range for the date input
        default_start_date = max_avail_date - pd.Timedelta(days=6) if max_avail_date - pd.Timedelta(days=6) >= min_avail_date else min_avail_date
        
        # Initialize session state for date range if not already set or invalid
        if 'selected_date_range' not in st.session_state or \
           not (isinstance(st.session_state.selected_date_range, tuple) and len(st.session_state.selected_date_range) == 2) or \
           not (min_avail_date <= st.session_state.selected_date_range[0] <= max_avail_date and \
                min_avail_date <= st.session_state.selected_date_range[1] <= max_avail_date):
            st.session_state.selected_date_range = (default_start_date, max_avail_date)
        
        # Date input widget
        selected_range_tuple = st.date_input(
            "Select date range for analysis:",
            value=st.session_state.selected_date_range,
            min_value=min_avail_date,
            max_value=max_avail_date,
            key="date_range_picker" 
        )
        
        # Update session state with selected date range
        if isinstance(selected_range_tuple, tuple) and len(selected_range_tuple) == 2:
            start_date_obj, end_date_obj = selected_range_tuple
            st.session_state.selected_date_range = (start_date_obj, end_date_obj) 
        else:
            start_date_obj, end_date_obj = st.session_state.selected_date_range 
            st.warning("Please ensure both a start and end date are selected for the range.")
        
        start_date_str = start_date_obj.strftime('%Y-%m-%d')
        end_date_str = end_date_obj.strftime('%Y-%m-%d')
        
        # Day of the week filter
        days_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if 'selected_days_of_week' not in st.session_state:
            st.session_state.selected_days_of_week = days_options 
        
        selected_days = st.multiselect(
            "Filter by Day of the Week (metrics will be based on these days):", 
            options=days_options,
            default=st.session_state.selected_days_of_week,
            key="day_of_week_filter"
        )
        # Ensure at least one day is selected; if not, default to all days
        st.session_state.selected_days_of_week = selected_days if selected_days else days_options
        
        # Fetch raw data for the selected date range
        raw_range_data = fetch_data_for_range(start_date_str, end_date_str)
        
        if raw_range_data.empty:
            st.warning(f"No data found for the selected date range: {start_date_obj.strftime('%b %d, %Y')} to {end_date_obj.strftime('%b %d, %Y')}.")
            return 
        
        # Filter data by selected days of the week
        day_of_week_map_int = {day_name: i for i, day_name in enumerate(days_options)}
        selected_day_indices = [day_of_week_map_int[day_name] for day_name in st.session_state.selected_days_of_week]
        
        data_for_period = raw_range_data[raw_range_data[COL_DATE].dt.dayofweek.isin(selected_day_indices)].copy() 
        
        if data_for_period.empty:
            st.warning(f"No data found for the selected days of the week ({', '.join(st.session_state.selected_days_of_week)}) within the chosen date range.")
            return

        # --- Data Overview for Selected Period (Tabs) ---
        st.subheader("Data Overview")

        # Define the tabs
        tbl_tabs = st.tabs(["Summary", "Average Hourly Data", "Raw Hourly Data"])

        # "Summary" tab content
        with tbl_tabs[0]: 
            s_dict = {} # Dictionary to store summary metrics
            
            # Calculate Avg Daily Max Price
            daily_grouped = data_for_period.groupby(data_for_period[COL_DATE].dt.date)
            if COL_PRICES in data_for_period.columns and pd.api.types.is_numeric_dtype(data_for_period[COL_PRICES]):
                avg_daily_max_price = float(daily_grouped[COL_PRICES].max().mean(skipna=True) or 0)
                s_dict["Avg Daily Max Price (PHP/MWh)"] = f"{avg_daily_max_price:,.2f}" if pd.notna(avg_daily_max_price) and avg_daily_max_price != 0 else "N/A"
            else:
                s_dict["Avg Daily Max Price (PHP/MWh)"] = "N/A (Data Missing)"

            # Max Hourly Total MQ for the range and its date/time
            if COL_TOTAL_MQ in data_for_period.columns and pd.api.types.is_numeric_dtype(data_for_period[COL_TOTAL_MQ]):
                max_hourly_mq = float(data_for_period[COL_TOTAL_MQ].max(skipna=True) or 0)
                if max_hourly_mq != 0:
                    max_mq_row = data_for_period.loc[data_for_period[COL_TOTAL_MQ] == max_hourly_mq].iloc[0]
                    max_mq_date = max_mq_row[COL_DATE].strftime('%Y-%m-%d')
                    max_mq_time = max_mq_row[COL_HOUR].strftime('%H:%M') if isinstance(max_mq_row[COL_HOUR], time) else 'N/A'
                    s_dict["Max Hourly Total MQ (kWh)"] = f"{max_hourly_mq:,.0f}"
                    s_dict["Max MQ Date/Time"] = f"{max_mq_date} {max_mq_time}"
                else:
                    s_dict["Max Hourly Total MQ (kWh)"] = "N/A"
                    s_dict["Max MQ Date/Time"] = "N/A"
            else:
                s_dict["Max Hourly Total MQ (kWh)"] = "N/A (Data Missing)"
                s_dict["Max MQ Date/Time"] = "N/A"

            # Calculate sum for Total MQ, Total BCQ, and WESM
            for c in [COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_WESM]:
                if c in data_for_period.columns and pd.api.types.is_numeric_dtype(data_for_period[c]):
                    try:
                        total_sum_for_period_kwh = float(data_for_period[c].sum(skipna=True) or 0)
                        
                        if c == COL_TOTAL_MQ:
                            label = "Sum Total MQ (MWh)"
                            value_mwh = total_sum_for_period_kwh / 1000
                            display_value = f"{value_mwh:,.3f}" if pd.notna(value_mwh) and value_mwh != 0 else "N/A"
                        elif c == COL_TOTAL_BCQ:
                            label = "Sum Total BCQ (MWh)" 
                            value_mwh = total_sum_for_period_kwh / 1000 
                            display_value = f"{value_mwh:,.3f}" if pd.notna(value_mwh) and value_mwh != 0 else "N/A" 
                        elif c == COL_WESM: 
                            label = "Total WESM (kWh)" 
                            display_value = f"{total_sum_for_period_kwh:,.0f}" if pd.notna(total_sum_for_period_kwh) else "N/A" 
                        
                        s_dict[label] = display_value
                    except Exception as e:
                        st.warning(f"Error calculating total sum for {c}: {e}")
                        if c == COL_TOTAL_MQ: s_dict["Sum Total MQ (MWh)"] = "Error"
                        elif c == COL_TOTAL_BCQ: s_dict["Sum Total BCQ (MWh)"] = "Error"
                        elif c == COL_WESM: s_dict["Total WESM (kWh)"] = "Error"
                else: 
                    if c == COL_TOTAL_MQ: s_dict["Sum Total MQ (MWh)"] = "N/A (Data Missing)"
                    elif c == COL_TOTAL_BCQ: s_dict["Sum Total BCQ (MWh)"] = "N/A (Data Missing)"
                    elif c == COL_WESM: s_dict["Total WESM (kWh)"] = "N/A (Data Missing)"
            
            # Calculate Overall Avg Price
            if COL_PRICES in data_for_period.columns and pd.api.types.is_numeric_dtype(data_for_period[COL_PRICES]):
                try:
                    overall_avg_price = float(data_for_period[COL_PRICES].mean(skipna=True) or 0)
                    s_dict["Overall Avg Price (PHP/MWh)"] = f"{overall_avg_price:,.2f}" if pd.notna(overall_avg_price) and overall_avg_price != 0 else "N/A"
                except Exception as e:
                    st.warning(f"Error calculating overall average price: {e}")
                    s_dict["Overall Avg Price (PHP/MWh)"] = "Error"
            else:
                s_dict["Overall Avg Price (PHP/MWh)"] = "N/A (Data Missing)"

            # Display metrics in the specified 3x2 grid layout
            if s_dict:
                row1_col1, row1_col2, row1_col3 = st.columns(3)
                row2_col1, row2_col2, row2_col3 = st.columns(3)
                
                # First row - specific order: Sum Total MQ, Sum Total BCQ, Total WESM
                with row1_col1:
                    st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 100%;">', unsafe_allow_html=True) 
                    st.metric(label="Sum Total MQ (MWh)", value=str(s_dict.get("Sum Total MQ (MWh)", "N/A")))
                        
                with row1_col2:
                    st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 100%;">', unsafe_allow_html=True)
                    st.metric(label="Sum Total BCQ (MWh)", value=str(s_dict.get("Sum Total BCQ (MWh)", "N/A")))
                        
                with row1_col3:
                    st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 100%;">', unsafe_allow_html=True)
                    st.metric(label="Total WESM (kWh)", value=str(s_dict.get("Total WESM (kWh)", "N/A")))
                
                # Second row - specific order: Overall Avg Price, Max Hourly Total MQ, Avg Daily Max Price
                with row2_col1:
                    st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 100%;">', unsafe_allow_html=True)
                    st.metric(label="Overall Avg Price (PHP/MWh)", value=str(s_dict.get("Overall Avg Price (PHP/MWh)", "N/A")))
                
                with row2_col2:
                    st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 100%;">', unsafe_allow_html=True)
                    col2_container = st.container()
                    col2_container.metric(
                        label="Max Hourly Total MQ (kWh)", 
                        value=str(s_dict.get("Max Hourly Total MQ (kWh)", "N/A"))
                    )
                    st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 100%;">', unsafe_allow_html=True)
                    if s_dict.get("Max MQ Date/Time", "N/A") != "N/A":
                        col2_container.markdown(
                            f"<div style='display: flex; justify-content: center; align-items: flex-start; height: 100%;'><span style='color: gray;'>on {s_dict['Max MQ Date/Time']}</span></div>",
                            unsafe_allow_html=True
                        )
                        
                with row2_col3:
                    st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 100%;">', unsafe_allow_html=True)
                    st.metric(label="Avg Daily Max Price (PHP/MWh)", value=str(s_dict.get("Avg Daily Max Price (PHP/MWh)", "N/A")))
                        
            else:
                st.info("No summary data to display for the selected criteria.")
                
        with tbl_tabs[1]: # "Average Hourly Data" tab
            if COL_HOUR in data_for_period.columns and not data_for_period[COL_HOUR].isnull().all():
                try:
                    # Ensure COL_HOUR_STR is created if not already
                    if COL_HOUR_STR not in data_for_period.columns or data_for_period[COL_HOUR_STR].isnull().all():
                        data_for_period[COL_HOUR_STR] = data_for_period[COL_HOUR].apply(
                            lambda x: x.strftime('%H:%M') if pd.notna(x) and isinstance(x, time) else 'N/A'
                        )

                    hourly_avg_table_data = data_for_period.groupby(COL_HOUR_STR)[
                        [COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_PRICES, COL_WESM]
                    ].mean().reset_index() 
                    
                    # Sorting logic for hours
                    if 'N/A' in hourly_avg_table_data[COL_HOUR_STR].values:
                        valid_hours_df = hourly_avg_table_data[hourly_avg_table_data[COL_HOUR_STR] != 'N/A'].copy()
                        if not valid_hours_df.empty:
                             valid_hours_df['Hour_Sort_Key'] = pd.to_datetime(valid_hours_df[COL_HOUR_STR], format='%H:%M').dt.time
                             valid_hours_df = valid_hours_df.sort_values('Hour_Sort_Key').drop(columns=['Hour_Sort_Key'])
                        na_hours_df = hourly_avg_table_data[hourly_avg_table_data[COL_HOUR_STR] == 'N/A']
                        hourly_avg_table_data = pd.concat([valid_hours_df, na_hours_df], ignore_index=True)
                    else: 
                         hourly_avg_table_data['Hour_Sort_Key'] = pd.to_datetime(hourly_avg_table_data[COL_HOUR_STR], format='%H:%M', errors='coerce').dt.time
                         hourly_avg_table_data = hourly_avg_table_data.sort_values('Hour_Sort_Key', na_position='last').drop(columns=['Hour_Sort_Key'])


                    hourly_avg_table_data.rename(columns={COL_HOUR_STR: 'Time'}, inplace=True)
                    
                    # Format columns for display
                    for col_to_format in [COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_PRICES, COL_WESM]:
                        if col_to_format in hourly_avg_table_data.columns:
                            hourly_avg_table_data[col_to_format] = hourly_avg_table_data[col_to_format].astype(float)
                    
                    st.dataframe(hourly_avg_table_data.style.format(precision=2, na_rep="N/A"), height=300, use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing hourly average table: {e}")
                    # Fallback to display raw head if error occurs
                    st.dataframe(data_for_period[[COL_HOUR, COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_PRICES, COL_WESM]].head(5).style.format(precision=2, na_rep="N/A"), 
                                 height=300, use_container_width=True)
            else:
                st.warning("Hour column not available or all null, cannot display hourly average table.")
        
        with tbl_tabs[2]: # "Raw Hourly Data" tab
            st.markdown("This table shows all hourly records for the selected date range and days of the week.")
            # Ensure COL_HOUR_STR is present for display
            if COL_HOUR_STR not in data_for_period.columns or data_for_period[COL_HOUR_STR].isnull().all():
                 data_for_period[COL_HOUR_STR] = data_for_period[COL_HOUR].apply(
                    lambda x: x.strftime('%H:%M') if pd.notna(x) and isinstance(x, time) else 'N/A'
                )
            
            display_cols_raw = [COL_DATE, COL_HOUR_STR, COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_PRICES, COL_WESM]
            # Filter out columns that might not exist if data fetching failed partially for some
            display_cols_raw = [col for col in display_cols_raw if col in data_for_period.columns]

            if not data_for_period.empty and display_cols_raw:
                # Format date for better readability in the table
                raw_display_df = data_for_period[display_cols_raw].copy()
                raw_display_df[COL_DATE] = raw_display_df[COL_DATE].dt.strftime('%Y-%m-%d')
                st.dataframe(raw_display_df.style.format(precision=2, na_rep="N/A"), height=400, use_container_width=True)
            else:
                st.info("No raw hourly data to display for the selected criteria.")


        # --- Average Hourly Metrics Visualization (Charts) ---
        st.subheader("Metrics Visualization")
        chart_tabs_viz = st.tabs(["Avg MQ, BCQ & Prices by Hour", "Avg WESM & Prices by Hour"]) 
        
        try:
            if COL_HOUR in data_for_period.columns and not data_for_period[COL_HOUR].isnull().all():
                # Create COL_HOUR_STR for chart x-axis if not already present
                if COL_HOUR_STR not in data_for_period.columns or data_for_period[COL_HOUR_STR].isnull().all():
                     data_for_period[COL_HOUR_STR] = data_for_period[COL_HOUR].apply(
                        lambda x: x.strftime('%H:%M') if pd.notna(x) and isinstance(x, time) else 'Unknown'
                    )
                # Group by hour and calculate means for charting
                hourly_avg_df_for_charts = data_for_period.groupby(COL_HOUR_STR).agg({
                    COL_TOTAL_MQ: 'mean', 
                    COL_TOTAL_BCQ: 'mean',
                    COL_WESM: 'mean',
                    COL_PRICES: 'mean'
                }).reset_index()
                
                # Add absolute WESM for chart y-axis (for bar height)
                if COL_WESM in hourly_avg_df_for_charts.columns:
                    hourly_avg_df_for_charts[COL_ABS_WESM] = hourly_avg_df_for_charts[COL_WESM].abs()

                # Sorting logic for hours (to ensure correct order on x-axis)
                if 'Unknown' in hourly_avg_df_for_charts[COL_HOUR_STR].values:
                    known_hours_df = hourly_avg_df_for_charts[hourly_avg_df_for_charts[COL_HOUR_STR] != 'Unknown'].copy()
                    if not known_hours_df.empty:
                        known_hours_df['Hour_Sort_Key'] = pd.to_datetime(known_hours_df[COL_HOUR_STR], format='%H:%M', errors='coerce').dt.time
                        known_hours_df = known_hours_df.sort_values('Hour_Sort_Key').drop(columns=['Hour_Sort_Key'])
                    unknown_hours_df = hourly_avg_df_for_charts[hourly_avg_df_for_charts[COL_HOUR_STR] == 'Unknown']
                    hourly_avg_df_for_charts = pd.concat([known_hours_df, unknown_hours_df], ignore_index=True)
                elif 'N/A' in hourly_avg_df_for_charts[COL_HOUR_STR].values: 
                    known_hours_df = hourly_avg_df_for_charts[hourly_avg_df_for_charts[COL_HOUR_STR] != 'N/A'].copy()
                    if not known_hours_df.empty:
                        known_hours_df['Hour_Sort_Key'] = pd.to_datetime(known_hours_df[COL_HOUR_STR], format='%H:%M', errors='coerce').dt.time
                        known_hours_df = known_hours_df.sort_values('Hour_Sort_Key').drop(columns=['Hour_Sort_Key'])
                    na_hours_df = hourly_avg_df_for_charts[hourly_avg_df_for_charts[COL_HOUR_STR] == 'N/A']
                    hourly_avg_df_for_charts = pd.concat([known_hours_df, na_hours_df], ignore_index=True)   
                else: 
                    if not hourly_avg_df_for_charts.empty: 
                        hourly_avg_df_for_charts['Hour_Sort_Key'] = pd.to_datetime(hourly_avg_df_for_charts[COL_HOUR_STR], format='%H:%M', errors='coerce').dt.time
                        hourly_avg_df_for_charts = hourly_avg_df_for_charts.sort_values('Hour_Sort_Key', na_position='last').drop(columns=['Hour_Sort_Key'])
                
                with chart_tabs_viz[0]: # Avg MQ, BCQ & Prices by Hour
                    if all(c in hourly_avg_df_for_charts.columns for c in [COL_HOUR_STR, COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_PRICES]):
                        base = alt.Chart(hourly_avg_df_for_charts).encode(x=alt.X(f'{COL_HOUR_STR}:N', title='Hour of Day (Average)', sort=None))
                        
                        # Line for Average Total MQ
                        line_mq = base.mark_line(point=True, color='steelblue').encode(
                            y=alt.Y(f'{COL_TOTAL_MQ}:Q', title='Avg Energy (kWh)', axis=alt.Axis(titleColor='steelblue'), scale=alt.Scale(zero=True)),
                            tooltip=[COL_HOUR_STR, alt.Tooltip(f'{COL_TOTAL_MQ}:Q', format=',.0f', title='Avg MQ')]
                        )
                        # Line for Average Total BCQ
                        line_bcq = base.mark_line(point=True, color='orange').encode(
                            y=alt.Y(f'{COL_TOTAL_BCQ}:Q', axis=alt.Axis(titleColor='steelblue'), scale=alt.Scale(zero=True)),
                            tooltip=[COL_HOUR_STR, alt.Tooltip(f'{COL_TOTAL_BCQ}:Q', format=',.0f', title='Avg BCQ')]
                        )
                        # Bar chart for Average Prices
                        line_prices = base.mark_bar(color='green', opacity=0.3).encode(
                            y=alt.Y(f'{COL_PRICES}:Q', title='Avg Price (PHP/kWh)', axis=alt.Axis(titleColor='green'), scale=alt.Scale(zero=False, domain=[0, 32000])),
                            tooltip=[COL_HOUR_STR, alt.Tooltip(f'{COL_PRICES}:Q', format=',.2f', title='Avg Price')]
                        )
                        # Combine charts with independent y-axes
                        combined_chart_mq_bcq_prices = alt.layer(line_mq + line_bcq, line_prices).resolve_scale(y='independent').properties(
                            title=f'Avg Hourly MQ, BCQ & Prices ({", ".join(st.session_state.selected_days_of_week)})', height=400
                        ).configure_axis(labelAngle=-45)
                        st.altair_chart(combined_chart_mq_bcq_prices, use_container_width=True)
                    else: st.warning("Missing data for MQ, BCQ, or Prices chart.")
                
                with chart_tabs_viz[1]: # Avg WESM & Prices by Hour
                    if all(c in hourly_avg_df_for_charts.columns for c in [COL_HOUR_STR, COL_WESM, COL_PRICES, COL_ABS_WESM]):
                        base_wesm = alt.Chart(hourly_avg_df_for_charts).encode(x=alt.X(f'{COL_HOUR_STR}:N', title='Hour of Day (Average)', sort=None))
                        # Bar chart for WESM (colored by sign)
                        bar_wesm = base_wesm.mark_bar(opacity=0.3).encode(
                            y=alt.Y(f'{COL_ABS_WESM}:Q', title='Avg WESM Volume (kWh)', axis=alt.Axis(titleColor='purple')), 
                            color=alt.condition(alt.datum[COL_WESM] > 0, alt.value('#4CAF50'), alt.value('#F44336')), # Green for export, Red for import
                            tooltip=[COL_HOUR_STR, alt.Tooltip(f'{COL_WESM}:Q', format=',.0f', title='Avg WESM (Net)')]
                        )
                        # Line for Average Prices
                        line_prices_wesm = base_wesm.mark_line(point=True, color='green', strokeDash=[3,3]).encode(
                            y=alt.Y(f'{COL_PRICES}:Q', title='Avg Price (PHP/kWh)', axis=alt.Axis(titleColor='green'), scale=alt.Scale(zero=False, domain=[0, 32000])),
                            tooltip=[COL_HOUR_STR, alt.Tooltip(f'{COL_PRICES}:Q', format=',.2f', title='Avg Price')]
                        )
                        # Combine charts with independent y-axes
                        combined_chart_wesm_prices = alt.layer(bar_wesm, line_prices_wesm).resolve_scale(y='independent').properties(
                            title=f'Avg Hourly WESM Volume & Prices ({", ".join(st.session_state.selected_days_of_week)})', height=400
                        ).configure_axis(labelAngle=-45)
                        st.altair_chart(combined_chart_wesm_prices, use_container_width=True)
                        with st.expander("Understanding WESM Values on Chart"):
                            st.markdown("- **Green Bars**: Net Export (BCQ > MQ).\n- **Red Bars**: Net Import (MQ > BCQ).\n- *Bar height represents the volume of WESM interaction.*")
                    else: st.warning("Missing data for WESM or Prices chart.")
            else: st.warning("Hour data column not available, cannot generate hourly charts.")
        except Exception as e:
            st.error(f"An error occurred while creating charts: {e}")
            st.exception(e)
    
        st.subheader("Energy Flow Visualization")
        
        # Combine both Sankey options into a single dropdown
        sankey_options = ['Total Flow']
        if COL_HOUR in data_for_period.columns and not data_for_period[COL_HOUR].isnull().all():
            unique_hours_for_sankey = sorted(
                [h.strftime('%H:%M') for h in data_for_period[COL_HOUR].dropna().unique() if isinstance(h, time)]
            )
            sankey_options += [f"Average Hourly Flow ({hour})" for hour in unique_hours_for_sankey]
        
        selected_sankey_option = st.selectbox(
            "Select Energy Flow Visualization:",
            sankey_options,
            index=0, # Default to 'Total Flow'
            key="sankey_type_selector"
        )
        
        sankey_chart_title_suffix = f"{', '.join(st.session_state.selected_days_of_week)} in {start_date_obj.strftime('%b %d')} - {end_date_obj.strftime('%b %d, %Y')}"
        
        if selected_sankey_option == 'Total Flow':
            total_mq_for_sankey_period = float(data_for_period[COL_TOTAL_MQ].sum(skipna=True) or 0)
            sankey_contributions = fetch_sankey_generator_contributions_total(start_date_str, end_date_str, selected_day_indices)
            sankey_consumptions = fetch_sankey_destination_consumption_total(start_date_str, end_date_str, selected_day_indices)
            node_labels, _, sources, targets, values, node_colors = create_sankey_nodes_links(
                sankey_contributions, sankey_consumptions, total_mq_for_sankey_period, "Total Supply"
            )
            if node_labels:
                fig_title = f"Total Energy Flow (Sum of Range: {sankey_chart_title_suffix})"
                sankey_fig = create_sankey_figure(fig_title, node_labels, sources, targets, values, node_colors)
                st.plotly_chart(sankey_fig, use_container_width=True)
            else:
                st.info(f"Insufficient data to create Total Sankey diagram for {sankey_chart_title_suffix}")
        
        elif selected_sankey_option.startswith('Average Hourly Flow ('):
            selected_hour = selected_sankey_option.split('(')[-1].rstrip(')')
            if COL_HOUR in data_for_period.columns and not data_for_period[COL_HOUR].isnull().all():
                sankey_interval_data = data_for_period[data_for_period[COL_HOUR].apply(
                    lambda x: x.strftime('%H:%M') if isinstance(x, time) else '' 
                ) == selected_hour]
                if not sankey_interval_data.empty:
                    avg_mq_for_sankey_interval = float(sankey_interval_data[COL_TOTAL_MQ].mean(skipna=True) or 0)
                    interval_time_db_format = selected_hour + ":00" # Database expects HH:MM:SS
                    sankey_contributions_avg = fetch_sankey_generator_contributions_averaged(start_date_str, end_date_str, selected_day_indices, interval_time_db_format)
                    sankey_consumptions_avg = fetch_sankey_destination_consumption_averaged(start_date_str, end_date_str, selected_day_indices, interval_time_db_format)
                    node_labels, _, sources, targets, values, node_colors = create_sankey_nodes_links(
                        sankey_contributions_avg, sankey_consumptions_avg, avg_mq_for_sankey_interval, f"Avg Supply at {selected_hour}"
                    )
                    if node_labels:
                        fig_title = f"Avg Hourly Energy Flow at {selected_hour} (Range: {sankey_chart_title_suffix})"
                        sankey_fig = create_sankey_figure(fig_title, node_labels, sources, targets, values, node_colors)
                        st.plotly_chart(sankey_fig, use_container_width=True)
                    else:
                        st.info(f"Insufficient data to create Average Hourly Sankey for {selected_hour} in {sankey_chart_title_suffix}")
                else:
                    st.warning(f"No data for hour {selected_hour} in the selected period.")
            else:
                st.warning("Hour data not available for Average Hourly Sankey.")


def main():
    app_content()

if __name__ == "__main__":
    main()
