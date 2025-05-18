import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, date, time # Added time for type hinting if needed
import altair as alt
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple, Any, Optional # Added Optional

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
        user = st.secrets["database"]["user"]
        password = st.secrets["database"]["password"]
        host = st.secrets["database"]["host"]
        db_name = st.secrets["database"]["db"] 
        port = int(st.secrets["database"]["port"])
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 15})
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
@st.cache_data(ttl=3600)
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

@st.cache_data(ttl=600)
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

        if COL_TIME in df.columns:
            try:
                if not pd.api.types.is_string_dtype(df[COL_TIME]):
                    df[COL_TIME] = df[COL_TIME].astype(str)
                df[COL_HOUR] = pd.to_datetime(df[COL_TIME].str.strip(), format='%H:%M:%S', errors='coerce').dt.time
            except Exception as e:
                st.warning(f"Warning converting '{COL_TIME}' column to time objects: {e}")
                df[COL_HOUR] = pd.NaT

        for col in [COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_PRICES]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

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

GENERATOR_LONG_TO_SHORT_MAP = {
    FDC_MISAMIS_POWER: 'FDC',
    GNPOWER_KAUSWAGAN: 'GNPK',
    PSALM_POWER: 'PSALM',
    SARANGANI_ENERGY: 'SEC',
    THERMA_SOUTH: 'TSI',
    MALITA_POWER: 'MPI'
}
GENERATOR_COLUMNS_TO_SCALE = list(GENERATOR_LONG_TO_SHORT_MAP.keys()) 

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

@st.cache_data(ttl=600)
def fetch_sankey_generator_contributions_averaged(start_date_str: str, end_date_str: str, selected_day_indices: List[int], interval_time_db_format: str) -> Dict[str, float]:
    """Fetches and averages generator contributions for the Sankey diagram."""
    contributions = {short_name: 0.0 for short_name in GENERATOR_LONG_TO_SHORT_MAP.values()}
    if not GENERATOR_LONG_TO_SHORT_MAP:
        st.warning("Generator mapping is empty. Cannot fetch contributions.")
        return contributions
    try:
        engine = get_sqlalchemy_engine()
        query_columns_list = [f'"{col_name}"' for col_name in GENERATOR_LONG_TO_SHORT_MAP.keys()]
        query_columns_str = ', '.join(query_columns_list)

        query = f"""
            SELECT "{COL_DATE}", {query_columns_str}
            FROM "BCQ_Hourly"
            WHERE "{COL_DATE}" BETWEEN %(start_date)s AND %(end_date)s AND "{COL_TIME}" = %(interval_time)s;
        """
        params = {"start_date": start_date_str, "end_date": end_date_str, "interval_time": interval_time_db_format}
        range_interval_data_df = pd.read_sql(query, engine, params=params)

        if range_interval_data_df.empty:
            return contributions

        range_interval_data_df[COL_DATE] = pd.to_datetime(range_interval_data_df[COL_DATE])
        filtered_df = range_interval_data_df[range_interval_data_df[COL_DATE].dt.dayofweek.isin(selected_day_indices)].copy()

        if filtered_df.empty:
            return contributions

        for long_name, short_name in GENERATOR_LONG_TO_SHORT_MAP.items():
            if long_name in filtered_df.columns:
                avg_value = pd.to_numeric(filtered_df[long_name], errors='coerce').mean() 
                if pd.notna(avg_value):
                    if long_name in GENERATOR_COLUMNS_TO_SCALE: 
                        avg_value *= 1000 
                    contributions[short_name] = float(avg_value) if avg_value > 0 else 0.0
        return contributions
    except Exception as e:
        st.error(f"Error fetching averaged Sankey generator contributions: {e}")
        return contributions

@st.cache_data(ttl=600)
def fetch_sankey_destination_consumption_averaged(start_date_str: str, end_date_str: str, selected_day_indices: List[int], interval_time_db_format: str) -> Dict[str, float]:
    """Fetches and averages destination consumption for the Sankey diagram."""
    consumption = {short_name: 0.0 for short_name in DESTINATION_LONG_TO_SHORT_MAP.values()}
    if not DESTINATION_LONG_TO_SHORT_MAP:
        st.warning("Destination mapping is empty. Cannot fetch consumption.")
        return consumption
    try:
        engine = get_sqlalchemy_engine()
        query_columns_list = [f'"{col_name}"' for col_name in DESTINATION_LONG_TO_SHORT_MAP.keys()]
        query_columns_str = ', '.join(query_columns_list)

        query = f"""
            SELECT "{COL_DATE}", {query_columns_str}
            FROM "MQ_Hourly"
            WHERE "{COL_DATE}" BETWEEN %(start_date)s AND %(end_date)s AND "{COL_TIME}" = %(interval_time)s;
        """
        params = {"start_date": start_date_str, "end_date": end_date_str, "interval_time": interval_time_db_format}
        range_interval_data_df = pd.read_sql(query, engine, params=params)

        if range_interval_data_df.empty:
            return consumption

        range_interval_data_df[COL_DATE] = pd.to_datetime(range_interval_data_df[COL_DATE])
        filtered_df = range_interval_data_df[range_interval_data_df[COL_DATE].dt.dayofweek.isin(selected_day_indices)].copy()

        if filtered_df.empty:
            return consumption

        for long_name, short_name in DESTINATION_LONG_TO_SHORT_MAP.items():
            if long_name in filtered_df.columns:
                avg_value = pd.to_numeric(filtered_df[long_name], errors='coerce').mean() 
                if pd.notna(avg_value):
                    consumption[short_name] = float(avg_value) if avg_value > 0 else 0.0
        return consumption
    except Exception as e:
        st.error(f"Error fetching averaged Sankey destination consumption: {e}")
        return consumption

def create_sankey_chart(
    interval_mq_val: float, 
    interval_wesm_val_unscaled: float, 
    chart_title_date_str: str, # Will be appended with "Sum of Range"
    interval_time_hh_mm_str: str, 
    start_date_for_fetch: str, 
    end_date_for_fetch: str,   
    days_indices_for_fetch: List[int] 
) -> Optional[go.Figure]:
    """Creates a Plotly Sankey diagram for average energy flow."""
    if pd.isna(interval_mq_val) or interval_mq_val < 0: 
        st.info(f"Invalid averaged interval data for Sankey ({interval_time_hh_mm_str}, {chart_title_date_str}): Avg MQ = {interval_mq_val:,.0f} kWh")
        return None

    interval_time_db_format = interval_time_hh_mm_str + ":00" 

    scaled_generator_contributions = fetch_sankey_generator_contributions_averaged(
        start_date_for_fetch, end_date_for_fetch, days_indices_for_fetch, interval_time_db_format
    )
    destination_consumptions = fetch_sankey_destination_consumption_averaged(
        start_date_for_fetch, end_date_for_fetch, days_indices_for_fetch, interval_time_db_format
    )

    sum_scaled_generator_contributions = float(sum(v for v in scaled_generator_contributions.values() if pd.notna(v)))
    actual_total_mq_for_interval = float(interval_mq_val) 
    wesm_value_for_sankey = float(sum_scaled_generator_contributions - actual_total_mq_for_interval)

    if sum_scaled_generator_contributions < 0.01 and actual_total_mq_for_interval < 0.01:
        st.info(f"Insufficient averaged flow data for Sankey: {interval_time_hh_mm_str} ({chart_title_date_str})")
        return None

    sankey_node_labels: List[str] = []
    node_indices: Dict[str, int] = {}
    sankey_sources_indices: List[int] = []
    sankey_targets_indices: List[int] = []
    sankey_values: List[float] = []
    
    COLOR_PALETTE = {
        "junction": "#E69F00", "generator": "#0072B2", "wesm_import": "#009E73",
        "load": "#D55E00", "wesm_export": "#CC79A7"
    }
    node_colors: List[str] = []
    
    def add_node(label: str, color: str) -> int:
        if label not in node_indices:
            node_indices[label] = len(sankey_node_labels)
            sankey_node_labels.append(label)
            node_colors.append(color)
        return node_indices[label]

    total_flow_through_junction = float(sum_scaled_generator_contributions)
    if wesm_value_for_sankey < 0: 
        total_flow_through_junction += float(abs(wesm_value_for_sankey))
    
    junction_node_label = f"Avg Max Demand ({total_flow_through_junction:,.0f} kWh)"
    junction_node_idx = add_node(junction_node_label, COLOR_PALETTE["junction"])

    for short_name, value in scaled_generator_contributions.items():
        value = float(value)
        if value > 0.01: 
            percentage = (value / sum_scaled_generator_contributions * 100) if sum_scaled_generator_contributions > 0 else 0
            gen_node_label = f"{short_name} ({value:,.0f} kWh, {percentage:.1f}%)"
            gen_node_idx = add_node(gen_node_label, COLOR_PALETTE["generator"])
            sankey_sources_indices.append(gen_node_idx)
            sankey_targets_indices.append(junction_node_idx)
            sankey_values.append(value)

    if wesm_value_for_sankey < 0: 
        import_value = float(abs(wesm_value_for_sankey))
        if import_value > 0.01:
            percentage = (import_value / total_flow_through_junction * 100) if total_flow_through_junction > 0 else 0
            wesm_import_label = f"WESM Import ({import_value:,.0f} kWh, {percentage:.1f}%)"
            wesm_import_node_idx = add_node(wesm_import_label, COLOR_PALETTE["wesm_import"])
            sankey_sources_indices.append(wesm_import_node_idx)
            sankey_targets_indices.append(junction_node_idx)
            sankey_values.append(import_value)

    sum_destination_consumptions = float(sum(v for v in destination_consumptions.values() if pd.notna(v) and v > 0.01))
    if sum_destination_consumptions > 0.01: 
        for short_name, value in destination_consumptions.items():
            value = float(value)
            if value > 0.01:
                percentage = (value / actual_total_mq_for_interval * 100) if actual_total_mq_for_interval > 0 else 0
                dest_node_label = f"{short_name} ({value:,.0f} kWh, {percentage:.1f}%)"
                dest_node_idx = add_node(dest_node_label, COLOR_PALETTE["load"])
                sankey_sources_indices.append(junction_node_idx)
                sankey_targets_indices.append(dest_node_idx)
                sankey_values.append(value)
    
    if wesm_value_for_sankey > 0: 
        export_value = float(wesm_value_for_sankey)
        if export_value > 0.01:
            percentage = (export_value / total_flow_through_junction * 100) if total_flow_through_junction > 0 else 0
            wesm_export_label = f"WESM Export ({export_value:,.0f} kWh, {percentage:.1f}%)"
            wesm_export_node_idx = add_node(wesm_export_label, COLOR_PALETTE["wesm_export"])
            sankey_sources_indices.append(junction_node_idx)
            sankey_targets_indices.append(wesm_export_node_idx)
            sankey_values.append(export_value)
            
    if not sankey_values or sum(sankey_values) < 0.1: 
        st.info(f"Insufficient averaged energy flow data to build Sankey for {interval_time_hh_mm_str} ({chart_title_date_str})")
        return None
    
    sankey_values = [float(val) for val in sankey_values]
    
    # --- Updated Sankey Title ---
    full_sankey_title = f"Avg Energy Flow: {interval_time_hh_mm_str} (Sum of Range: {chart_title_date_str})"

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap", 
        node=dict(
            pad=20, 
            thickness=15,
            line=dict(color="#A9A9A9", width=0.5), 
            label=sankey_node_labels,
            color=node_colors 
        ),
        link=dict(
            source=sankey_sources_indices,
            target=sankey_targets_indices,
            value=sankey_values,
            hovertemplate='%{source.label} → %{target.label}: %{value:,.0f} kWh<extra></extra>' 
        )
    )])
    
    fig.update_layout(
        title=dict(text=full_sankey_title, font=dict(size=16)), # Use the updated title
        font=dict(family="Arial, sans-serif", size=12), 
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)', 
        height=600, 
        margin=dict(l=20, r=20, t=50, b=20) 
    )
    return fig

def app_content():
    """Main function to render the Streamlit application content."""
    st.title("📊 Energy Trading Dashboard (Averages)")
    st.sidebar.header("Navigation")
    page_options = ["Dashboard", "About"]
    if 'current_page' not in st.session_state: st.session_state.current_page = "Dashboard"
    
    page_key = "nav_radio_main_app" 
    page = st.sidebar.radio("Go to", page_options, index=page_options.index(st.session_state.current_page), key=page_key)
    st.session_state.current_page = page 
    
    if page == "About":
        show_about_page()
    else: 
        show_dashboard()

def show_dashboard():
    """Displays the main dashboard content including filters, KPIs, tables, and charts."""
    
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

    spacer_left, main_content, spacer_right = st.columns([0.1, 5.8, 0.1]) 

    with main_content:
        available_dates = fetch_available_dates()
        if not available_dates:
            st.error("No available dates found in the database. Please check the data source and connection.")
            st.stop() 
        
        min_avail_date, max_avail_date = min(available_dates), max(available_dates)
        
        default_start_date = max_avail_date - pd.Timedelta(days=6) if max_avail_date - pd.Timedelta(days=6) >= min_avail_date else min_avail_date
        if 'selected_date_range' not in st.session_state or \
           not (isinstance(st.session_state.selected_date_range, tuple) and len(st.session_state.selected_date_range) == 2) or \
           not (min_avail_date <= st.session_state.selected_date_range[0] <= max_avail_date and \
                min_avail_date <= st.session_state.selected_date_range[1] <= max_avail_date):
            st.session_state.selected_date_range = (default_start_date, max_avail_date)
        
        selected_range_tuple = st.date_input(
            "Select date range for analysis:",
            value=st.session_state.selected_date_range,
            min_value=min_avail_date,
            max_value=max_avail_date,
            key="date_range_picker" 
        )
        
        if isinstance(selected_range_tuple, tuple) and len(selected_range_tuple) == 2:
            start_date_obj, end_date_obj = selected_range_tuple
            st.session_state.selected_date_range = (start_date_obj, end_date_obj) 
        else:
            start_date_obj, end_date_obj = st.session_state.selected_date_range 
            st.warning("Please ensure both a start and end date are selected for the range.")
        
        start_date_str = start_date_obj.strftime('%Y-%m-%d')
        end_date_str = end_date_obj.strftime('%Y-%m-%d')
        
        days_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if 'selected_days_of_week' not in st.session_state:
            st.session_state.selected_days_of_week = days_options 
        
        selected_days = st.multiselect(
            "Filter by Day of the Week (averages will be based on these days):",
            options=days_options,
            default=st.session_state.selected_days_of_week,
            key="day_of_week_filter"
        )
        st.session_state.selected_days_of_week = selected_days if selected_days else days_options
        
        raw_range_data = fetch_data_for_range(start_date_str, end_date_str)
        
        if raw_range_data.empty:
            st.warning(f"No data found for the selected date range: {start_date_obj.strftime('%b %d, %Y')} to {end_date_obj.strftime('%b %d, %Y')}.")
            return 
        
        day_of_week_map_int = {day_name: i for i, day_name in enumerate(days_options)}
        selected_day_indices = [day_of_week_map_int[day_name] for day_name in st.session_state.selected_days_of_week]
        
        data_for_averaging = raw_range_data[raw_range_data[COL_DATE].dt.dayofweek.isin(selected_day_indices)].copy()
        
        if data_for_averaging.empty:
            st.warning(f"No data found for the selected days of the week ({', '.join(st.session_state.selected_days_of_week)}) within the chosen date range.")
            return

        # --- KPIs: Average Daily Summary Metrics ---
        st.subheader(f"Average Daily Summary Metrics (Range: {start_date_obj.strftime('%b %d, %Y')} to {end_date_obj.strftime('%b %d, %Y')} for {', '.join(st.session_state.selected_days_of_week)})")
        col1, col2, col3, col4 = st.columns(4)
        
        daily_grouped = data_for_averaging.groupby(data_for_averaging[COL_DATE].dt.date)

        if COL_PRICES in data_for_averaging.columns and pd.api.types.is_numeric_dtype(data_for_averaging[COL_PRICES]):
            avg_daily_max_price = float(daily_grouped[COL_PRICES].max().mean(skipna=True) or 0)
            avg_daily_avg_price = float(daily_grouped[COL_PRICES].mean().mean(skipna=True) or 0)
            avg_daily_min_price = float(daily_grouped[COL_PRICES].min().mean(skipna=True) or 0)
            
            col1.metric("Avg Daily Max Price", f"{avg_daily_max_price:,.2f} PHP/kWh" if pd.notna(avg_daily_max_price) and avg_daily_max_price != 0 else "N/A")
            col2.metric("Avg Daily Avg Price", f"{avg_daily_avg_price:,.2f} PHP/kWh" if pd.notna(avg_daily_avg_price) and avg_daily_avg_price != 0 else "N/A")
            col3.metric("Avg Daily Min Price", f"{avg_daily_min_price:,.2f} PHP/kWh" if pd.notna(avg_daily_min_price) and avg_daily_min_price != 0 else "N/A")
        else:
            with col1: st.metric(label="Price N/A", value="-")
            with col2: st.metric(label="Price N/A", value="-")
            with col3: st.metric(label="Price N/A", value="-")


        if COL_TOTAL_MQ in data_for_averaging.columns and pd.api.types.is_numeric_dtype(data_for_averaging[COL_TOTAL_MQ]):
            avg_of_daily_max_mq = float(daily_grouped[COL_TOTAL_MQ].max().mean(skipna=True) or 0)
            col4.metric("Avg Daily Max Total MQ", f"{avg_of_daily_max_mq:,.0f} kWh" if pd.notna(avg_of_daily_max_mq) and avg_of_daily_max_mq != 0 else "N/A", help="Average of the maximum Total MQ recorded each selected day.")
        else:
            with col4: st.metric("Avg Daily Max MQ", "N/A", "MQ N/A")


        # --- Data Tables in Tabs ---
        st.subheader("Data Tables (Averages for Selected Days and Range)")
        tbl_tabs = st.tabs(["Average Hourly Data", "Average of Daily Summaries"])
        
        with tbl_tabs[0]: 
            if COL_HOUR in data_for_averaging.columns and not data_for_averaging[COL_HOUR].isnull().all():
                try:
                    data_for_averaging[COL_HOUR_STR] = data_for_averaging[COL_HOUR].apply(
                        lambda x: x.strftime('%H:%M') if pd.notna(x) and isinstance(x, time) else 'N/A'
                    )
                    hourly_avg_table_data = data_for_averaging.groupby(COL_HOUR_STR)[
                        [COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_PRICES, COL_WESM]
                    ].mean().reset_index() 
                    
                    if 'N/A' in hourly_avg_table_data[COL_HOUR_STR].values:
                        valid_hours_df = hourly_avg_table_data[hourly_avg_table_data[COL_HOUR_STR] != 'N/A'].copy()
                        if not valid_hours_df.empty:
                             valid_hours_df['Hour_Sort_Key'] = pd.to_datetime(valid_hours_df[COL_HOUR_STR], format='%H:%M').dt.time
                             valid_hours_df = valid_hours_df.sort_values('Hour_Sort_Key').drop(columns=['Hour_Sort_Key'])
                        na_hours_df = hourly_avg_table_data[hourly_avg_table_data[COL_HOUR_STR] == 'N/A']
                        hourly_avg_table_data = pd.concat([valid_hours_df, na_hours_df], ignore_index=True)
                    else:
                         hourly_avg_table_data['Hour_Sort_Key'] = pd.to_datetime(hourly_avg_table_data[COL_HOUR_STR], format='%H:%M').dt.time
                         hourly_avg_table_data = hourly_avg_table_data.sort_values('Hour_Sort_Key').drop(columns=['Hour_Sort_Key'])

                    hourly_avg_table_data.rename(columns={COL_HOUR_STR: 'Time (Avg Across Selected Days)'}, inplace=True)
                    
                    for col_to_format in [COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_PRICES, COL_WESM]:
                        if col_to_format in hourly_avg_table_data.columns:
                            hourly_avg_table_data[col_to_format] = hourly_avg_table_data[col_to_format].astype(float)
                    
                    st.dataframe(hourly_avg_table_data.style.format(precision=2, na_rep="N/A"), height=300, use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing hourly average table: {e}")
                    st.dataframe(data_for_averaging[[COL_HOUR, COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_PRICES, COL_WESM]].head(5).style.format(precision=2, na_rep="N/A"), 
                                 height=300, use_container_width=True)
            else:
                st.warning("Hour column not available or all null, cannot display hourly average table.")
        
        with tbl_tabs[1]: 
            s_dict = {} 
            for c in [COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_WESM]:
                if c in data_for_averaging.columns and pd.api.types.is_numeric_dtype(data_for_averaging[c]):
                    try:
                        # Corrected calculation: sum daily totals, then average those daily sums
                        daily_sum_kwh = float(daily_grouped[c].sum() or 0)
                        avg_daily_sum_kwh = float(daily_grouped[c].sum().mean(skipna=True) or 0)
                        
                        if c == COL_TOTAL_MQ:
                            label = "Sum Total MQ (MWh)"
                            value_mwh = daily_sum_kwh / 1000
                            display_value = f"{value_mwh:,.3f}" if pd.notna(value_mwh) and value_mwh != 0 else "N/A"
                        elif c == COL_TOTAL_BCQ:
                            label = "Sum Total BCQ (MWh)" # Corrected label
                            value_mwh = daily_sum_kwh / 1000 # Corrected unit conversion
                            display_value = f"{value_mwh:,.3f}" if pd.notna(value_mwh) and value_mwh != 0 else "N/A" # Corrected formatting
                        else: # For WESM
                            label = f"WESM Sum {c.replace('_', ' ')} (kWh)"
                            display_value = f"{daily_sum_kwh:,.0f}" if pd.notna(daily_sum_kwh) and daily_sum_kwh != 0 else "N/A"
                        
                        s_dict[label] = display_value
                    except Exception as e:
                        st.warning(f"Error calculating average daily sum for {c}: {e}")
                        s_dict[f"Avg Daily Sum {c.replace('_', ' ')} (kWh)"] = "Error" 
                else:
                    if c == COL_TOTAL_MQ:
                        s_dict["Sum Total MQ (MWh)"] = "N/A (Data Missing)"
                    elif c == COL_TOTAL_BCQ:
                        s_dict["Sum Total BCQ (MWh)"] = "N/A (Data Missing)" # Corrected label for missing data
                    else:
                        s_dict[f"Avg Daily Sum {c.replace('_', ' ')} (kWh)"] = "N/A (Data Missing)"
            
            if COL_PRICES in data_for_averaging.columns and pd.api.types.is_numeric_dtype(data_for_averaging[COL_PRICES]):
                try:
                    avg_overall_price = float(daily_grouped[COL_PRICES].mean().mean(skipna=True) or 0)
                    s_dict["Overall Avg Price (PHP/kWh)"] = f"{avg_overall_price:,.2f}" if pd.notna(avg_overall_price) and avg_overall_price != 0 else "N/A"
                except Exception as e:
                    st.warning(f"Error calculating overall average price: {e}")
                    s_dict["Overall Avg Price (PHP/kWh)"] = "Error"
            else:
                s_dict["Overall Avg Price (PHP/kWh)"] = "N/A (Data Missing)"

            if s_dict:
                num_metrics = len(s_dict)
                cols_per_row = min(num_metrics, 3) 
                summary_cols = st.columns(cols_per_row)
                
                col_idx = 0
                for key, value in s_dict.items():
                    with summary_cols[col_idx % cols_per_row]:
                        st.metric(label=key, value=str(value))
                    col_idx +=1
            else:
                st.info("No summary data to display for the selected criteria.")

        # --- Charts ---
        st.subheader("Average Hourly Metrics Visualization")
        chart_tabs = st.tabs(["Avg MQ & BCQ by Hour", "Avg WESM by Hour", "Avg Price by Hour"])
        
        try:
            if COL_HOUR in data_for_averaging.columns and not data_for_averaging[COL_HOUR].isnull().all():
                if COL_HOUR_STR not in data_for_averaging.columns:
                     data_for_averaging[COL_HOUR_STR] = data_for_averaging[COL_HOUR].apply(
                        lambda x: x.strftime('%H:%M') if pd.notna(x) and isinstance(x, time) else 'Unknown'
                    )
                hourly_avg_df_for_charts = data_for_averaging.groupby(COL_HOUR_STR).agg({
                    COL_TOTAL_MQ: 'mean', 
                    COL_TOTAL_BCQ: 'mean',
                    COL_WESM: 'mean',
                    COL_PRICES: 'mean'
                }).reset_index()
                
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
                    hourly_avg_df_for_charts['Hour_Sort_Key'] = pd.to_datetime(hourly_avg_df_for_charts[COL_HOUR_STR], format='%H:%M', errors='coerce').dt.time
                    hourly_avg_df_for_charts = hourly_avg_df_for_charts.sort_values('Hour_Sort_Key').drop(columns=['Hour_Sort_Key'])
                
                with chart_tabs[0]:
                    if all(c in hourly_avg_df_for_charts.columns for c in [COL_HOUR_STR, COL_TOTAL_MQ, COL_TOTAL_BCQ]):
                        mq_bcq_data_melted = pd.melt(
                            hourly_avg_df_for_charts, 
                            id_vars=[COL_HOUR_STR], 
                            value_vars=[COL_TOTAL_MQ, COL_TOTAL_BCQ],
                            var_name='Metric', 
                            value_name='Value (kWh)'
                        )
                        
                        mq_bcq_chart = alt.Chart(mq_bcq_data_melted).mark_line(point=True).encode(
                            x=alt.X(f'{COL_HOUR_STR}:N', title='Hour of Day (Average)', sort=None), 
                            y=alt.Y('Value (kWh):Q', title='Average Energy (kWh)', scale=alt.Scale(zero=False)),
                            color=alt.Color('Metric:N', legend=alt.Legend(title='Metric Type')),
                            tooltip=[COL_HOUR_STR, 'Metric', alt.Tooltip('Value (kWh):Q', format=',.0f')]
                        ).properties(
                            title=f'Average Hourly MQ & BCQ ({", ".join(st.session_state.selected_days_of_week)})',
                            height=400
                        ).configure_axis(labelAngle=-45) 
                        st.altair_chart(mq_bcq_chart, use_container_width=True)
                    else:
                        st.warning("Missing required data columns (Hour, MQ, or BCQ) for the MQ & BCQ chart.")
                
                with chart_tabs[1]:
                    if COL_WESM in hourly_avg_df_for_charts.columns and not hourly_avg_df_for_charts[COL_WESM].isnull().all():
                        wesm_chart = alt.Chart(hourly_avg_df_for_charts).mark_bar().encode(
                            x=alt.X(f'{COL_HOUR_STR}:N', title='Hour of Day (Average)', sort=None),
                            y=alt.Y(f'{COL_WESM}:Q', title='Average WESM (kWh)'),
                            color=alt.condition(
                                alt.datum[COL_WESM] > 0,
                                alt.value('#4CAF50'), 
                                alt.value('#F44336')  
                            ),
                            tooltip=[alt.Tooltip(f'{COL_HOUR_STR}:N', title='Hour'), alt.Tooltip(f'{COL_WESM}:Q', title='Avg WESM (kWh)', format=',.0f')]
                        ).properties(
                            title=f'Average Hourly WESM (+Export/-Import) ({", ".join(st.session_state.selected_days_of_week)})',
                            height=400
                        ).configure_axis(labelAngle=-45)
                        st.altair_chart(wesm_chart, use_container_width=True)
                        with st.expander("Understanding WESM Values on Chart"):
                            st.markdown("- **Positive WESM (Green Bars)**: Indicates net energy export to the grid (BCQ > MQ).\n- **Negative WESM (Red Bars)**: Indicates net energy import from the grid (MQ > BCQ).")
                    else:
                        st.warning("WESM data not available or all null for the WESM chart.")
                
                with chart_tabs[2]:
                    if COL_PRICES in hourly_avg_df_for_charts.columns and not hourly_avg_df_for_charts[COL_PRICES].isnull().all():
                        price_chart = alt.Chart(hourly_avg_df_for_charts).mark_line(point=True, color='#FF9800').encode(
                            x=alt.X(f'{COL_HOUR_STR}:N', title='Hour of Day (Average)', sort=None),
                            y=alt.Y(f'{COL_PRICES}:Q', title='Average Price (PHP/kWh)', scale=alt.Scale(zero=False)),
                            tooltip=[alt.Tooltip(f'{COL_HOUR_STR}:N', title='Hour'), alt.Tooltip(f'{COL_PRICES}:Q', title='Avg Price', format=',.2f')]
                        ).properties(
                            title=f'Average Hourly Prices ({", ".join(st.session_state.selected_days_of_week)})',
                            height=400
                        ).configure_axis(labelAngle=-45)
                        st.altair_chart(price_chart, use_container_width=True)
                    else:
                        st.warning("Price data not available or all null for the price chart.")
            else:
                st.warning("Hour data column not available or contains all null values, cannot generate hourly charts.")
        except Exception as e:
            st.error(f"An error occurred while creating charts: {e}")
    
        # --- Sankey Diagram (Average Energy Flow) ---
        st.subheader("Average Energy Flow Visualization (Sankey Diagram)")
        
        if COL_HOUR in data_for_averaging.columns and not data_for_averaging[COL_HOUR].isnull().all():
            unique_hours_for_sankey = sorted(
                [h.strftime('%H:%M') for h in data_for_averaging[COL_HOUR].dropna().unique() if isinstance(h, time)]
            )
            
            if unique_hours_for_sankey:
                default_sankey_hour = '14:00' if '14:00' in unique_hours_for_sankey else unique_hours_for_sankey[0]
                selected_sankey_hour_str = st.selectbox(
                    "Select hour for average energy flow visualization:", 
                    options=unique_hours_for_sankey,
                    index=unique_hours_for_sankey.index(default_sankey_hour) if default_sankey_hour in unique_hours_for_sankey else 0,
                    key="sankey_hour_selector"
                )
                
                sankey_interval_data = data_for_averaging[data_for_averaging[COL_HOUR].apply(
                    lambda x: x.strftime('%H:%M') if isinstance(x, time) else '' 
                ) == selected_sankey_hour_str]
                
                if not sankey_interval_data.empty:
                    avg_mq_for_sankey_interval = float(sankey_interval_data[COL_TOTAL_MQ].mean(skipna=True) or 0)
                    avg_wesm_for_sankey_interval = float(sankey_interval_data[COL_WESM].mean(skipna=True) or 0) 
                    
                    # Updated title string for Sankey
                    sankey_chart_title_str = f"{', '.join(st.session_state.selected_days_of_week)} in {start_date_obj.strftime('%b %d')} - {end_date_obj.strftime('%b %d, %Y')}"
                    
                    sankey_fig = create_sankey_chart(
                        interval_mq_val=avg_mq_for_sankey_interval,
                        interval_wesm_val_unscaled=avg_wesm_for_sankey_interval, 
                        chart_title_date_str=sankey_chart_title_str, # This part is for the "Sum of Range"
                        interval_time_hh_mm_str=selected_sankey_hour_str,
                        start_date_for_fetch=start_date_str,
                        end_date_for_fetch=end_date_str,
                        days_indices_for_fetch=selected_day_indices
                    )
                    
                    if sankey_fig:
                        st.plotly_chart(sankey_fig, use_container_width=True)
                else:
                    st.warning(f"No data available for hour {selected_sankey_hour_str} within the selected days and date range to generate the Sankey diagram.")
            else:
                st.warning("No valid hours with data available in the filtered set for energy flow visualization.")
        else:
            st.warning("Hour data column not available or all null, cannot generate Sankey diagram.")

def show_about_page():
    """Displays the 'About' page content with information about the dashboard."""
    st.title("About this Dashboard")
    
    st.markdown("""
    ## Energy Trading Dashboard (Averages)
    
    This interactive dashboard is designed for the analysis and visualization of energy trading data, with a focus on presenting **average metrics** across user-selected date ranges and specific days of the week. It aims to help identify trends, patterns, and typical energy behavior rather than displaying raw data for individual moments.
    
    ### Key Features:
    
    1.  **Flexible Data Filtering:**
        * Select custom date ranges for analysis.
        * Filter data by specific days of the week (e.g., only Weekdays, or only Weekends). Averages are then computed based *only* on data from these selected days within the chosen range.
       
    2.  **Insightful KPIs:**
        * **Average Daily Max/Avg/Min Price:** Provides a sense of typical price extremes and central tendency on the selected days.
        * **Average Daily Max Total MQ:** Shows the average of the highest metered quantity observed on the selected types of days.
        
    3.  **Detailed Data Tables (Averages):**
        * **Average Hourly Data:** Displays the mean values for Total MQ, Total BCQ, Prices, and WESM for each hour, averaged over the selected days and date range.
        * **Average of Daily Summaries:** Shows the average of daily total consumption (MQ), daily total contracted quantity (BCQ), daily total WESM volume, and the overall average price.
        
    4.  **Dynamic Visualizations (Averages):**
        * **MQ & BCQ by Hour:** Line chart illustrating the average trend of Metered Quantity and Bilateral Contract Quantity throughout a typical selected day.
        * **WESM by Hour:** Bar chart showing average net energy import (negative/red) or export (positive/green) for each hour.
        * **Price by Hour:** Line chart depicting the average price fluctuation across the hours of a typical selected day.
        * **Energy Flow Sankey Diagram:** Visualizes the average flow of energy for a *selected hour*. It shows contributions from different generation sources (potentially scaled), how this energy meets the average metered demand (MQ) for that hour, and the role of WESM (import/export).
       
    ### Key Terms:
    
    * **MQ (Metered Quantity):** The actual amount of electrical energy consumed or delivered, as measured by a meter. Typically in kWh.
    * **BCQ (Bilateral Contract Quantity):** The amount of electrical energy scheduled or nominated to be bought/sold through direct contracts between parties, outside the spot market. Typically in kWh.
    * **WESM (Wholesale Electricity Spot Market):** The central venue for trading electricity in the Philippines.
    * **WESM (Calculated) = Total BCQ - Total MQ:**
        * A **positive** value suggests that contracted energy (BCQ) exceeds actual consumption (MQ), implying a potential net **export** to the WESM or under-consumption relative to contracts.
        * A **negative** value suggests that actual consumption (MQ) exceeds contracted energy (BCQ), implying a net **import** from the WESM was needed.
    
    ### Understanding the Sankey Diagram:
    
    The Sankey diagram for a selected hour illustrates the *average* energy distribution:
    * **Sources (Left):** Average contribution from various generators (e.g., FDC, GNPK) and potentially WESM Imports. These values are fetched and averaged for the chosen hour over the selected days/range. Generator values might be scaled (e.g., from MWh to kWh) for consistency.
    * **Junction Node (Center):** Represents the total average energy available to meet demand for that hour. This is the sum of average generator outputs and any average WESM import.
    * **Destinations (Right):** Average consumption by different load points or feeders (e.g., M1, M2) and potentially WESM Exports. These are also averaged values for the chosen hour. The sum of these should approximate the average Total MQ for that hour.
    
    ### Data Sources:
    
    The dashboard connects to a PostgreSQL database containing tables such as:
    * `MQ_Hourly`: Hourly metered quantity data, including consumption by various destination nodes.
    * `BCQ_Hourly`: Hourly bilateral contract quantity data, including contributions from various generator sources.
    * `Prices_Hourly`: Hourly electricity price data from the WESM.
    
    ### Important Note on Averages:
    
    All metrics, tables, and charts in this dashboard display **averages** calculated over the user-defined date range and selected days of the week. They do **not** represent instantaneous values for any single specific day or hour but rather the typical behavior observed under the specified conditions. This approach is useful for understanding general patterns and making informed decisions based on historical trends.
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Dashboard Version: 1.2.4\nLast Updated: May 13, 2025") # Updated version

# --- MAIN APP EXECUTION ---
if __name__ == "__main__":
    app_content()
