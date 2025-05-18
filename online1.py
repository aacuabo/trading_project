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
        with engine.connect() as conn: # Test connection
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

# --- SANKEY CHART HELPER FUNCTIONS (MODIFIED FOR TOTALS) ---

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
def fetch_sankey_generator_contributions_total(start_date_str: str, end_date_str: str, selected_day_indices: List[int]) -> Dict[str, float]:
    """Fetches and sums generator contributions for the Sankey diagram over the entire period."""
    contributions = {short_name: 0.0 for short_name in GENERATOR_LONG_TO_SHORT_MAP.values()}
    if not GENERATOR_LONG_TO_SHORT_MAP:
        st.warning("Generator mapping is empty. Cannot fetch contributions.")
        return contributions
    try:
        engine = get_sqlalchemy_engine()
        query_columns_list = [f'SUM("{col_name}") AS "{col_name}_sum"' for col_name in GENERATOR_LONG_TO_SHORT_MAP.keys()]
        query_columns_str = ', '.join(query_columns_list)

        # Note: Grouping by Date first to filter by dayofweek, then summing those daily sums,
        # or fetching all rows and summing in pandas.
        # For simplicity and to ensure correct day filtering, fetch relevant rows then sum in pandas.
        select_cols_for_fetch = '", "'.join(GENERATOR_LONG_TO_SHORT_MAP.keys())
        
        query = f"""
            SELECT "{COL_DATE}", "{select_cols_for_fetch}"
            FROM "BCQ_Hourly"
            WHERE "{COL_DATE}" BETWEEN %(start_date)s AND %(end_date)s;
        """
        params = {"start_date": start_date_str, "end_date": end_date_str}
        period_data_df = pd.read_sql(query, engine, params=params)

        if period_data_df.empty:
            return contributions

        period_data_df[COL_DATE] = pd.to_datetime(period_data_df[COL_DATE])
        filtered_df = period_data_df[period_data_df[COL_DATE].dt.dayofweek.isin(selected_day_indices)].copy()

        if filtered_df.empty:
            return contributions

        for long_name, short_name in GENERATOR_LONG_TO_SHORT_MAP.items():
            if long_name in filtered_df.columns:
                total_value = pd.to_numeric(filtered_df[long_name], errors='coerce').sum() 
                if pd.notna(total_value):
                    if long_name in GENERATOR_COLUMNS_TO_SCALE: 
                        total_value *= 1000 
                    contributions[short_name] = float(total_value) if total_value > 0 else 0.0
        return contributions
    except Exception as e:
        st.error(f"Error fetching total Sankey generator contributions: {e}")
        return contributions

@st.cache_data(ttl=600)
def fetch_sankey_destination_consumption_total(start_date_str: str, end_date_str: str, selected_day_indices: List[int]) -> Dict[str, float]:
    """Fetches and sums destination consumption for the Sankey diagram over the entire period."""
    consumption = {short_name: 0.0 for short_name in DESTINATION_LONG_TO_SHORT_MAP.values()}
    if not DESTINATION_LONG_TO_SHORT_MAP:
        st.warning("Destination mapping is empty. Cannot fetch consumption.")
        return consumption
    try:
        engine = get_sqlalchemy_engine()
        select_cols_for_fetch = '", "'.join(DESTINATION_LONG_TO_SHORT_MAP.keys())
        query = f"""
            SELECT "{COL_DATE}", "{select_cols_for_fetch}"
            FROM "MQ_Hourly"
            WHERE "{COL_DATE}" BETWEEN %(start_date)s AND %(end_date)s;
        """
        params = {"start_date": start_date_str, "end_date": end_date_str}
        period_data_df = pd.read_sql(query, engine, params=params)

        if period_data_df.empty:
            return consumption

        period_data_df[COL_DATE] = pd.to_datetime(period_data_df[COL_DATE])
        filtered_df = period_data_df[period_data_df[COL_DATE].dt.dayofweek.isin(selected_day_indices)].copy()

        if filtered_df.empty:
            return consumption

        for long_name, short_name in DESTINATION_LONG_TO_SHORT_MAP.items():
            if long_name in filtered_df.columns:
                total_value = pd.to_numeric(filtered_df[long_name], errors='coerce').sum() 
                if pd.notna(total_value):
                    consumption[short_name] = float(total_value) if total_value > 0 else 0.0
        return consumption
    except Exception as e:
        st.error(f"Error fetching total Sankey destination consumption: {e}")
        return consumption

def create_sankey_chart_total(
    total_mq_val: float, 
    chart_title_suffix: str, # e.g., "Weekdays in May 01 - May 07, 2025"
    start_date_for_fetch: str, 
    end_date_for_fetch: str,   
    days_indices_for_fetch: List[int] 
) -> Optional[go.Figure]:
    """Creates a Plotly Sankey diagram for TOTAL energy flow over the selected period."""
    if pd.isna(total_mq_val) or total_mq_val < 0: 
        st.info(f"Invalid total MQ data for Sankey ({chart_title_suffix}): Total MQ = {total_mq_val:,.0f} kWh")
        return None

    # Fetch total contributions and consumptions for the entire period
    scaled_generator_contributions_total = fetch_sankey_generator_contributions_total(
        start_date_for_fetch, end_date_for_fetch, days_indices_for_fetch
    )
    destination_consumptions_total = fetch_sankey_destination_consumption_total(
        start_date_for_fetch, end_date_for_fetch, days_indices_for_fetch
    )

    sum_scaled_generator_contributions = float(sum(v for v in scaled_generator_contributions_total.values() if pd.notna(v)))
    
    # WESM value for Sankey logic: Based on the sum of scaled generator contributions and the total MQ
    wesm_value_for_sankey = float(sum_scaled_generator_contributions - total_mq_val)

    if sum_scaled_generator_contributions < 0.01 and total_mq_val < 0.01:
        st.info(f"Insufficient total flow data for Sankey: {chart_title_suffix}")
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
    
    junction_node_label = f"Total Supply ({total_flow_through_junction:,.0f} kWh)"
    junction_node_idx = add_node(junction_node_label, COLOR_PALETTE["junction"])

    for short_name, value in scaled_generator_contributions_total.items():
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

    sum_destination_consumptions = float(sum(v for v in destination_consumptions_total.values() if pd.notna(v) and v > 0.01))
    if sum_destination_consumptions > 0.01: 
        for short_name, value in destination_consumptions_total.items():
            value = float(value)
            if value > 0.01:
                percentage = (value / total_mq_val * 100) if total_mq_val > 0 else 0
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
        st.info(f"Insufficient total energy flow data to build Sankey for {chart_title_suffix}")
        return None
    
    sankey_values = [float(val) for val in sankey_values]
    
    full_sankey_title = f"Total Energy Flow (Sum of Range: {chart_title_suffix})"

    fig = go.Figure(data=[go.Sankey(
        arrangement="snap", 
        node=dict(pad=20, thickness=15, line=dict(color="#A9A9A9", width=0.5), label=sankey_node_labels, color=node_colors),
        link=dict(source=sankey_sources_indices, target=sankey_targets_indices, value=sankey_values, hovertemplate='%{source.label} → %{target.label}: %{value:,.0f} kWh<extra></extra>')
    )])
    
    fig.update_layout(
        title=dict(text=full_sankey_title, font=dict(size=16)),
        font=dict(family="Arial, sans-serif", size=12), 
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
        height=600, margin=dict(l=20, r=20, t=50, b=20) 
    )
    return fig

def app_content():
    """Main function to render the Streamlit application content."""
    st.title("📊 Energy Trading Dashboard (Averages & Totals)") # Updated title
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
            "Filter by Day of the Week (metrics will be based on these days):", # Updated help text
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
        
        data_for_period = raw_range_data[raw_range_data[COL_DATE].dt.dayofweek.isin(selected_day_indices)].copy() # Renamed for clarity
        
        if data_for_period.empty:
            st.warning(f"No data found for the selected days of the week ({', '.join(st.session_state.selected_days_of_week)}) within the chosen date range.")
            return

        # --- KPIs: Average Daily Summary Metrics (still averages for these top-level KPIs) ---
        st.subheader(f"Average Daily Summary Metrics (Range: {start_date_obj.strftime('%b %d, %Y')} to {end_date_obj.strftime('%b %d, %Y')} for {', '.join(st.session_state.selected_days_of_week)})")
        # Using 4 columns for the top KPIs as originally designed.
        # If a different layout (e.g. 2x2) is desired, this part needs specific instructions.
        col1, col2, col3, col4 = st.columns(4) 
        
        daily_grouped = data_for_period.groupby(data_for_period[COL_DATE].dt.date)

        if COL_PRICES in data_for_period.columns and pd.api.types.is_numeric_dtype(data_for_period[COL_PRICES]):
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

        if COL_TOTAL_MQ in data_for_period.columns and pd.api.types.is_numeric_dtype(data_for_period[COL_TOTAL_MQ]):
            avg_of_daily_max_mq = float(daily_grouped[COL_TOTAL_MQ].max().mean(skipna=True) or 0)
            col4.metric("Avg Daily Max Total MQ", f"{avg_of_daily_max_mq:,.0f} kWh" if pd.notna(avg_of_daily_max_mq) and avg_of_daily_max_mq != 0 else "N/A", help="Average of the maximum Total MQ recorded each selected day.")
        else:
            with col4: st.metric("Avg Daily Max MQ", "N/A", "MQ N/A")

        # --- Data Tables and Summaries in Tabs ---
        st.subheader("Data Overview for Selected Period")
        # Renamed tabs and reordered
        tbl_tabs = st.tabs(["Summary (Totals & Overall Avg)", "Average Hourly Data"]) 
        
        with tbl_tabs[0]: # Renamed to "Summary" and now the first tab
            s_dict = {} 
            for c in [COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_WESM]:
                if c in data_for_period.columns and pd.api.types.is_numeric_dtype(data_for_period[c]):
                    try:
                        # Calculate TOTAL SUM for the entire selected period and days
                        total_sum_for_period_kwh = float(data_for_period[c].sum(skipna=True) or 0)
                        
                        if c == COL_TOTAL_MQ:
                            label = "Sum Total MQ (MWh)"
                            value_mwh = total_sum_for_period_kwh / 1000
                            display_value = f"{value_mwh:,.3f}" if pd.notna(value_mwh) and value_mwh != 0 else "N/A"
                        elif c == COL_TOTAL_BCQ:
                            label = "Sum Total BCQ (MWh)" 
                            value_mwh = total_sum_for_period_kwh / 1000 
                            display_value = f"{value_mwh:,.3f}" if pd.notna(value_mwh) and value_mwh != 0 else "N/A" 
                        elif c == COL_WESM: # Explicitly handle WESM
                            label = "Total WESM (kWh)" # Changed label
                            display_value = f"{total_sum_for_period_kwh:,.0f}" if pd.notna(total_sum_for_period_kwh) else "N/A" # WESM can be 0
                        
                        s_dict[label] = display_value
                    except Exception as e:
                        st.warning(f"Error calculating total sum for {c}: {e}")
                        # Fallback labels in case of error
                        if c == COL_TOTAL_MQ: s_dict["Sum Total MQ (MWh)"] = "Error"
                        elif c == COL_TOTAL_BCQ: s_dict["Sum Total BCQ (MWh)"] = "Error"
                        elif c == COL_WESM: s_dict["Total WESM (kWh)"] = "Error"
                else: # Data missing for the column
                    if c == COL_TOTAL_MQ: s_dict["Sum Total MQ (MWh)"] = "N/A (Data Missing)"
                    elif c == COL_TOTAL_BCQ: s_dict["Sum Total BCQ (MWh)"] = "N/A (Data Missing)"
                    elif c == COL_WESM: s_dict["Total WESM (kWh)"] = "N/A (Data Missing)"
            
            # Overall Average Price for the period (this remains an average)
            if COL_PRICES in data_for_period.columns and pd.api.types.is_numeric_dtype(data_for_period[COL_PRICES]):
                try:
                    # Calculate the average price over the entire 'data_for_period'
                    overall_avg_price = float(data_for_period[COL_PRICES].mean(skipna=True) or 0)
                    s_dict["Overall Avg Price (PHP/kWh)"] = f"{overall_avg_price:,.2f}" if pd.notna(overall_avg_price) and overall_avg_price != 0 else "N/A"
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

        with tbl_tabs[1]: # "Average Hourly Data" - now the second tab
            if COL_HOUR in data_for_period.columns and not data_for_period[COL_HOUR].isnull().all():
                try:
                    data_for_period[COL_HOUR_STR] = data_for_period[COL_HOUR].apply(
                        lambda x: x.strftime('%H:%M') if pd.notna(x) and isinstance(x, time) else 'N/A'
                    )
                    hourly_avg_table_data = data_for_period.groupby(COL_HOUR_STR)[
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
                    st.dataframe(data_for_period[[COL_HOUR, COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_PRICES, COL_WESM]].head(5).style.format(precision=2, na_rep="N/A"), 
                                 height=300, use_container_width=True)
            else:
                st.warning("Hour column not available or all null, cannot display hourly average table.")
        

        # --- Average Hourly Metrics Visualization (Charts) ---
        st.subheader("Average Hourly Metrics Visualization")
        chart_tabs_viz = st.tabs(["Avg MQ, BCQ & Prices by Hour", "Avg WESM & Prices by Hour"]) # Renamed for clarity
        
        try:
            if COL_HOUR in data_for_period.columns and not data_for_period[COL_HOUR].isnull().all():
                if COL_HOUR_STR not in data_for_period.columns: # Ensure Hour_Str is present
                     data_for_period[COL_HOUR_STR] = data_for_period[COL_HOUR].apply(
                        lambda x: x.strftime('%H:%M') if pd.notna(x) and isinstance(x, time) else 'Unknown'
                    )
                hourly_avg_df_for_charts = data_for_period.groupby(COL_HOUR_STR).agg({
                    COL_TOTAL_MQ: 'mean', 
                    COL_TOTAL_BCQ: 'mean',
                    COL_WESM: 'mean',
                    COL_PRICES: 'mean'
                }).reset_index()
                
                # Sort by hour properly
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
                
                # Chart 1: Avg MQ, BCQ & Prices by Hour
                with chart_tabs_viz[0]:
                    if all(c in hourly_avg_df_for_charts.columns for c in [COL_HOUR_STR, COL_TOTAL_MQ, COL_TOTAL_BCQ, COL_PRICES]):
                        base = alt.Chart(hourly_avg_df_for_charts).encode(
                            x=alt.X(f'{COL_HOUR_STR}:N', title='Hour of Day (Average)', sort=None)
                        )
                        
                        line_mq = base.mark_line(point=True, color='steelblue').encode(
                            y=alt.Y(f'{COL_TOTAL_MQ}:Q', title='Avg Energy (kWh)', axis=alt.Axis(titleColor='steelblue'), scale=alt.Scale(zero=True)),
                            tooltip=[COL_HOUR_STR, alt.Tooltip(f'{COL_TOTAL_MQ}:Q', format=',.0f', title='Avg MQ')]
                        )
                        line_bcq = base.mark_line(point=True, color='orange').encode(
                            y=alt.Y(f'{COL_TOTAL_BCQ}:Q', axis=alt.Axis(titleColor='steelblue'), scale=alt.Scale(zero=True)), # Share MQ/BCQ axis
                            tooltip=[COL_HOUR_STR, alt.Tooltip(f'{COL_TOTAL_BCQ}:Q', format=',.0f', title='Avg BCQ')]
                        )
                        line_prices = base.mark_line(point=True, color='green', strokeDash=[3,3]).encode(
                            y=alt.Y(f'{COL_PRICES}:Q', title='Avg Price (PHP/kWh)', axis=alt.Axis(titleColor='green'), scale=alt.Scale(zero=False)), # Prices usually not zero-based
                            tooltip=[COL_HOUR_STR, alt.Tooltip(f'{COL_PRICES}:Q', format=',.2f', title='Avg Price')]
                        )
                        
                        # Layer the charts and resolve scales for multiple y-axes
                        combined_chart_mq_bcq_prices = alt.layer(line_mq + line_bcq, line_prices).resolve_scale(
                            y='independent'
                        ).properties(
                            title=f'Avg Hourly MQ, BCQ & Prices ({", ".join(st.session_state.selected_days_of_week)})',
                            height=400
                        ).configure_axis(labelAngle=-45)
                        st.altair_chart(combined_chart_mq_bcq_prices, use_container_width=True)
                    else:
                        st.warning("Missing data for MQ, BCQ, or Prices chart.")
                
                # Chart 2: Avg WESM & Prices by Hour
                with chart_tabs_viz[1]:
                    if all(c in hourly_avg_df_for_charts.columns for c in [COL_HOUR_STR, COL_WESM, COL_PRICES]):
                        base_wesm = alt.Chart(hourly_avg_df_for_charts).encode(
                            x=alt.X(f'{COL_HOUR_STR}:N', title='Hour of Day (Average)', sort=None)
                        )
                        bar_wesm = base_wesm.mark_bar().encode(
                            y=alt.Y(f'{COL_WESM}:Q', title='Avg WESM (kWh)', axis=alt.Axis(titleColor='purple')),
                            color=alt.condition(alt.datum[COL_WESM] > 0, alt.value('#4CAF50'), alt.value('#F44336')),
                            tooltip=[COL_HOUR_STR, alt.Tooltip(f'{COL_WESM}:Q', format=',.0f', title='Avg WESM')]
                        )
                        line_prices_wesm = base_wesm.mark_line(point=True, color='green', strokeDash=[3,3]).encode(
                            y=alt.Y(f'{COL_PRICES}:Q', title='Avg Price (PHP/kWh)', axis=alt.Axis(titleColor='green'), scale=alt.Scale(zero=False)),
                            tooltip=[COL_HOUR_STR, alt.Tooltip(f'{COL_PRICES}:Q', format=',.2f', title='Avg Price')]
                        )
                        combined_chart_wesm_prices = alt.layer(bar_wesm, line_prices_wesm).resolve_scale(
                            y='independent'
                        ).properties(
                            title=f'Avg Hourly WESM & Prices ({", ".join(st.session_state.selected_days_of_week)})',
                            height=400
                        ).configure_axis(labelAngle=-45)
                        st.altair_chart(combined_chart_wesm_prices, use_container_width=True)
                        with st.expander("Understanding WESM Values on Chart"):
                            st.markdown("- **Positive WESM (Green Bars)**: Net Export.\n- **Negative WESM (Red Bars)**: Net Import.")
                    else:
                        st.warning("Missing data for WESM or Prices chart.")
            else:
                st.warning("Hour data column not available, cannot generate hourly charts.")
        except Exception as e:
            st.error(f"An error occurred while creating charts: {e}")
            st.exception(e) # Provides full traceback for debugging
    
        # --- Sankey Diagram (TOTAL Energy Flow for the selected period) ---
        st.subheader("Total Energy Flow Visualization (Sankey Diagram)")
        
        # Calculate total MQ for the entire selected period for Sankey
        total_mq_for_sankey_period = float(data_for_period[COL_TOTAL_MQ].sum(skipna=True) or 0)
            
        sankey_chart_title_suffix = f"{', '.join(st.session_state.selected_days_of_week)} in {start_date_obj.strftime('%b %d')} - {end_date_obj.strftime('%b %d, %Y')}"
        
        sankey_fig_total = create_sankey_chart_total(
            total_mq_val=total_mq_for_sankey_period,
            chart_title_suffix=sankey_chart_title_suffix,
            start_date_for_fetch=start_date_str,
            end_date_for_fetch=end_date_str,
            days_indices_for_fetch=selected_day_indices
        )
        
        if sankey_fig_total:
            st.plotly_chart(sankey_fig_total, use_container_width=True)
        # else: create_sankey_chart_total will show an st.info message if data is insufficient


def show_about_page():
    """Displays the 'About' page content with information about the dashboard."""
    st.title("About this Dashboard")
    
    st.markdown("""
    ## Energy Trading Dashboard (Averages & Totals)
    
    This interactive dashboard is designed for the analysis and visualization of energy trading data. It presents:
    - **Average daily metrics** for top-level KPIs.
    - **Total sums** for MQ, BCQ, and WESM over the selected period in the 'Summary' tab.
    - **Average hourly trends** for MQ, BCQ, WESM, and Prices.
    - A **total energy flow Sankey diagram** summarizing the entire selected period.
    
    ### Key Features:
    
    1.  **Flexible Data Filtering:**
        * Select custom date ranges.
        * Filter by specific days of the week. 
       
    2.  **Insightful KPIs & Summaries:**
        * **Average Daily Metrics:** Max/Avg/Min Price, Max Total MQ.
        * **Period Totals (Summary Tab):** Sum Total MQ (MWh), Sum Total BCQ (MWh), Total WESM (kWh), and Overall Average Price for the selected period.
        
    3.  **Detailed Data Tables:**
        * **Average Hourly Data:** Mean values for MQ, BCQ, Prices, WESM for each hour, averaged over selected days/range.
        
    4.  **Dynamic Visualizations:**
        * **Average Hourly Charts:** Combined line charts for MQ, BCQ, and Prices; and WESM (bars) with Prices, both featuring dual y-axes.
        * **Total Energy Flow Sankey Diagram:** Visualizes the aggregate flow of energy over the entire selected date range and days, showing total contributions from generation sources, how this energy meets the total metered demand (MQ), and the net WESM interaction.
       
    ### Key Terms:
    (Definitions remain the same)
    * **MQ (Metered Quantity):** kWh.
    * **BCQ (Bilateral Contract Quantity):** kWh.
    * **WESM (Wholesale Electricity Spot Market)**
    * **WESM (Calculated) = Total BCQ - Total MQ**
    
    ### Understanding the Sankey Diagram (Total Flow):
    
    The Sankey diagram now illustrates the **total** energy distribution for the entire selected period (filtered by days of the week):
    * **Sources (Left):** Total contribution from various generators and WESM Imports over the period.
    * **Junction Node (Center):** Total energy supplied to meet demand over the period.
    * **Destinations (Right):** Total consumption by different load points and WESM Exports over the period.
    
    ### Data Sources:
    (Data sources remain the same)
    * `MQ_Hourly`, `BCQ_Hourly`, `Prices_Hourly`
    
    ### Note on Metrics:
    Top-level KPIs and hourly charts show **averages**. The 'Summary' tab and the main Sankey diagram show **totals** for the selected period.
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Dashboard Version: 1.3.0\nLast Updated: May 18, 2025") # Updated version

# --- MAIN APP EXECUTION ---
if __name__ == "__main__":
    app_content()
