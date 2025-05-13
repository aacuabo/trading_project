
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, date, time # Added time for type hinting if needed
import altair as alt
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple, Any, Optional # Added Optional

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
        db_name = st.secrets["database"]["db"] # Renamed to avoid conflict with db module
        port = int(st.secrets["database"]["port"])
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 15})
        with engine.connect() as conn:
            pass # Test connection
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
def fetch_data_for_range(start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """Fetches hourly MQ, BCQ, and Prices data for a selected date range."""
    try:
        engine = get_sqlalchemy_engine()
        query = """
            SELECT mq."Date", mq."Time", mq."Total_MQ", bcq."Total_BCQ", p."Prices"
            FROM "MQ_Hourly" AS mq
            LEFT JOIN "BCQ_Hourly" AS bcq ON mq."Date" = bcq."Date" AND mq."Time" = bcq."Time"
            LEFT JOIN "Prices_Hourly" AS p ON mq."Date" = p."Date" AND mq."Time" = p."Time"
            WHERE mq."Date" BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY mq."Date", mq."Time";
        """
        df = pd.read_sql(query, engine, params={"start_date": start_date_str, "end_date": end_date_str})

        if df.empty:
            return pd.DataFrame()

        df['Date'] = pd.to_datetime(df['Date'])

        if 'Time' in df.columns:
            try:
                if not pd.api.types.is_string_dtype(df['Time']):
                    df['Time'] = df['Time'].astype(str)
                df['Hour'] = pd.to_datetime(df['Time'].str.strip(), format='%H:%M:%S', errors='coerce').dt.time
            except Exception as e:
                st.warning(f"Warning converting 'Time' column to time objects: {e}")
                df['Hour'] = pd.NaT

        for col in ["Total_MQ", "Total_BCQ", "Prices"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if all(c in df.columns for c in ["Total_BCQ", "Total_MQ"]) and \
           pd.api.types.is_numeric_dtype(df["Total_BCQ"]) and \
           pd.api.types.is_numeric_dtype(df["Total_MQ"]):
            df['WESM'] = df['Total_BCQ'] - df['Total_MQ']
        else:
            df['WESM'] = pd.NA
        return df
    except Exception as e:
        st.error(f"Error fetching data for range: {e}")
        return pd.DataFrame()

# --- SANKEY CHART HELPER FUNCTIONS ---

GENERATOR_LONG_TO_SHORT_MAP = {
    "FDC_Misamis_Power_Corporation__FDC": 'FDC',
    "GNPower_Kauswagan_Ltd._Co._GNPKLCO": 'GNPK',
    "Power_Sector_Assets_and_Liabilities_Management_Corporation_PSAL": 'PSALM',
    "Sarangani_Energy_Corporation_SEC": 'SEC',
    "Therma_South,_Inc._TSI": 'TSI',
    "Malita_Power_Inc._SMCPC": 'MPI'
}
GENERATOR_COLUMNS_TO_SCALE = list(GENERATOR_LONG_TO_SHORT_MAP.keys())

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
def fetch_sankey_generator_contributions_averaged(start_date_str: str, end_date_str: str, selected_day_indices: List[int], interval_time_db_format: str) -> Dict[str, float]:
    contributions = {short_name: 0.0 for short_name in GENERATOR_LONG_TO_SHORT_MAP.values()}
    if not GENERATOR_LONG_TO_SHORT_MAP:
        st.warning("Generator mapping is empty. Cannot fetch contributions.")
        return contributions
    try:
        engine = get_sqlalchemy_engine()
        query_columns_list = [f'"{col_name}"' for col_name in GENERATOR_LONG_TO_SHORT_MAP.keys()]
        query_columns_str = ', '.join(query_columns_list)

        query = f"""
            SELECT "Date", {query_columns_str}
            FROM "BCQ_Hourly"
            WHERE "Date" BETWEEN %(start_date)s AND %(end_date)s AND "Time" = %(interval_time)s;
        """
        params = {"start_date": start_date_str, "end_date": end_date_str, "interval_time": interval_time_db_format}
        range_interval_data_df = pd.read_sql(query, engine, params=params)

        if range_interval_data_df.empty:
            return contributions

        range_interval_data_df['Date'] = pd.to_datetime(range_interval_data_df['Date'])
        filtered_df = range_interval_data_df[range_interval_data_df['Date'].dt.dayofweek.isin(selected_day_indices)].copy()

        if filtered_df.empty:
            return contributions

        for long_name, short_name in GENERATOR_LONG_TO_SHORT_MAP.items():
            if long_name in filtered_df.columns:
                avg_value = pd.to_numeric(filtered_df[long_name], errors='coerce').mean()
                if pd.notna(avg_value):
                    if long_name in GENERATOR_COLUMNS_TO_SCALE:
                        avg_value *= 1000
                    contributions[short_name] = float(avg_value) if avg_value > 0 else 0.0  # Explicitly convert to float
        return contributions
    except Exception as e:
        st.error(f"Error fetching averaged Sankey generator contributions: {e}")
        return contributions

@st.cache_data(ttl=600)
def fetch_sankey_destination_consumption_averaged(start_date_str: str, end_date_str: str, selected_day_indices: List[int], interval_time_db_format: str) -> Dict[str, float]:
    consumption = {short_name: 0.0 for short_name in DESTINATION_LONG_TO_SHORT_MAP.values()}
    if not DESTINATION_LONG_TO_SHORT_MAP:
        st.warning("Destination mapping is empty. Cannot fetch consumption.")
        return consumption
    try:
        engine = get_sqlalchemy_engine()
        query_columns_list = [f'"{col_name}"' for col_name in DESTINATION_LONG_TO_SHORT_MAP.keys()]
        query_columns_str = ', '.join(query_columns_list)

        query = f"""
            SELECT "Date", {query_columns_str}
            FROM "MQ_Hourly"
            WHERE "Date" BETWEEN %(start_date)s AND %(end_date)s AND "Time" = %(interval_time)s;
        """
        params = {"start_date": start_date_str, "end_date": end_date_str, "interval_time": interval_time_db_format}
        range_interval_data_df = pd.read_sql(query, engine, params=params)

        if range_interval_data_df.empty:
            return consumption

        range_interval_data_df['Date'] = pd.to_datetime(range_interval_data_df['Date'])
        filtered_df = range_interval_data_df[range_interval_data_df['Date'].dt.dayofweek.isin(selected_day_indices)].copy()

        if filtered_df.empty:
            return consumption

        for long_name, short_name in DESTINATION_LONG_TO_SHORT_MAP.items():
            if long_name in filtered_df.columns:
                avg_value = pd.to_numeric(filtered_df[long_name], errors='coerce').mean()
                if pd.notna(avg_value):
                    consumption[short_name] = float(avg_value) if avg_value > 0 else 0.0  # Explicitly convert to float
        return consumption
    except Exception as e:
        st.error(f"Error fetching averaged Sankey destination consumption: {e}")
        return consumption

def create_sankey_chart(
    interval_mq_val: float,
    interval_wesm_val_unscaled: float,
    chart_title_date_str: str, # Descriptive string for title
    interval_time_hh_mm_str: str,
    start_date_for_fetch: str, # Added
    end_date_for_fetch: str,   # Added
    days_indices_for_fetch: List[int] # Added
) -> Optional[go.Figure]: # Updated return type hint with Optional
    if pd.isna(interval_mq_val) or interval_mq_val < 0: # Values are now averages
        st.info(f"Invalid averaged interval data ({interval_time_hh_mm_str}, {chart_title_date_str}): Avg MQ = {interval_mq_val:,.0f} kWh")
        return None

    interval_time_db_format = interval_time_hh_mm_str + ":00"

    scaled_generator_contributions = fetch_sankey_generator_contributions_averaged(
        start_date_for_fetch, end_date_for_fetch, days_indices_for_fetch, interval_time_db_format
    )
    destination_consumptions = fetch_sankey_destination_consumption_averaged(
        start_date_for_fetch, end_date_for_fetch, days_indices_for_fetch, interval_time_db_format
    )

    # Ensure all values are explicitly floats
    sum_scaled_generator_contributions = float(sum(v for v in scaled_generator_contributions.values() if pd.notna(v)))
    actual_total_mq_for_interval = float(interval_mq_val) # This is now an average MQ for the representative interval

    # WESM for Sankey is based on these averaged, possibly scaled, values
    wesm_value_for_sankey = float(sum_scaled_generator_contributions - actual_total_mq_for_interval)

    if sum_scaled_generator_contributions < 0.01 and actual_total_mq_for_interval < 0.01:
        st.info(f"Insufficient averaged flow data for {interval_time_hh_mm_str} ({chart_title_date_str})")
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
    if wesm_value_for_sankey < 0: # Net import for the Sankey logic
        total_flow_through_junction += float(abs(wesm_value_for_sankey))
    
    # Title reflects averages
    junction_node_label = f"Avg Max Demand ({total_flow_through_junction:,.0f} kWh)"
    junction_node_idx = add_node(junction_node_label, COLOR_PALETTE["junction"])

    for short_name, value in scaled_generator_contributions.items():
        value = float(value)  # Ensure float type
        if value > 0.01:
            percentage = (value / sum_scaled_generator_contributions * 100) if sum_scaled_generator_contributions > 0 else 0
            gen_node_label = f"{short_name} ({value:,.0f} kWh, {percentage:.1f}%)"
            gen_node_idx = add_node(gen_node_label, COLOR_PALETTE["generator"])
            sankey_sources_indices.append(gen_node_idx)
            sankey_targets_indices.append(junction_node_idx)
            sankey_values.append(value)

    if wesm_value_for_sankey < 0: # Net import contributes to junction
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
            value = float(value)  # Ensure float type
            if value > 0.01:
                percentage = (value / actual_total_mq_for_interval * 100) if actual_total_mq_for_interval > 0 else 0
                dest_node_label = f"{short_name} ({value:,.0f} kWh, {percentage:.1f}%)"
                dest_node_idx = add_node(dest_node_label, COLOR_PALETTE["load"])
                sankey_sources_indices.append(junction_node_idx)
                sankey_targets_indices.append(dest_node_idx)
                sankey_values.append(value)
    
    if wesm_value_for_sankey > 0: # Net export flows from junction
        export_value = float(wesm_value_for_sankey)
        if export_value > 0.01:
            percentage = (export_value / total_flow_through_junction * 100) if total_flow_through_junction > 0 else 0 # Percentage of total supply
            wesm_export_label = f"WESM Export ({export_value:,.0f} kWh, {percentage:.1f}%)"
            wesm_export_node_idx = add_node(wesm_export_label, COLOR_PALETTE["wesm_export"])
            sankey_sources_indices.append(junction_node_idx)
            sankey_targets_indices.append(wesm_export_node_idx)
            sankey_values.append(export_value)
            
    if not sankey_values or sum(sankey_values) < 0.1:
        st.info(f"Insufficient averaged energy flow for {interval_time_hh_mm_str} ({chart_title_date_str})")
        return None
    
    # Ensure all values are float to avoid type issues
    sankey_values = [float(val) for val in sankey_values]
    
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(pad=20, thickness=15, line=dict(color="#A9A9A9", width=0.5), label=sankey_node_labels, color=node_colors),
        link=dict(source=sankey_sources_indices, target=sankey_targets_indices, value=sankey_values, hovertemplate='%{source.label} → %{target.label}: %{value:,.0f} kWh<extra></extra>')
    )])
    
    fig.update_layout(
        title=dict(text=f"Avg Energy Flow: {interval_time_hh_mm_str}, {chart_title_date_str}", font=dict(size=16)), # Updated title
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        height=600, margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def app_content():
    st.title("📊 Energy Trading Dashboard (Averages)") # Updated title
    st.sidebar.header("Navigation")
    page_options = ["Dashboard", "About"]
    if 'current_page' not in st.session_state: st.session_state.current_page = "Dashboard"
    
    page_key = "nav_radio_authed" 
    page = st.sidebar.radio("Go to", page_options, index=page_options.index(st.session_state.current_page), key=page_key)
    st.session_state.current_page = page
    
    if page == "About": show_about_page()
    else: show_dashboard()

def show_dashboard():
    spacer_left, main_content, spacer_right = st.columns([0.1, 5.8, 0.1]) 

    with main_content:
        available_dates = fetch_available_dates()
        if not available_dates:
            st.error("No available dates. Check database and connection.")
            st.stop()
        
        min_avail_date, max_avail_date = min(available_dates), max(available_dates)
        
        default_start_date = max_avail_date - pd.Timedelta(days=6) if max_avail_date - pd.Timedelta(days=6) >= min_avail_date else min_avail_date
        if 'selected_date_range' not in st.session_state or \
           not (isinstance(st.session_state.selected_date_range, tuple) and len(st.session_state.selected_date_range) == 2) or \
           not (min_avail_date <= st.session_state.selected_date_range[0] <= max_avail_date and \
                min_avail_date <= st.session_state.selected_date_range[1] <= max_avail_date):
            st.session_state.selected_date_range = (default_start_date, max_avail_date)
        
        selected_range_tuple = st.date_input(
            "Select date range", value=st.session_state.selected_date_range,
            min_value=min_avail_date, max_value=max_avail_date, key="date_range_picker"
        )
        
        if isinstance(selected_range_tuple, tuple) and len(selected_range_tuple) == 2:
            start_date_obj, end_date_obj = selected_range_tuple
            st.session_state.selected_date_range = (start_date_obj, end_date_obj)
        else:
            start_date_obj, end_date_obj = st.session_state.selected_date_range # Keep last valid
            st.warning("Please ensure both a start and end date are selected for the range.")
        
        start_date_str = start_date_obj.strftime('%Y-%m-%d')
        end_date_str = end_date_obj.strftime('%Y-%m-%d')
        
        days_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        if 'selected_days_of_week' not in st.session_state:
            st.session_state.selected_days_of_week = days_options
        
        selected_days = st.multiselect(
            "Filter by Day of the Week", options=days_options,
            default=st.session_state.selected_days_of_week, key="day_of_week_filter"
        )
        st.session_state.selected_days_of_week = selected_days if selected_days else days_options
        
        raw_range_data = fetch_data_for_range(start_date_str, end_date_str)
        
        if raw_range_data.empty:
            st.warning(f"No data found for the selected range: {start_date_str} to {end_date_str}.")
            return
        
        day_of_week_map_int = {day_name: i for i, day_name in enumerate(days_options)}
        selected_day_indices = [day_of_week_map_int[day_name] for day_name in st.session_state.selected_days_of_week]
        
        data_for_averaging = raw_range_data[raw_range_data['Date'].dt.dayofweek.isin(selected_day_indices)].copy()
        
        if data_for_averaging.empty:
            st.warning(f"No data found for the selected days of the week ({', '.join(st.session_state.selected_days_of_week)}) within the date range.")
            return

        # --- KPIs ---
        st.subheader(f"Average Daily Summary Metrics (Range: {start_date_obj.strftime('%b %d, %Y')} to {end_date_obj.strftime('%b %d, %Y')} for {', '.join(st.session_state.selected_days_of_week)})")
        col1, col2, col3, col4 = st.columns(4)
        
        daily_grouped = data_for_averaging.groupby(data_for_averaging['Date'].dt.date)

        if "Prices" in data_for_averaging.columns and pd.api.types.is_numeric_dtype(data_for_averaging["Prices"]):
            # Force numerical operations and handle potential NaN values
            avg_daily_max_price = float(daily_grouped['Prices'].max(skipna=True).mean(skipna=True) or 0)
            avg_daily_avg_price = float(daily_grouped['Prices'].mean(skipna=True).mean(skipna=True) or 0)
            avg_daily_min_price = float(daily_grouped['Prices'].min(skipna=True).mean(skipna=True) or 0)
            
            col1.metric("Avg Daily Max Price", f"{avg_daily_max_price:,.2f}" if pd.notna(avg_daily_max_price) and avg_daily_max_price != 0 else "N/A")
            col2.metric("Avg Daily Avg Price", f"{avg_daily_avg_price:,.2f}" if pd.notna(avg_daily_avg_price) and avg_daily_avg_price != 0 else "N/A")
            col3.metric("Avg Daily Min Price", f"{avg_daily_min_price:,.2f}" if pd.notna(avg_daily_min_price) and avg_daily_min_price != 0 else "N/A")
        else:
            [c.metric(label="Price N/A", value="-") for c in [col1, col2, col3]]

        if "Total_MQ" in data_for_averaging.columns and pd.api.types.is_numeric_dtype(data_for_averaging["Total_MQ"]):
            # Force numerical operation and handle potential NaN
            avg_of_daily_max_mq = float(daily_grouped['Total_MQ'].max(skipna=True).mean(skipna=True) or 0)
            col4.metric("Avg Daily Max Total MQ", f"{avg_of_daily_max_mq:,.2f}" if pd.notna(avg_of_daily_max_mq) and avg_of_daily_max_mq != 0 else "N/A", "Avg of Daily Maxes")
        else:
            col4.metric("Avg Daily Max MQ", "N/A", "MQ N/A")

        # --- Data Tables ---
        st.subheader("Data Tables (Averages)")
        tbl_tabs = st.tabs(["Average Hourly Data", "Average of Daily Summaries"])
        with tbl_tabs[0]: # Average Hourly Data
            if 'Hour' in data_for_averaging.columns and not data_for_averaging['Hour'].isnull().all():
                try:
                    # Convert Hour to string first to avoid time object issues
                    data_for_averaging['Hour_Str'] = data_for_averaging['Hour'].apply(
                        lambda x: x.strftime('%H:%M') if pd.notna(x) else 'N/A'
                    )
                    hourly_avg_table_data = data_for_averaging.groupby('Hour_Str')[
                        ["Total_MQ", "Total_BCQ", "Prices", "WESM"]
                    ].mean(skipna=True).reset_index()
                    hourly_avg_table_data.rename(columns={'Hour_Str': 'Time (Avg Across Selected Days)'}, inplace=True)
                    
                    # Ensure all numeric columns are float type
                    for col in ["Total_MQ", "Total_BCQ", "Prices", "WESM"]:
                        if col in hourly_avg_table_data.columns:
                            hourly_avg_table_data[col] = hourly_avg_table_data[col].astype(float)
                    
                    st.dataframe(hourly_avg_table_data.style.format(precision=2, na_rep="N/A"), height=300, use_container_width=True)
                except Exception as e:
                    st.error(f"Error processing hourly average table: {e}")
                    st.dataframe(data_for_averaging[['Hour', "Total_MQ", "Total_BCQ", "Prices", "WESM"]].head(5).style.format(precision=2, na_rep="N/A"), 
                                 height=300, use_container_width=True)
            else:
                st.warning("Hour column not available or all null for hourly average table.")
        
        with tbl_tabs[1]: # Average of Daily Summaries
            s_dict = {}
            for c in ["Total_MQ", "Total_BCQ", "WESM"]:
                if c in data_for_averaging.columns and pd.api.types.is_numeric_dtype(data_for_averaging[c]):
                    try:
                        avg_daily_sum = float(daily_grouped[c].sum(skipna=True).mean(skipna=True) or 0)
                        s_dict[f"Avg Daily Sum {c} (kWh)"] = f"{avg_daily_sum:,.2f}" if pd.notna(avg_daily_sum) and avg_daily_sum != 0 else "N/A"
                    except Exception as e:
                        st.warning(f"Error calculating average daily sum for {c}: {e}")
                        s_dict[f"Avg Daily Sum {c} (kWh)"] = "Error"
                else:
                    s_dict[f"Avg Daily Sum {c} (kWh)"] = "N/A"
            
            if "Prices" in data_for_averaging.columns and pd.api.types.is_numeric_dtype(data_for_averaging["Prices"]):
                try:
                    avg_overall_price = float(daily_grouped["Prices"].mean(skipna=True).mean(skipna=True) or 0)
                    s_dict["Overall Avg Price (PHP/kWh)"] = f"{avg_overall_price:,.2f}" if pd.notna(avg_overall_price) and avg_overall_price != 0 else "N/A"
                except Exception as e:
                    st.warning(f"Error calculating overall average price: {e}")
                    s_dict["Overall Avg Price (PHP/kWh)"] = "Error"
            else:
                s_dict["Overall Avg Price (PHP/kWh)"] = "N/A"

           st.subheader("Average Hourly Metrics Visualization")
            chart_tabs = st.tabs(["Avg MQ & BCQ by Hour", "Avg WESM by Hour", "Avg Price by Hour"])
            
            # Process data for charts
            try:
                if 'Hour' in data_for_averaging.columns and not data_for_averaging['Hour'].isnull().all():
                    data_for_averaging['Hour_Str'] = data_for_averaging['Hour'].apply(
                        lambda x: x.strftime('%H:%M') if pd.notna(x) else 'Unknown'
                    )
                    hourly_avg_df = data_for_averaging.groupby('Hour_Str').agg({
                        'Total_MQ': 'mean', 
                        'Total_BCQ': 'mean',
                        'WESM': 'mean',
                        'Prices': 'mean'
                    }).reset_index()
                    
                    # Sort by hour properly
                    hourly_avg_df['Hour_Sort'] = pd.to_datetime(hourly_avg_df['Hour_Str'], format='%H:%M', errors='coerce')
                    hourly_avg_df = hourly_avg_df.sort_values('Hour_Sort').drop('Hour_Sort', axis=1)
                    
                    # Fill in Tab 1: MQ & BCQ Chart
                    with chart_tabs[0]:
                        if all(c in hourly_avg_df.columns for c in ['Hour_Str', 'Total_MQ', 'Total_BCQ']):
                            # Prepare data for Altair
                            mq_bcq_data = pd.melt(
                                hourly_avg_df, 
                                id_vars=['Hour_Str'], 
                                value_vars=['Total_MQ', 'Total_BCQ'],
                                var_name='Metric', 
                                value_name='Value'
                            )
                            
                            # Create Altair chart for MQ & BCQ
                            mq_bcq_chart = alt.Chart(mq_bcq_data).mark_line(point=True).encode(
                                x=alt.X('Hour_Str:N', title='Hour of Day', sort=None),
                                y=alt.Y('Value:Q', title='Average Energy (kWh)', scale=alt.Scale(zero=False)),
                                color=alt.Color('Metric:N', legend=alt.Legend(title='Metric')),
                                tooltip=['Hour_Str', 'Metric', alt.Tooltip('Value:Q', format=',.2f')]
                            ).properties(
                                title=f'Average Hourly MQ & BCQ ({", ".join(st.session_state.selected_days_of_week)} in selected range)',
                                width=700,
                                height=400
                            ).configure_axis(
                                labelAngle=45
                            )
                            
                            st.altair_chart(mq_bcq_chart, use_container_width=True)
                        else:
                            st.warning("Missing required data columns for MQ & BCQ chart.")
                    
                    # Fill in Tab 2: WESM Chart
                    with chart_tabs[1]:
                        if 'WESM' in hourly_avg_df.columns and not hourly_avg_df['WESM'].isnull().all():
                            # Red for negative values (import), green for positive (export)
                            wesm_chart = alt.Chart(hourly_avg_df).mark_bar().encode(
                                x=alt.X('Hour_Str:N', title='Hour of Day', sort=None),
                                y=alt.Y('WESM:Q', title='Average WESM (kWh)'),
                                color=alt.condition(
                                    alt.datum.WESM > 0,
                                    alt.value('#4CAF50'),  # Green for export
                                    alt.value('#F44336')   # Red for import
                                ),
                                tooltip=[
                                    alt.Tooltip('Hour_Str:N', title='Hour'),
                                    alt.Tooltip('WESM:Q', title='Avg WESM', format=',.2f')
                                ]
                            ).properties(
                                title=f'Average Hourly WESM (+Export/-Import) ({", ".join(st.session_state.selected_days_of_week)} in selected range)',
                                width=700,
                                height=400
                            ).configure_axis(
                                labelAngle=45
                            )
                            
                            st.altair_chart(wesm_chart, use_container_width=True)
                            
                            # Add text explanation of WESM values
                            with st.expander("Understanding WESM Values"):
                                st.markdown("""
                                - **Positive WESM (Green)**: Net Export - You're selling excess energy to the grid
                                - **Negative WESM (Red)**: Net Import - You're buying additional energy from the grid
                                """)
                        else:
                            st.warning("WESM data not available or all null.")
                    
                    # Fill in Tab 3: Price Chart
                    with chart_tabs[2]:
                        if 'Prices' in hourly_avg_df.columns and not hourly_avg_df['Prices'].isnull().all():
                            price_chart = alt.Chart(hourly_avg_df).mark_line(point=True, color='#FF9800').encode(
                                x=alt.X('Hour_Str:N', title='Hour of Day', sort=None),
                                y=alt.Y('Prices:Q', title='Average Price (PHP/kWh)', scale=alt.Scale(zero=False)),
                                tooltip=[
                                    alt.Tooltip('Hour_Str:N', title='Hour'),
                                    alt.Tooltip('Prices:Q', title='Avg Price', format=',.2f')
                                ]
                            ).properties(
                                title=f'Average Hourly Prices ({", ".join(st.session_state.selected_days_of_week)} in selected range)',
                                width=700,
                                height=400
                            ).configure_axis(
                                labelAngle=45
                            )
                            
                            st.altair_chart(price_chart, use_container_width=True)
                        else:
                            st.warning("Price data not available or all null.")
                else:
                    st.warning("Hour data not available for charts.")
            except Exception as e:
                st.error(f"Error creating charts: {e}")
        
        # --- Sankey Diagram (Energy Flow) ---
        st.subheader("Average Energy Flow Visualization")
        
        if 'Hour' in data_for_averaging.columns and not data_for_averaging['Hour'].isnull().all():
            # Get unique hours for selection
            unique_hours = sorted(
                [h.strftime('%H:%M') for h in data_for_averaging['Hour'].dropna().unique() if isinstance(h, time)]
            )
            
            if unique_hours:
                # Default to peak hour (around 14:00) if available, otherwise first hour
                default_hour = '14:00' if '14:00' in unique_hours else unique_hours[0]
                selected_hour = st.selectbox(
                    "Select hour for energy flow visualization:", 
                    options=unique_hours,
                    index=unique_hours.index(default_hour) if default_hour in unique_hours else 0
                )
                
                # Calculate average values for selected hour across the filtered dates
                hour_data = data_for_averaging[data_for_averaging['Hour'].apply(
                    lambda x: x.strftime('%H:%M') if isinstance(x, time) else '' 
                ) == selected_hour]
                
                if not hour_data.empty:
                    avg_mq = float(hour_data['Total_MQ'].mean())
                    avg_wesm = float(hour_data['WESM'].mean())
                    
                    sankey_chart = create_sankey_chart(
                        interval_mq_val=avg_mq,
                        interval_wesm_val_unscaled=avg_wesm,
                        chart_title_date_str=f"{', '.join(st.session_state.selected_days_of_week)} in {start_date_obj.strftime('%b %d')} - {end_date_obj.strftime('%b %d, %Y')}",
                        interval_time_hh_mm_str=selected_hour,
                        start_date_for_fetch=start_date_str,
                        end_date_for_fetch=end_date_str,
                        days_indices_for_fetch=selected_day_indices
                    )
                    
                    if sankey_chart:
                        st.plotly_chart(sankey_chart, use_container_width=True)
                    else:
                        st.info(f"Insufficient data to create energy flow diagram for {selected_hour}.")
                else:
                    st.warning(f"No data available for hour {selected_hour} in the selected range and days.")
            else:
                st.warning("No valid hours available for energy flow visualization.")
        else:
            st.warning("Hour data not available for energy flow visualization.")

def show_about_page():
    """Display the About page content."""
    st.title("About this Dashboard")
    
    st.markdown("""
    ## Energy Trading Dashboard (Averages)
    
    This dashboard provides analysis and visualization of energy trading data, focusing on averages across selected date ranges and days of the week.
    
    ### Key Features:
    
    1. **Data Analysis**
       - Filter by date range and specific days of the week
       - View average metrics across selected timeframes
       - Analyze energy trading patterns
       
    2. **Visualizations**
       - MQ & BCQ average hourly trends
       - WESM import/export patterns
       - Price fluctuations throughout the day
       - Energy flow Sankey diagrams showing generation sources and consumption
       
    3. **Key Terms**
       - **MQ (Metered Quantity)**: Actual energy consumption measured at the meter
       - **BCQ (Bilateral Contract Quantity)**: Energy purchased through bilateral contracts
       - **WESM (Wholesale Electricity Spot Market)**: Philippine electricity market where trading occurs
       - **WESM = BCQ - MQ**: Positive values indicate export, negative indicate import
    
    ### Understanding the Sankey Diagram
    
    The energy flow visualization shows:
    - **Generation Sources**: Energy input from different generators (scaled appropriately)
    - **Junction Node**: Total available energy at the specified hour
    - **Destination/Loads**: Where the energy is consumed
    - **WESM Import/Export**: Energy bought from or sold to the market
    
    ### Data Sources
    
    This dashboard connects to a PostgreSQL database containing:
    - MQ_Hourly: Metered consumption data
    - BCQ_Hourly: Contract purchase data
    - Prices_Hourly: Market price data
    
    ### Note on Averages
    
    All visualizations show averages for the selected date range and days of the week, not actual values for specific dates. This is designed to help identify patterns and typical behavior.
    """)
    
    # Display dashboard version and last update info
    st.sidebar.markdown("---")
    st.sidebar.info("Dashboard Version: 1.2.0  \nLast Updated: May 13, 2025")

# --- MAIN APP EXECUTION ---
if __name__ == "__main__":
    app_content()
