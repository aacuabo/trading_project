import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, date, time # Added time for type hinting if needed
import altair as alt
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple, Any # Added Any

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
                    contributions[short_name] = avg_value if avg_value > 0 else 0.0
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
                    consumption[short_name] = avg_value if avg_value > 0 else 0.0
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
) -> go.Figure | None: # Added return type hint
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

    sum_scaled_generator_contributions = sum(v for v in scaled_generator_contributions.values() if pd.notna(v))
    actual_total_mq_for_interval = interval_mq_val # This is now an average MQ for the representative interval

    # WESM for Sankey is based on these averaged, possibly scaled, values
    wesm_value_for_sankey = sum_scaled_generator_contributions - actual_total_mq_for_interval

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

    total_flow_through_junction = sum_scaled_generator_contributions
    if wesm_value_for_sankey < 0: # Net import for the Sankey logic
        total_flow_through_junction += abs(wesm_value_for_sankey)
    
    # Title reflects averages
    junction_node_label = f"Avg Max Demand ({total_flow_through_junction:,.0f} kWh)"
    junction_node_idx = add_node(junction_node_label, COLOR_PALETTE["junction"])

    for short_name, value in scaled_generator_contributions.items():
        if value > 0.01:
            percentage = (value / sum_scaled_generator_contributions * 100) if sum_scaled_generator_contributions > 0 else 0
            gen_node_label = f"{short_name} ({value:,.0f} kWh, {percentage:.1f}%)"
            gen_node_idx = add_node(gen_node_label, COLOR_PALETTE["generator"])
            sankey_sources_indices.append(gen_node_idx)
            sankey_targets_indices.append(junction_node_idx)
            sankey_values.append(value)

    if wesm_value_for_sankey < 0: # Net import contributes to junction
        import_value = abs(wesm_value_for_sankey)
        if import_value > 0.01:
            percentage = (import_value / total_flow_through_junction * 100) if total_flow_through_junction > 0 else 0
            wesm_import_label = f"WESM Import ({import_value:,.0f} kWh, {percentage:.1f}%)"
            wesm_import_node_idx = add_node(wesm_import_label, COLOR_PALETTE["wesm_import"])
            sankey_sources_indices.append(wesm_import_node_idx)
            sankey_targets_indices.append(junction_node_idx)
            sankey_values.append(import_value)

    sum_destination_consumptions = sum(v for v in destination_consumptions.values() if pd.notna(v) and v > 0.01)
    if sum_destination_consumptions > 0.01:
        for short_name, value in destination_consumptions.items():
            if value > 0.01:
                percentage = (value / actual_total_mq_for_interval * 100) if actual_total_mq_for_interval > 0 else 0
                dest_node_label = f"{short_name} ({value:,.0f} kWh, {percentage:.1f}%)"
                dest_node_idx = add_node(dest_node_label, COLOR_PALETTE["load"])
                sankey_sources_indices.append(junction_node_idx)
                sankey_targets_indices.append(dest_node_idx)
                sankey_values.append(value)
    
    if wesm_value_for_sankey > 0: # Net export flows from junction
        export_value = wesm_value_for_sankey
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
            avg_daily_max_price = daily_grouped['Prices'].max(skipna=True).mean(skipna=True)
            avg_daily_avg_price = daily_grouped['Prices'].mean(skipna=True).mean(skipna=True)
            avg_daily_min_price = daily_grouped['Prices'].min(skipna=True).mean(skipna=True)
            col1.metric("Avg Daily Max Price", f"{avg_daily_max_price:,.2f}" if pd.notna(avg_daily_max_price) else "N/A")
            col2.metric("Avg Daily Avg Price", f"{avg_daily_avg_price:,.2f}" if pd.notna(avg_daily_avg_price) else "N/A")
            col3.metric("Avg Daily Min Price", f"{avg_daily_min_price:,.2f}" if pd.notna(avg_daily_min_price) else "N/A")
        else:
            [c.metric(label="Price N/A", value="-") for c in [col1, col2, col3]]

        if "Total_MQ" in data_for_averaging.columns and pd.api.types.is_numeric_dtype(data_for_averaging["Total_MQ"]):
            avg_of_daily_max_mq = daily_grouped['Total_MQ'].max(skipna=True).mean(skipna=True)
            col4.metric("Avg Daily Max Total MQ", f"{avg_of_daily_max_mq:,.2f}" if pd.notna(avg_of_daily_max_mq) else "N/A", "Avg of Daily Maxes")
        else:
            col4.metric("Avg Daily Max MQ", "N/A", "MQ N/A")

        # --- Data Tables ---
        st.subheader("Data Tables (Averages)")
        tbl_tabs = st.tabs(["Average Hourly Data", "Average of Daily Summaries"])
        with tbl_tabs[0]: # Average Hourly Data
            if 'Hour' in data_for_averaging.columns and not data_for_averaging['Hour'].isnull().all():
                hourly_avg_table_data = data_for_averaging.groupby(
                    data_for_averaging['Hour'].apply(lambda x: x.strftime('%H:%M') if pd.notna(x) else 'N/A')
                )[["Total_MQ", "Total_BCQ", "Prices", "WESM"]].mean(skipna=True).reset_index()
                hourly_avg_table_data.rename(columns={'Hour': 'Time (Avg Across Selected Days)'}, inplace=True)
                st.dataframe(hourly_avg_table_data.style.format(precision=2, na_rep="N/A"), height=300, use_container_width=True)
            else:
                st.warning("Hour column not available or all null for hourly average table.")
        
        with tbl_tabs[1]: # Average of Daily Summaries
            s_dict = {}
            for c in ["Total_MQ", "Total_BCQ", "WESM"]:
                if c in data_for_averaging.columns and pd.api.types.is_numeric_dtype(data_for_averaging[c]):
                    avg_daily_sum = daily_grouped[c].sum(skipna=True).mean(skipna=True)
                    s_dict[f"Avg Daily Sum {c} (kWh)"] = f"{avg_daily_sum:,.2f}" if pd.notna(avg_daily_sum) else "N/A"
                else:
                    s_dict[f"Avg Daily Sum {c} (kWh)"] = "N/A"
            
            if "Prices" in data_for_averaging.columns and pd.api.types.is_numeric_dtype(data_for_averaging["Prices"]):
                avg_overall_price = daily_grouped["Prices"].mean(skipna=True).mean(skipna=True)
                s_dict["Overall Avg Price (PHP/kWh)"] = f"{avg_overall_price:,.2f}" if pd.notna(avg_overall_price) else "N/A"
            else:
                s_dict["Overall Avg Price (PHP/kWh)"] = "N/A"
            st.dataframe(pd.DataFrame([s_dict]).style.format(precision=2, na_rep="N/A"), use_container_width=True)

        # --- Interactive Charts ---
        st.subheader("📈 Average Hourly Energy Metrics Over Time (Interactive)")
        if 'Hour' in data_for_averaging.columns and not data_for_averaging['Hour'].isnull().all():
            data_for_hourly_chart = data_for_averaging.groupby(data_for_averaging['Hour'])[
                ["Total_MQ", "Total_BCQ", "Prices", "WESM"]
            ].mean(skipna=True).reset_index()
            data_for_hourly_chart['Time_Display'] = data_for_hourly_chart['Hour'].apply(lambda t: datetime.combine(date.min, t) if pd.notna(t) else pd.NaT) # Use date.min for consistent time axis

            chart_melt_cols = [
                c for c in ["Total_MQ", "Total_BCQ", "Prices", "WESM"]
                if c in data_for_hourly_chart.columns and not data_for_hourly_chart[c].isnull().all()
            ]

            if chart_melt_cols:
                chart_tabs = st.tabs(["Avg Energy & Prices", "Avg WESM Balance"])
                title_suffix = f"Avg for {start_date_obj.strftime('%b %d')} to {end_date_obj.strftime('%b %d')} ({','.join(st.session_state.selected_days_of_week)})"
                with chart_tabs[0]:
                    ep_c = [c for c in ["Total_MQ", "Total_BCQ", "Prices"] if c in chart_melt_cols]
                    if ep_c:
                        melt_ep_avg = data_for_hourly_chart.melt(id_vars=["Time_Display"], value_vars=ep_c, var_name="Metric", value_name="Value").dropna(subset=['Value'])
                        base_chart_avg = alt.Chart(melt_ep_avg).encode(x=alt.X("Time_Display:T", title="Time of Day (Averages)", axis=alt.Axis(format="%H:%M")))
                        
                        line_charts_energy_avg = alt.Chart(pd.DataFrame()) # Initialize empty
                        bar_chart_prices_avg = alt.Chart(pd.DataFrame())   # Initialize empty

                        energy_metrics_avg = [m for m in ["Total_MQ", "Total_BCQ"] if m in ep_c]
                        if energy_metrics_avg:
                            line_charts_energy_avg = base_chart_avg.transform_filter(
                                alt.FieldOneOfPredicate(field='Metric', oneOf=energy_metrics_avg)
                            ).mark_line(point=True).encode(
                                y=alt.Y("Value:Q", title="Avg Energy (kWh)", scale=alt.Scale(zero=False)),
                                color=alt.Color("Metric:N", legend=alt.Legend(title="Energy Metrics", orient='bottom'),
                                                scale=alt.Scale(domain=['Total_MQ', 'Total_BCQ'], range=['#FFC20A', '#1A85FF'])),
                                tooltip=[alt.Tooltip("Time_Display:T", format="%H:%M", title="Time"), "Metric:N", alt.Tooltip("Value:Q", format=",.2f", title="Avg Value")]
                            )
                        if "Prices" in ep_c:
                            bar_chart_prices_avg = base_chart_avg.transform_filter(
                                alt.datum.Metric == "Prices"
                            ).mark_bar(color="#40B0A6").encode(
                                y=alt.Y("Value:Q", title="Avg Price (PHP/kWh)", scale=alt.Scale(zero=False)),
                                tooltip=[alt.Tooltip("Time_Display:T", format="%H:%M", title="Time"), "Metric:N", alt.Tooltip("Value:Q", format=",.2f", title="Avg Value")]
                            )
                        
                        if energy_metrics_avg and "Prices" in ep_c: comb_ch_avg = alt.layer(bar_chart_prices_avg, line_charts_energy_avg).resolve_scale(y='independent')
                        elif energy_metrics_avg: comb_ch_avg = line_charts_energy_avg
                        elif "Prices" in ep_c: comb_ch_avg = bar_chart_prices_avg
                        else: comb_ch_avg = alt.Chart(pd.DataFrame()).mark_text(text="No Avg Energy/Price Data selected for chart.").encode()
                        
                        st.altair_chart(comb_ch_avg.properties(title=f"Avg Hourly Energy & Prices - {title_suffix}").interactive(), use_container_width=True)
                    else: st.info("No Avg MQ, BCQ or Price data available for this chart.")

                with chart_tabs[1]:
                    wesm_available_avg = "WESM" in chart_melt_cols
                    prices_available_avg = "Prices" in chart_melt_cols # For overlay
                    charts_to_layer_avg = []

                    if wesm_available_avg:
                        wesm_d_avg = data_for_hourly_chart[["Time_Display", "WESM"]].dropna(subset=["WESM"])
                        if not wesm_d_avg.empty:
                            ch_wesm_avg = alt.Chart(wesm_d_avg).mark_bar().encode(
                                x=alt.X("Time_Display:T", title="Time of Day (Averages)", axis=alt.Axis(format="%H:%M")),
                                y=alt.Y("WESM:Q", title="Avg WESM Balance (kWh)", scale=alt.Scale(zero=True)), # WESM can be positive/negative, zero=True is good
                                color=alt.condition(alt.datum.WESM < 0, alt.value("#ff9900"), alt.value("#4c78a8")),
                                tooltip=[alt.Tooltip("Time_Display:T", format="%H:%M", title="Time"), alt.Tooltip("WESM:Q", format=",.2f", title="Avg WESM")]
                            )
                            charts_to_layer_avg.append(ch_wesm_avg)
                            # Display average of daily WESM sums
                            if not daily_grouped['WESM'].sum(skipna=True).empty:
                                avg_daily_wesm_sum = daily_grouped['WESM'].sum(skipna=True).mean(skipna=True)
                                ws_suffix = f"Net Import ({abs(avg_daily_wesm_sum):,.2f} kWh)" if avg_daily_wesm_sum < 0 else (f"Net Export ({avg_daily_wesm_sum:,.2f} kWh)" if avg_daily_wesm_sum > 0 else "Balanced (0 kWh)")
                                st.info(f"Avg Daily WESM Sum: {ws_suffix}")
                            else: st.info("WESM sum cannot be calculated.")
                        else: st.info("No Avg WESM data for chart.")
                    
                    if prices_available_avg: # For overlaying price line
                        prices_d_avg = data_for_hourly_chart[["Time_Display", "Prices"]].dropna(subset=["Prices"])
                        if not prices_d_avg.empty:
                            ch_prices_line_avg = alt.Chart(prices_d_avg).mark_line(point=True, color="#E45756", strokeDash=[3,3]).encode(
                                x=alt.X("Time_Display:T"), # No title, shares with WESM chart
                                y=alt.Y("Prices:Q", title="Avg Price (PHP/kWh)", scale=alt.Scale(zero=False)),
                                tooltip=[alt.Tooltip("Time_Display:T", format="%H:%M", title="Time"), alt.Tooltip("Prices:Q", format=",.2f", title="Avg Price")]
                            )
                            charts_to_layer_avg.append(ch_prices_line_avg)
                        # No st.info here, as it's an overlay
                    
                    if len(charts_to_layer_avg) >= 1 : # Updated to allow single chart too
                        final_wesm_chart = alt.layer(*charts_to_layer_avg).resolve_scale(y='independent') if len(charts_to_layer_avg) > 1 else charts_to_layer_avg[0]
                        st.altair_chart(final_wesm_chart.properties(title=f"Avg Hourly WESM Balance (& Prices) - {title_suffix}", height=400).interactive(), use_container_width=True)
                    else:
                        st.info("No Avg WESM or Price data available for this tab.")
            else:
                st.warning(f"No plottable columns with averaged data for the selected period.")
        else:
            st.warning("Hour column not available or all null for preparing averaged charts.")

        # --- Sankey Diagram ---
        representative_peak_interval_time_str = "N/A"
        avg_peak_interval_mq = 0.0
        avg_peak_interval_bcq = 0.0
        can_gen_sankey = False

        if 'Hour' in data_for_averaging.columns and not data_for_averaging['Hour'].isnull().all() and \
           'Total_MQ' in data_for_averaging.columns and not data_for_averaging['Total_MQ'].isnull().all():
            
            avg_hourly_mq_summary = data_for_averaging.groupby(data_for_averaging['Hour'])['Total_MQ'].mean(skipna=True)
            if not avg_hourly_mq_summary.empty:
                peak_avg_mq_hour_obj = avg_hourly_mq_summary.idxmax() # datetime.time object
                avg_peak_interval_mq = avg_hourly_mq_summary.loc[peak_avg_mq_hour_obj]

                if 'Total_BCQ' in data_for_averaging.columns: # Ensure BCQ is available for WESM calculation
                    avg_hourly_bcq_summary = data_for_averaging.groupby(data_for_averaging['Hour'])['Total_BCQ'].mean(skipna=True)
                    if peak_avg_mq_hour_obj in avg_hourly_bcq_summary:
                         avg_peak_interval_bcq = avg_hourly_bcq_summary.loc[peak_avg_mq_hour_obj]
                    # else avg_peak_interval_bcq remains 0.0

                if pd.notna(avg_peak_interval_mq) and avg_peak_interval_mq > 0.001: # Ensure some flow
                    representative_peak_interval_time_str = peak_avg_mq_hour_obj.strftime("%H:%M")
                    can_gen_sankey = True
        
        if can_gen_sankey:
            st.subheader(f"⚡ Average Energy Flow (Representative Peak Avg MQ Interval: {representative_peak_interval_time_str} for selected days)")
            avg_peak_interval_wesm = avg_peak_interval_bcq - avg_peak_interval_mq
            
            sankey_chart_title = f"Avg for {start_date_obj.strftime('%b %d')} to {end_date_obj.strftime('%b %d')} ({','.join(st.session_state.selected_days_of_week)})"
            sankey_fig = create_sankey_chart(
                interval_mq_val=avg_peak_interval_mq,
                interval_wesm_val_unscaled=avg_peak_interval_wesm,
                chart_title_date_str=sankey_chart_title,
                interval_time_hh_mm_str=representative_peak_interval_time_str,
                start_date_for_fetch=start_date_str,
                end_date_for_fetch=end_date_str,
                days_indices_for_fetch=selected_day_indices
            )
            if sankey_fig: st.plotly_chart(sankey_fig, use_container_width=True)
        else:
            st.subheader("⚡ Average Energy Flow Sankey Chart")
            st.info("Average Sankey chart not generated. Conditions for a representative peak interval not met (e.g., Avg Max MQ is zero or negative, or essential data is unavailable).")


def show_about_page():
    st.header("About This Dashboard")
    st.write("""
    ### Energy Trading Dashboard V1.8 (Averaged Data)
    This dashboard provides visualization and analysis of **averaged** energy trading data over a selected date range and filtered by day(s) of the week.
    - **Data Source**: Hourly measurements from a PostgreSQL database.
    - **Metrics**: Metered Quantities (MQ), Bilateral Contract Quantities (BCQ), Prices. All displayed metrics are averages based on user selections.
    - **WESM Balance**: Calculated as `Total_BCQ - Total_MQ`. For charts and tables showing hourly averages, WESM is the average of hourly WESM values.
    
    #### Sankey Diagram Specifics (Averaged Flow):
    - **Interval**: Shows average energy flow for the hourly interval that has the **highest average `Total_MQ`** across the selected dates and days of the week.
    - **Generator Data**: Averaged contributions from `BCQ_Hourly` for the representative interval. Values for specific generators are scaled.
    - **Load Data**: Averaged consumption from `MQ_Hourly` for the representative interval (unscaled).
    - **WESM for Sankey**: Recalculated based on the averaged scaled generator total and averaged unscaled MQ for the representative interval.
    - **Structure**: Simplified to show direct flows from sources (Avg Scaled Generators, Avg WESM Import) via a central "Avg Max Demand" node to sinks (Avg Individual Loads, Avg WESM Export).
    
    ### Features
    - Secure passcode access (passcode configured in `secrets.toml`).
    - Interactive date range selection and day-of-the-week filtering.
    - Averaged summary metrics and data tables.
    - Time-series charts for average hourly energy and prices.
    - Detailed Sankey diagram for the representative average peak MQ interval.
    
    For issues, contact the system administrator.
    """)
    st.markdown("---"); st.markdown(f"<p style='text-align: center;'>App Version 1.8 | Last Updated: {datetime.now(datetime.now().astimezone().tzinfo).strftime('%Y-%m-%d %H:%M:%S %Z')}</p>", unsafe_allow_html=True)

def main():
    """Main function to handle passcode and app display."""
    try:
        CORRECT_PASSCODE = st.secrets["app_settings"]["passcode"]
    except KeyError:
        st.error("Passcode not found in secrets.toml. Please ensure `[app_settings]` section with `passcode` key is configured.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading passcode from secrets: {e}")
        st.stop()

    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 Secure Access")
        # Ensure unique key for text_input if this part can rerun before authentication
        password_attempt = st.text_input("Enter Passcode:", type="password", key="passcode_input_main_v2")

        if password_attempt: # Check if something was entered
            if password_attempt == CORRECT_PASSCODE:
                st.session_state.authenticated = True
                # To clear the password input field, we can't directly empty text_input after it's rendered.
                # A rerun will naturally clear it if the widget isn't re-rendered with the same value.
                st.rerun() # Rerun to proceed to app_content
            else:
                st.error("Incorrect passcode. Please try again.")
        # No explicit submit button needed as text_input triggers rerun on change if not empty.
    
    if st.session_state.authenticated:
        app_content()
        
# Removed the general st.rerun button as page navigation and input widgets handle reruns.
# If you need a manual refresh for other reasons, you can add it back.
# if st.button("Refresh Data/Rerun App"):
#    st.rerun()
    
if __name__ == "__main__":
    main()
