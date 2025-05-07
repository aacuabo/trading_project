import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, date
import altair as alt
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Energy Trading Dashboard")

# --- DATABASE CONFIGURATION ---
@st.cache_resource
def get_sqlalchemy_engine():
    """Establishes and returns a SQLAlchemy database engine using Streamlit secrets."""
    try:
        user = st.secrets["database"]["user"]
        password = st.secrets["database"]["password"]
        host = st.secrets["database"]["host"]
        db = st.secrets["database"]["db"]
        port = int(st.secrets["database"]["port"])
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
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
            LEFT JOIN "BCQ_Hourly" AS bcq ON mq."Date" = bcq."Date" AND mq."Time" = bcq."Time"
            LEFT JOIN "Prices_Hourly" AS p ON mq."Date" = p."Date" AND mq."Time" = p."Time"
            WHERE mq."Date" = %s
            ORDER BY mq."Time";
        """
        df = pd.read_sql(query, engine, params=[(selected_date_str,)])

        if df.empty:
            return pd.DataFrame()

        if 'Time' in df.columns:
            try:
                if not pd.api.types.is_string_dtype(df['Time']):
                    df['Time'] = df['Time'].astype(str)
                df['Time'] = pd.to_datetime(selected_date_str + ' ' + df['Time'].str.strip(), errors='coerce')
                df.dropna(subset=['Time'], inplace=True)
            except Exception as e:
                st.warning(f"Warning converting time values: {e}. Some time data may be incorrect.")

        for col in ["Total_MQ", "Total_BCQ", "Prices"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
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

# Columns to be scaled by 100 (these are the keys from GENERATOR_LONG_TO_SHORT_MAP)
GENERATOR_COLUMNS_TO_SCALE = list(GENERATOR_LONG_TO_SHORT_MAP.keys())

# IMPORTANT: VERIFY and UPDATE the KEYS of this dictionary to your ACTUAL database column names from MQ_Hourly
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
def fetch_sankey_generator_contributions(selected_date_str: str, interval_time_db_format: str):
    """Fetches and scales generator contributions from BCQ_Hourly."""
    contributions = {short_name: 0.0 for short_name in GENERATOR_LONG_TO_SHORT_MAP.values()}
    if not GENERATOR_LONG_TO_SHORT_MAP:
        st.warning("Generator mapping is empty. Cannot fetch contributions.")
        return contributions
    try:
        engine = get_sqlalchemy_engine()
        query_columns_list = [f'"{col_name}"' for col_name in GENERATOR_LONG_TO_SHORT_MAP.keys()]
        query_columns_str = ', '.join(query_columns_list)

        query = f"""
            SELECT {query_columns_str}
            FROM "BCQ_Hourly"
            WHERE "Date" = %(selected_date)s AND "Time" = %(interval_time)s;
        """
        params = {"selected_date": selected_date_str, "interval_time": interval_time_db_format}
        interval_data_df = pd.read_sql(query, engine, params=params)

        if not interval_data_df.empty:
            row = interval_data_df.iloc[0]
            for long_name, short_name in GENERATOR_LONG_TO_SHORT_MAP.items():
                if long_name in row:
                    value = pd.to_numeric(row[long_name], errors='coerce')
                    if pd.notna(value):
                        if long_name in GENERATOR_COLUMNS_TO_SCALE: # Apply scaling
                            value *= 1000
                        contributions[short_name] = value if value > 0 else 0.0
                    else:
                        contributions[short_name] = 0.0
        return contributions
    except Exception as e:
        st.error(f"Error fetching Sankey generator contributions: {e}")
        return contributions

@st.cache_data(ttl=600)
def fetch_sankey_destination_consumption(selected_date_str: str, interval_time_db_format: str):
    """Fetches actual destination consumption from MQ_Hourly."""
    # This function remains unchanged as destination consumptions are not scaled by 100.
    consumption = {short_name: 0.0 for short_name in DESTINATION_LONG_TO_SHORT_MAP.values()}
    if not DESTINATION_LONG_TO_SHORT_MAP:
        st.warning("Destination mapping is empty. Cannot fetch consumption.")
        return consumption
    try:
        engine = get_sqlalchemy_engine()
        query_columns_list = [f'"{col_name}"' for col_name in DESTINATION_LONG_TO_SHORT_MAP.keys()]
        query_columns_str = ', '.join(query_columns_list)

        query = f"""
            SELECT {query_columns_str}
            FROM "MQ_Hourly"
            WHERE "Date" = %(selected_date)s AND "Time" = %(interval_time)s;
        """
        params = {"selected_date": selected_date_str, "interval_time": interval_time_db_format}
        interval_data_df = pd.read_sql(query, engine, params=params)

        if not interval_data_df.empty:
            row = interval_data_df.iloc[0]
            for long_name, short_name in DESTINATION_LONG_TO_SHORT_MAP.items():
                if long_name in row:
                    value = pd.to_numeric(row[long_name], errors='coerce')
                    consumption[short_name] = value if pd.notna(value) and value > 0 else 0.0
        return consumption
    except Exception as e:
        st.error(f"Error fetching Sankey destination consumption: {e}")
        return consumption

def create_sankey_chart(interval_mq_val: float, interval_wesm_val_unscaled: float, selected_date_str: str, interval_time_hh_mm_str: str):
    """
    Creates a professional Sankey diagram showing energy flow for a specific interval.
    Uses a colorblind-friendly palette and clear, concise labeling.
    
    Args:
        interval_mq_val: Total MQ for the interval (unscaled)
        interval_wesm_val_unscaled: (Total_BCQ_unscaled - Total_MQ_unscaled) for the interval
        selected_date_str: Date in string format (YYYY-MM-DD)
        interval_time_hh_mm_str: Time interval in HH:MM format
        
    Returns:
        Plotly Figure object or None if insufficient data
    """
    if pd.isna(interval_mq_val) or interval_mq_val < 0:
        st.info(f"Invalid interval data ({interval_time_hh_mm_str}, {selected_date_str}): MQ = {interval_mq_val:,.0f} kWh")
        return None

    interval_time_db_format = interval_time_hh_mm_str + ":00"  # Assumes DB Time is HH:MM:SS string

    # Fetch generator contributions (scaled by 100x inside the function)
    scaled_generator_contributions = fetch_sankey_generator_contributions(selected_date_str, interval_time_db_format)
    # Fetch destination consumptions (unscaled)
    destination_consumptions = fetch_sankey_destination_consumption(selected_date_str, interval_time_db_format)

    sum_scaled_generator_contributions = sum(v for v in scaled_generator_contributions.values() if pd.notna(v))
    actual_total_mq_for_interval = interval_mq_val  # Unscaled MQ

    # Recalculate WESM balance based on scaled generation and unscaled MQ
    wesm_value_for_sankey = sum_scaled_generator_contributions - actual_total_mq_for_interval

    # Require significant data to generate Sankey
    if sum_scaled_generator_contributions < 0.01 and actual_total_mq_for_interval < 0.01:
        st.info(f"Insufficient flow data for {interval_time_hh_mm_str} on {selected_date_str}")
        return None

    # Initialize Sankey diagram structures
    sankey_node_labels, node_indices = [], {}
    sankey_sources_indices, sankey_targets_indices, sankey_values = [], [], []
    
    # Define a colorblind-friendly palette
    # Based on ColorBrewer and Wong's color schemes for colorblind accessibility
    COLOR_PALETTE = {
        "junction": "#f0e442",         # Gray
        "generator": "#0072B2",        # Blue
        "wesm_import": "#009e73",      # Vermilion (darker orange)
        "load": "#d55e00",             # Green
        "wesm_export": "#CC79A7"       # Reddish purple
    }
    
    node_colors = []
    
    def add_node(label, color):
        if label not in node_indices:
            node_indices[label] = len(sankey_node_labels)
            sankey_node_labels.append(label)
            node_colors.append(color)
        return node_indices[label]

    # Central Junction Node
    total_flow_through_junction = sum_scaled_generator_contributions
    if wesm_value_for_sankey < 0:  # WESM Import
        total_flow_through_junction += abs(wesm_value_for_sankey)
    
    junction_node_label = f"Energy Junction ({total_flow_through_junction:,.0f} kWh)"
    junction_node_idx = add_node(junction_node_label, COLOR_PALETTE["junction"])

    # --- Links TO Junction Node ---
    # 1. Generator Contributions
    for short_name, value in scaled_generator_contributions.items():
        if value > 0.01:  # Only show significant contributions
            percentage = (value / sum_scaled_generator_contributions * 100) if sum_scaled_generator_contributions > 0 else 0
            gen_node_label = f"{short_name} ({value:,.0f} kWh, {percentage:.1f}%)"
            gen_node_idx = add_node(gen_node_label, COLOR_PALETTE["generator"])
            sankey_sources_indices.append(gen_node_idx)
            sankey_targets_indices.append(junction_node_idx)
            sankey_values.append(value)

    # 2. WESM Import
    if wesm_value_for_sankey < 0:
        import_value = abs(wesm_value_for_sankey)
        if import_value > 0.01:
            percentage = (import_value / total_flow_through_junction * 100) if total_flow_through_junction > 0 else 0
            wesm_import_label = f"WESM Import ({import_value:,.0f} kWh, {percentage:.1f}%)"
            wesm_import_node_idx = add_node(wesm_import_label, COLOR_PALETTE["wesm_import"])
            sankey_sources_indices.append(wesm_import_node_idx)
            sankey_targets_indices.append(junction_node_idx)
            sankey_values.append(import_value)

    # --- Links FROM Junction Node ---
    # Individual Local Loads
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
    
    # WESM Export
    if wesm_value_for_sankey > 0:
        export_value = wesm_value_for_sankey
        if export_value > 0.01:
            percentage = (export_value / total_flow_through_junction * 100) if total_flow_through_junction > 0 else 0
            wesm_export_label = f"WESM Export ({export_value:,.0f} kWh, {percentage:.1f}%)"
            wesm_export_node_idx = add_node(wesm_export_label, COLOR_PALETTE["wesm_export"])
            sankey_sources_indices.append(junction_node_idx)
            sankey_targets_indices.append(wesm_export_node_idx)
            sankey_values.append(export_value)
            
    if not sankey_values or sum(sankey_values) < 0.1:
        st.info(f"Insufficient energy flow for {interval_time_hh_mm_str} on {selected_date_str}")
        return None
    
    # Create the Sankey diagram with professional styling
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",  # More professional layout
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
    
    # Professional layout with clear title and proper spacing
    fig.update_layout(
        title=dict(
            text=f"Energy Flow: {interval_time_hh_mm_str}, {selected_date_str}",
            font=dict(size=16, color='#333333')
        ),
        font=dict(family="Arial, sans-serif", size=12, color='#333333'),
        plot_bgcolor='rgba(255,255,255,0)',
        paper_bgcolor='rgba(255,255,255,0)',
        height=600,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# --- STREAMLIT UI (main, show_dashboard, show_about_page functions) ---
# These functions largely remain the same. The key is that show_dashboard()
# passes the unscaled interval_mq_val and unscaled interval_wesm_val to create_sankey_chart.
# create_sankey_chart then handles scaling and recalculation for Sankey logic.

def main():
    st.title("📊 Daily Energy Trading Dashboard")
    st.sidebar.header("Navigation")
    page_options = ["Dashboard", "About"]
    if 'current_page' not in st.session_state: st.session_state.current_page = "Dashboard"
    page = st.sidebar.radio("Go to", page_options, index=page_options.index(st.session_state.current_page), key="nav_radio")
    st.session_state.current_page = page
    if page == "About": show_about_page()
    else: show_dashboard()

def show_dashboard():
    spacer_left, main_content, spacer_right = st.columns([0.1, 5.8, 0.1]) 
    with main_content:
        available_dates = fetch_available_dates()
        if not available_dates:
            st.error("No available dates. Check database and connection."); st.stop()
        
        min_avail_date, max_avail_date = min(available_dates), max(available_dates)
        if 'selected_date' not in st.session_state or \
           not isinstance(st.session_state.selected_date, date) or \
           not (min_avail_date <= st.session_state.selected_date <= max_avail_date):
            st.session_state.selected_date = max_avail_date
        
        selected_date_obj = st.date_input("Select date", value=st.session_state.selected_date, min_value=min_avail_date, max_value=max_avail_date, key="date_picker")
        st.session_state.selected_date = selected_date_obj
        
        selected_date_str = selected_date_obj.strftime('%Y-%m-%d')
        data = fetch_data(selected_date_str)
        
        if data.empty:
            st.warning(f"No data for {selected_date_str}."); return
            
        if all(c in data.columns for c in ["Total_BCQ", "Total_MQ"]) and \
           pd.api.types.is_numeric_dtype(data["Total_BCQ"]) and \
           pd.api.types.is_numeric_dtype(data["Total_MQ"]):
            data['WESM'] = data['Total_BCQ'] - data['Total_MQ'] # Unscaled WESM for general display
        else:
            data['WESM'] = pd.NA
            st.warning("Could not calculate WESM as Total_BCQ or Total_MQ is missing or not numeric.")

        st.subheader("Daily Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        if "Prices" in data.columns and pd.api.types.is_numeric_dtype(data["Prices"]):
            pv = data["Prices"].dropna();
            col1.metric("Max Price", f"{pv.max():,.2f}" if not pv.empty else "N/A")
            col2.metric("Avg Price", f"{pv.mean():,.2f}" if not pv.empty else "N/A")
            col3.metric("Min Price", f"{pv.min():,.2f}" if not pv.empty else "N/A")
        else: [c.warning("Price N/A") for c in [col1, col2, col3]]

        max_mq_val_display, max_mq_time_display = "N/A", "N/A"
        if "Total_MQ" in data.columns and pd.api.types.is_numeric_dtype(data["Total_MQ"]) and not data["Total_MQ"].isnull().all():
            max_mq_val_for_day = data["Total_MQ"].max(skipna=True)
            if pd.notna(max_mq_val_for_day):
                # For Sankey, we use the interval of max MQ.
                # For display here, it's just the max MQ value of the day.
                max_mq_idx_for_day_display = data["Total_MQ"].idxmax(skipna=True)
                t_obj_display = data.loc[max_mq_idx_for_day_display, "Time"]
                t_str_display = t_obj_display.strftime("%H:%M") if pd.notna(t_obj_display) and hasattr(t_obj_display, 'strftime') else "N/A"
                max_mq_val_display = f"{max_mq_val_for_day:,.2f}"
                max_mq_time_display = f"at {t_str_display}"
                col4.metric("Max Total MQ", max_mq_val_display, max_mq_time_display)
            else: col4.info("MQ all NaN.")
        else: col4.warning("MQ N/A")

        st.subheader("Data Tables")
        tbl_tabs = st.tabs(["Hourly Data", "Daily Summary"])
        with tbl_tabs[0]:
            df_display = data.copy(); 
            if 'Time' in df_display and pd.api.types.is_datetime64_any_dtype(df_display['Time']):
                df_display['Time'] = df_display['Time'].dt.strftime('%H:%M')
            st.dataframe(df_display.style.format(precision=2, na_rep="N/A"),height=300, use_container_width=True)
        with tbl_tabs[1]:
            s_dict = {}
            for c in ["Total_MQ", "Total_BCQ", "WESM"]: # These are based on unscaled data from fetch_data
                if c in data and pd.api.types.is_numeric_dtype(data[c]):
                    s_dict[f"{c} (kWh)"] = data[c].sum(skipna=True)
                else:
                    s_dict[f"{c} (kWh)"] = "N/A"
            
            if "Prices" in data and pd.api.types.is_numeric_dtype(data["Prices"]):
                 s_dict["Avg Price (PHP/kWh)"] = data["Prices"].mean(skipna=True) if not data["Prices"].dropna().empty else "N/A"
            else:
                 s_dict["Avg Price (PHP/kWh)"] = "N/A"
            st.dataframe(pd.DataFrame([s_dict]).style.format(precision=2, na_rep="N/A"), use_container_width=True)

        st.subheader("📈 Energy Metrics Over Time (Interactive)")
        if 'Time' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Time']):
            melt_cols = [c for c in ["Total_MQ", "Total_BCQ", "Prices", "WESM"] if c in data and data[c].isnull().sum() < len(data[c])]

            if melt_cols:
                chart_tabs = st.tabs(["Energy & Prices", "WESM Balance"])
                with chart_tabs[0]:
                    ep_c = [c for c in ["Total_MQ", "Total_BCQ", "Prices"] if c in melt_cols]
                    if ep_c:
                        melt_ep = data.melt(id_vars=["Time"], value_vars=ep_c, var_name="Metric", value_name="Value").dropna(subset=['Value'])
                        base_chart = alt.Chart(melt_ep).encode(x=alt.X("Time:T", title="Time", axis=alt.Axis(format="%H:%M")))
                        line_charts = alt.Chart(pd.DataFrame()) 
                        bar_chart = alt.Chart(pd.DataFrame())  

                        energy_metrics = [m for m in ["Total_MQ", "Total_BCQ"] if m in ep_c]
                        if energy_metrics:
                            line_charts = base_chart.transform_filter(
                                alt.FieldOneOfPredicate(field='Metric', oneOf=energy_metrics)
                            ).mark_line(point=True).encode(
                                y=alt.Y("Value:Q", title="Energy (kWh)", scale=alt.Scale(zero=False)), # Changed zero to False for better viz if values are far from zero
                                color=alt.Color("Metric:N", legend=alt.Legend(orient='bottom'), 
                                                scale=alt.Scale(domain=['Total_MQ', 'Total_BCQ'], range=['#FFC20A', '#1A85FF'])),
                                tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", alt.Tooltip("Value:Q", format=",.2f")]
                            )

                        if "Prices" in ep_c:
                            bar_chart = base_chart.transform_filter(
                                alt.datum.Metric == "Prices"
                            ).mark_bar(color="#40B0A6").encode(
                                y=alt.Y("Value:Q", title="Price (PHP/kWh)", scale=alt.Scale(zero=False)), # Changed zero to False
                                tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", alt.Tooltip("Value:Q", format=",.2f")]
                            )
                        
                        if energy_metrics and "Prices" in ep_c:
                            comb_ch = alt.layer(bar_chart, line_charts).resolve_scale(y='independent')
                        elif energy_metrics:
                            comb_ch = line_charts
                        elif "Prices" in ep_c:
                            comb_ch = bar_chart
                        else:
                            comb_ch = alt.Chart(pd.DataFrame()).mark_text(text="No Energy/Price Data for chart.").encode()
                        
                        st.altair_chart(comb_ch.properties(title=f"Metrics for {selected_date_str}").interactive(), use_container_width=True)
                    else: st.info("No MQ, BCQ or Price data for this chart.")
                with chart_tabs[1]: # WESM chart (based on unscaled WESM)
                    if "WESM" in melt_cols:
                        wesm_d = data[["Time", "WESM"]].dropna(subset=["WESM"])
                        if not wesm_d.empty:
                            ch_wesm = alt.Chart(wesm_d).mark_bar().encode(
                                x=alt.X("Time:T", title="Time", axis=alt.Axis(format="%H:%M")), 
                                y=alt.Y("WESM:Q", title="WESM Balance (kWh)"),
                                color=alt.condition(alt.datum.WESM < 0, alt.value("#ff9900"), alt.value("#4c78a8")),
                                tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), alt.Tooltip("WESM:Q", format=",.2f")]
                            ).properties(title=f"WESM Hourly Balance for {selected_date_str} (based on unscaled data)", height=400).interactive()
                            st.altair_chart(ch_wesm, use_container_width=True)
                            wt = wesm_d["WESM"].sum(); 
                            ws = f"Net Import ({abs(wt):,.2f} kWh)" if wt < 0 else (f"Net Export ({wt:,.2f} kWh)" if wt > 0 else "Balanced (0 kWh)")
                            st.info(f"Daily WESM (unscaled): {ws}")
                        else: st.info("No WESM data for chart.")
                    else: st.info("WESM data column N/A or all NaN.")
            else: st.warning(f"Plotting columns missing/null for {selected_date_str}.")
        else: st.warning("Time column invalid for charts.")

        # Sankey Chart Section
        max_mq_interval_time_str_header = ""
        can_gen_sankey = False
        # These are unscaled values for the chosen interval, fetched from the main 'data' DataFrame
        interval_mq_unscaled, interval_bcq_unscaled, interval_wesm_unscaled = 0.0, 0.0, 0.0

        # Find the interval of maximum Total_MQ for the Sankey chart
        if "Total_MQ" in data.columns and not data["Total_MQ"].isnull().all() and \
           "Total_BCQ" in data.columns: # BCQ needed for context, even if it might be all NaN for the interval
            
            max_mq_val_for_day = data["Total_MQ"].max(skipna=True)

            if pd.notna(max_mq_val_for_day):
                sankey_interval_idx = data["Total_MQ"].idxmax(skipna=True)
                sankey_interval_row = data.loc[sankey_interval_idx]

                interval_mq_unscaled = sankey_interval_row["Total_MQ"]
                if pd.isna(interval_mq_unscaled): interval_mq_unscaled = 0.0
                
                # Get BCQ for this specific interval. It might be NaN.
                if "Total_BCQ" in sankey_interval_row and pd.notna(sankey_interval_row["Total_BCQ"]):
                    interval_bcq_unscaled = sankey_interval_row["Total_BCQ"]
                else: # If Total_BCQ is NaN for this specific interval in the main data
                    interval_bcq_unscaled = 0.0 
                    # st.warning(f"Total_BCQ is N/A for the Sankey interval ({sankey_interval_row['Time']:%H:%M}). WESM for Sankey will assume 0 unscaled BCQ for this interval if MQ is also 0.")
                
                interval_wesm_unscaled = interval_bcq_unscaled - interval_mq_unscaled

                time_obj = sankey_interval_row["Time"]
                if pd.notna(time_obj) and hasattr(time_obj, 'strftime'):
                    max_mq_interval_time_str_header = time_obj.strftime("%H:%M")
                    # Condition to generate Sankey: MQ >= 0 and (MQ > 0 or unscaled BCQ > 0 to allow for pure export if MQ is 0)
                    if interval_mq_unscaled >= 0 and (interval_mq_unscaled > 0.001 or interval_bcq_unscaled > 0.001):
                         can_gen_sankey = True
                else: # Fallback for time string
                    max_mq_interval_time_str_header = str(time_obj)
                    can_gen_sankey = True # Attempt anyway
        
        if can_gen_sankey:
            st.subheader(f"⚡ Energy Flow for Interval ({max_mq_interval_time_str_header} on {selected_date_str})")
            # Pass the unscaled MQ and unscaled WESM for that interval
            # create_sankey_chart will handle fetching scaled generator data and recalculating WESM for its internal logic
            sankey_fig = create_sankey_chart(interval_mq_unscaled, interval_wesm_unscaled, selected_date_str, max_mq_interval_time_str_header)
            if sankey_fig: st.plotly_chart(sankey_fig, use_container_width=True)
        else:
            st.subheader("⚡ Energy Flow Sankey Chart")
            st.info("Sankey chart not generated. Conditions for the chosen interval not met (e.g., MQ and unscaled BCQ are zero or unavailable).")


def show_about_page():
    st.header("About This Dashboard")
    st.write("""
    ### Energy Trading Dashboard V1.5
    This dashboard provides visualization and analysis of energy trading data.
    - **Data Source**: Hourly measurements from a PostgreSQL database.
    - **Metrics**: Metered Quantities (MQ), Bilateral Contract Quantities (BCQ), Prices.
    - **WESM Balance**: Calculated as `Total_BCQ - Total_MQ` (using unscaled data for general charts).
    
    #### Sankey Diagram Specifics:
    - **Interval**: Shows energy flow for the hourly interval with the highest `Total_MQ` on the selected day.
    - **Generator Data**: Fetched from `BCQ_Hourly` table. Values for specific generators (FDC, GNPK, PSALM, SEC, TSI, MPI) are **multiplied by 100** for the Sankey display.
    - **Load Data**: Fetched from `MQ_Hourly` table (unscaled).
    - **WESM for Sankey**: Recalculated based on the **scaled** generator total and unscaled MQ for the interval.
    - **Structure**: Simplified to show direct flows from sources (Scaled Generators, WESM Import) via a central "Energy Junction" to sinks (Individual Loads, WESM Export).
    
    ### Features
    - Interactive date selection.
    - Summary metrics and data tables.
    - Time-series charts for energy and prices.
    - Detailed Sankey diagram for the peak MQ interval.
    
    ### WESM Interpretation (General Charts)
    - `WESM = Total_BCQ - Total_MQ` (unscaled).
    - Negative WESM: Net Import. Positive WESM: Net Export.
    
    For issues, contact the system administrator.
    """)
    st.markdown("---"); st.markdown(f"<p style='text-align: center;'>App Version 1.5 (Scaled Gen & Simplified Sankey) | Last Updated: {datetime.now(datetime.now().astimezone().tzinfo).strftime('%Y-%m-%d %H:%M:%S %Z')}</p>", unsafe_allow_html=True)


if __name__ == "__main__": main()
