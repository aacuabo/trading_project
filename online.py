import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, date # date is not explicitly used from datetime, but good to have
import altair as alt
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple # Tuple not used here, but good practice

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
        # Test connection
        with engine.connect() as conn:
            pass  # Just testing if connection works
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
    "FDC Misamis Power Corporation (FDC)": 'FDC',
    "GNPower Kauswagan Ltd. Co. (GNPKLCO)": 'GNPK',
    "Power Sector Assets & Liabilities Management Corporation (PSALMGMIN)": 'PSALM',
    "Sarangani Energy Corporation (SEC)": 'SEC',
    "Therma South, Inc. (TSI)": 'TSI',
    "Malita Power Inc. (SMCPC)": 'MPI'
}

DESTINATION_LONG_TO_SHORT_MAP = {
    "14BGN_T1L1_KIDCOTE01_NET": 'M1/M6/M8', "14BGN_T1L1_KIDCOTE02_NET": 'M2',
    "14BGN_T1L1_KIDCOTE03_NET": 'M3', "14BGN_T1L1_KIDCOTE04_NET": 'M4',
    "14BGN_T2L1_KIDCOTE05_NET": 'M5', "14BGN_T1L1_KIDCOTE08_NET": 'M7',
    "14BGN_T1L1_KIDCOTE10_NET": 'M9', "14BGN_T1L1_KIDCSCV01_DEL": 'KIDCSCV01_DEL',
    "14BGN_T1L1_KIDCSCV02_DEL": 'KIDCSCV02_DEL'
}

@st.cache_data(ttl=600)
def fetch_sankey_generator_contributions(selected_date_str: str, interval_time_db_format: str):
    """Fetches actual generator contributions from BCQ_Hourly for a specific date and time interval."""
    contributions = {short_name: 0.0 for short_name in GENERATOR_LONG_TO_SHORT_MAP.values()}
    if not GENERATOR_LONG_TO_SHORT_MAP:
        st.warning("Generator mapping is empty. Cannot fetch contributions.")
        return contributions
    try:
        engine = get_sqlalchemy_engine()
        # Keys of GENERATOR_LONG_TO_SHORT_MAP are the exact column names in BCQ_Hourly
        query_columns_str = ', '.join([f'"{col_name}"' for col_name in GENERATOR_LONG_TO_SHORT_MAP.keys()])

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
                if long_name in row: # Check if column was successfully fetched
                    value = pd.to_numeric(row[long_name], errors='coerce')
                    contributions[short_name] = value if pd.notna(value) and value > 0 else 0.0
        # else:
            # No data for this specific interval, contributions remain zero.
            # st.info(f"No generator contribution data found in BCQ_Hourly for {selected_date_str} at {interval_time_db_format}.")
        return contributions
    except Exception as e:
        st.error(f"Error fetching Sankey generator contributions: {e}")
        return contributions # Return initialized (zero) contributions on error

@st.cache_data(ttl=600)
def fetch_sankey_destination_consumption(selected_date_str: str, interval_time_db_format: str):
    """Fetches actual destination consumption from MQ_Hourly for a specific date and time interval."""
    consumption = {short_name: 0.0 for short_name in DESTINATION_LONG_TO_SHORT_MAP.values()}
    if not DESTINATION_LONG_TO_SHORT_MAP:
        st.warning("Destination mapping is empty. Cannot fetch consumption.")
        return consumption
    try:
        engine = get_sqlalchemy_engine()
        # Keys of DESTINATION_LONG_TO_SHORT_MAP are the exact column names in MQ_Hourly
        query_columns_str = ', '.join([f'"{col_name}"' for col_name in DESTINATION_LONG_TO_SHORT_MAP.keys()])

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
                if long_name in row: # Check if column was successfully fetched
                    value = pd.to_numeric(row[long_name], errors='coerce')
                    consumption[short_name] = value if pd.notna(value) and value > 0 else 0.0
        # else:
            # No data for this specific interval, consumption remains zero.
            # st.info(f"No destination consumption data found in MQ_Hourly for {selected_date_str} at {interval_time_db_format}.")
        return consumption
    except Exception as e:
        st.error(f"Error fetching Sankey destination consumption: {e}")
        return consumption # Return initialized (zero) consumption on error


def create_sankey_chart(interval_mq_val: float, interval_wesm_val: float, selected_date_str: str, interval_time_hh_mm_str: str):
    # interval_mq_val is Total_MQ for the interval
    # interval_wesm_val is (Total_BCQ - Total_MQ) for the interval
    # interval_time_hh_mm_str is in "HH:MM" format for display and title

    if pd.isna(interval_mq_val) or interval_mq_val < 0: # MQ should not be negative
        st.info(f"Max Interval MQ is invalid ({interval_mq_val:,.0f} kWh) for {interval_time_hh_mm_str} on {selected_date_str}. Sankey not generated.")
        return None
    # Allow generation even if MQ is zero, if WESM export exists
    if interval_mq_val == 0 and (pd.isna(interval_wesm_val) or interval_wesm_val <= 0): # Allow if WESM export > 0
         if not (interval_wesm_val > 0) : # only show info if not WESM export
            st.info(f"Max Interval MQ is zero and no significant WESM activity for {interval_time_hh_mm_str} on {selected_date_str}. Sankey not generated.")
            return None
            
    if pd.isna(interval_wesm_val): # Default WESM to 0 if it's NaN for the interval
        interval_wesm_val = 0

    # This is Total_BCQ for the interval: Total_MQ + (Total_BCQ - Total_MQ)
    actual_total_bcq_for_interval = interval_mq_val + interval_wesm_val
    actual_total_mq_for_interval = interval_mq_val

    # Assuming DB Time column is HH:MM:SS string format. interval_time_hh_mm_str is HH:MM.
    interval_time_db_format = interval_time_hh_mm_str + ":00"

    generator_contributions = fetch_sankey_generator_contributions(selected_date_str, interval_time_db_format)
    destination_consumptions = fetch_sankey_destination_consumption(selected_date_str, interval_time_db_format)
    
    # Verify fetched data sum (optional, for debugging or stricter checks)
    # sum_fetched_bcq = sum(v for v in generator_contributions.values() if pd.notna(v))
    # sum_fetched_mq = sum(v for v in destination_consumptions.values() if pd.notna(v))
    # if not np.isclose(sum_fetched_bcq, actual_total_bcq_for_interval):
    #     st.warning(f"Sum of fetched generator contributions ({sum_fetched_bcq:,.0f}) does not match calculated Total BCQ ({actual_total_bcq_for_interval:,.0f}). Sankey might be based on partial data.")
    # if not np.isclose(sum_fetched_mq, actual_total_mq_for_interval):
    #     st.warning(f"Sum of fetched destination consumptions ({sum_fetched_mq:,.0f}) does not match Total MQ ({actual_total_mq_for_interval:,.0f}). Sankey might be based on partial data.")


    sankey_node_labels, node_indices, sankey_sources_indices, sankey_targets_indices, sankey_values, node_colors = [], {}, [], [], [], []
    
    def add_node(label, color="grey"):
        if label not in node_indices:
            node_indices[label] = len(sankey_node_labels)
            sankey_node_labels.append(label)
            node_colors.append(color)
        return node_indices[label]

    # The middle node represents the total energy that is either generated locally or imported,
    # and then distributed to local consumption or exported.
    # Its "value" isn't a single measure but a hub.
    # Let's use Total BCQ as a reference for the "energy pool"
    middle_node_label = f"Energy Hub ({actual_total_bcq_for_interval:,.0f} kWh Total Supply)"
    if actual_total_bcq_for_interval < 0.01 and actual_total_mq_for_interval > 0.01: # e.g. all import and MQ
         middle_node_label = f"Energy Hub ({actual_total_mq_for_interval:,.0f} kWh Total Demand)"


    # Denominator for percentages: Use Total BCQ for sources, Total MQ for sinks from local.
    # WESM percentages relative to MQ as per original logic, or could be BCQ.
    # For simplicity, let's stick to MQ as the primary reference for percentages in node labels as before.
    denominator_for_percentages = actual_total_mq_for_interval if actual_total_mq_for_interval > 0.001 else 1.0

    # --- SOURCES to Middle Node ---
    # 1. Generators (contributing to BCQ)
    total_gen_contribution_value = 0
    for short_name, value in generator_contributions.items():
        if value > 0.01: # Only show significant contributions
            percentage_of_total_bcq = (value / actual_total_bcq_for_interval * 100) if actual_total_bcq_for_interval > 0 else 0
            gen_node_label = f"{short_name} ({value:,.0f} kWh, {percentage_of_total_bcq:.1f}% of Gen)"
            gen_node_idx = add_node(gen_node_label, "blue")
            # Flow from generator to a conceptual "Total Generation / BCQ Pool" node
            # For now, let's assume generators flow towards the main "Hub" which will then distribute
            # The value of this flow is the generator's output.
            # sankey_sources_indices.append(gen_node_idx); sankey_targets_indices.append(middle_node_idx); sankey_values.append(value)
            total_gen_contribution_value += value
    
    # Create a single "Local Generation" node if individual generators are too many or for aggregation
    if total_gen_contribution_value > 0.01:
        gen_pool_label = f"Local Generation ({total_gen_contribution_value:,.0f} kWh)"
        gen_pool_idx = add_node(gen_pool_label, "darkblue") # Unified generation node
        # Add links from individual generators to this pool if you want that level of detail,
        # or just use this pool node as the source to the hub.
        # For now, just represent total local generation as one source to the hub.
        # The actual total_bcq_for_interval is the sum of this and WESM import.
    else: # if no local generation, gen_pool_idx might not be created
        gen_pool_idx = -1 # Placeholder

    # Middle Node (Hub)
    # Label redefined to reflect its role better based on actual flow
    hub_node_val_display = actual_total_bcq_for_interval # Default to total supply
    if interval_wesm_val < 0 : # Net import situation
        hub_node_val_display = actual_total_mq_for_interval + abs(interval_wesm_val) # MQ met by local + import
    elif interval_wesm_val > 0: # Net export situation
        hub_node_val_display = actual_total_mq_for_interval + interval_wesm_val # MQ + Export = BCQ

    # Let the middle node be the "Distribution Hub / Contractual Supply"
    # Flows into this hub are from Local Generation and WESM Import.
    # Flows out are to Local Demand (MQ) and WESM Export.
    # The value of the hub itself is Total BCQ.
    middle_node_label = f"Distribution Hub ({actual_total_bcq_for_interval:,.0f} kWh)"
    middle_node_idx = add_node(middle_node_label, "orange")

    # Link from Aggregated Local Generation to Hub
    if total_gen_contribution_value > 0.01 and gen_pool_idx != -1:
         # Individual generator contributions (actual values from BCQ_Hourly)
        for short_name, value in generator_contributions.items():
            if value > 0.01:
                percentage_of_total_bcq = (value / actual_total_bcq_for_interval * 100) if actual_total_bcq_for_interval > 0 else 0
                gen_node_label = f"Gen: {short_name} ({value:,.0f} kWh, {percentage_of_total_bcq:.1f}%)"
                gen_node_idx = add_node(gen_node_label, "blue")
                sankey_sources_indices.append(gen_node_idx)
                sankey_targets_indices.append(middle_node_idx)
                sankey_values.append(value)


    # 2. WESM Import (if WESM < 0, meaning BCQ < MQ, implies WESM is a source to meet MQ needs beyond BCQ)
    # Corrected: if WESM < 0 (BCQ - MQ < 0  => BCQ < MQ). Energy is imported.
    # This import contributes to meeting the MQ.
    # The actual_total_bcq_for_interval = MQ + WESM. If WESM is negative (import), then BCQ = MQ - |WESM_import|.
    # This means local generation (BCQ) was less than MQ, and WESM import filled the gap.
    # The "Distribution Hub" value is `actual_total_bcq_for_interval`.
    # WESM Import flows INTO this hub.
    if interval_wesm_val < 0: # Net Import (BCQ < MQ)
        import_value = abs(interval_wesm_val) # This is MQ - BCQ
        if import_value > 0.01:
            percentage = (import_value / (actual_total_bcq_for_interval + import_value)) * 100 if (actual_total_bcq_for_interval + import_value) > 0 else 0 # % of total sources to hub
            wesm_label = f"WESM Import ({import_value:,.0f} kWh, {percentage:.1f}%)"
            wesm_node_idx = add_node(wesm_label, "red")
            sankey_sources_indices.append(wesm_node_idx)
            sankey_targets_indices.append(middle_node_idx) # Import flows to the hub
            sankey_values.append(import_value)

    # --- SINKS from Middle Node ---
    # 1. Local Demand (MQ components)
    # The sum of destination_consumptions is actual_total_mq_for_interval
    if actual_total_mq_for_interval > 0.01 :
        # Optionally, create an aggregated "Local Demand (MQ)" node
        local_demand_node_label = f"Local Demand (MQ) ({actual_total_mq_for_interval:,.0f} kWh)"
        local_demand_node_idx = add_node(local_demand_node_label, "lightgreen")
        sankey_sources_indices.append(middle_node_idx) # Hub supplies local demand
        sankey_targets_indices.append(local_demand_node_idx)
        sankey_values.append(actual_total_mq_for_interval)

        for short_name, value in destination_consumptions.items():
            if value > 0.01: # Only show significant consumption
                percentage_of_total_mq = (value / actual_total_mq_for_interval * 100) if actual_total_mq_for_interval > 0 else 0
                dest_node_label = f"Load: {short_name} ({value:,.0f} kWh, {percentage_of_total_mq:.1f}%)"
                dest_node_idx = add_node(dest_node_label, "green")
                sankey_sources_indices.append(local_demand_node_idx) # Local demand aggregate supplies individual loads
                sankey_targets_indices.append(dest_node_idx)
                sankey_values.append(value)
    
    # 2. WESM Export (if WESM > 0, meaning BCQ > MQ, excess BCQ is exported)
    if interval_wesm_val > 0: # Net Export (BCQ > MQ)
        export_value = interval_wesm_val
        if export_value > 0.01:
            percentage = (export_value / actual_total_bcq_for_interval) * 100 if actual_total_bcq_for_interval > 0 else 0 # % of total BCQ that is exported
            wesm_label = f"WESM Export ({export_value:,.0f} kWh, {percentage:.1f}%)"
            wesm_node_idx = add_node(wesm_label, "purple")
            sankey_sources_indices.append(middle_node_idx) # Hub exports excess
            sankey_targets_indices.append(wesm_node_idx)
            sankey_values.append(export_value)
            
    if not sankey_values or sum(sankey_values) < 0.1:
        st.info(f"Not enough significant flow data for Max MQ interval ({interval_time_hh_mm_str}) on {selected_date_str} to draw Sankey chart. Sum of values: {sum(sankey_values):.2f}")
        return None
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=25, thickness=20, line=dict(color="black", width=0.5), label=sankey_node_labels, color=node_colors),
        link=dict(source=sankey_sources_indices, target=sankey_targets_indices, value=sankey_values,
                  # Add hover text for links
                  hovertemplate='Flow from %{source.label} to %{target.label}: %{value:,.0f} kWh<extra></extra>') 
    )])
    fig.update_layout(title_text=f"Energy Flow for Interval ({interval_time_hh_mm_str}) on {selected_date_str}", font_size=10, height=700) # Increased height
    return fig

# --- STREAMLIT UI ---
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
    spacer_left, main_content, spacer_right = st.columns([0.2, 5, 0.2]) # Wider main content
    with main_content:
        available_dates = fetch_available_dates()
        if not available_dates:
            st.error("No available dates. Check database and connection."); st.stop()
        
        min_avail_date, max_avail_date = min(available_dates), max(available_dates)
        # Ensure st.session_state.selected_date is a Python date object for comparison
        if 'selected_date' not in st.session_state or \
           not isinstance(st.session_state.selected_date, date) or \
           not (min_avail_date <= st.session_state.selected_date <= max_avail_date):
            st.session_state.selected_date = max_avail_date
        
        selected_date_obj = st.date_input("Select date", value=st.session_state.selected_date, min_value=min_avail_date, max_value=max_avail_date, key="date_picker")
        st.session_state.selected_date = selected_date_obj # selected_date_obj is already a date object
        
        selected_date_str = selected_date_obj.strftime('%Y-%m-%d')
        data = fetch_data(selected_date_str)
        
        if data.empty:
            st.warning(f"No data for {selected_date_str}."); return
            
        if all(c in data.columns for c in ["Total_BCQ", "Total_MQ"]) and \
           pd.api.types.is_numeric_dtype(data["Total_BCQ"]) and \
           pd.api.types.is_numeric_dtype(data["Total_MQ"]):
            data['WESM'] = data['Total_BCQ'] - data['Total_MQ']
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
            max_mq_val = data["Total_MQ"].max(skipna=True)
            if pd.notna(max_mq_val):
                max_mq_idx = data["Total_MQ"].idxmax(skipna=True)
                t_obj = data.loc[max_mq_idx, "Time"]
                t_str = t_obj.strftime("%H:%M") if pd.notna(t_obj) and hasattr(t_obj, 'strftime') else "N/A"
                max_mq_val_display = f"{max_mq_val:,.2f}"
                max_mq_time_display = f"at {t_str}"
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
            for c in ["Total_MQ", "Total_BCQ", "WESM"]:
                if c in data and pd.api.types.is_numeric_dtype(data[c]):
                    s_dict[f"{c} (kWh)"] = data[c].sum(skipna=True)
                else:
                    s_dict[f"{c} (kWh)"] = "N/A"
            
            if "Prices" in data and pd.api.types.is_numeric_dtype(data["Prices"]):
                 s_dict["Avg Price (PHP/kWh)"] = data["Prices"].mean(skipna=True)
            else:
                 s_dict["Avg Price (PHP/kWh)"] = "N/A"
            st.dataframe(pd.DataFrame([s_dict]).style.format(precision=2, na_rep="N/A"), use_container_width=True)


        st.subheader("📈 Energy Metrics Over Time (Interactive)")
        if 'Time' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Time']):
            melt_cols = [c for c in ["Total_MQ", "Total_BCQ", "Prices", "WESM"] if c in data and not data[c].isnull().all()]
            if melt_cols:
                chart_tabs = st.tabs(["Energy & Prices", "WESM Balance"])
                with chart_tabs[0]:
                    ep_c = [c for c in ["Total_MQ", "Total_BCQ", "Prices"] if c in melt_cols]
                    if ep_c:
                        melt_ep = data.melt(id_vars=["Time"], value_vars=ep_c, var_name="Metric", value_name="Value").dropna(subset=['Value'])
                        ch_en = alt.Chart(pd.DataFrame()).mark_text() # Empty chart
                        if any(m in ep_c for m in ["Total_MQ", "Total_BCQ"]):
                            ch_en = alt.Chart(melt_ep[melt_ep["Metric"].isin(["Total_MQ", "Total_BCQ"])]).mark_line(point=True).encode(
                                x=alt.X("Time:T", title="Time", axis=alt.Axis(format="%H:%M")), 
                                y=alt.Y("Value:Q", title="Energy (kWh)", scale=alt.Scale(zero=True)),
                                color=alt.Color("Metric:N", legend=alt.Legend(orient='bottom'), scale=alt.Scale(domain=['Total_MQ', 'Total_BCQ'], range=['#FFC20A', '#1A85FF'])),
                                tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", alt.Tooltip("Value:Q", format=",.2f")]
                            ).properties(title="Energy Metrics")
                        ch_pr = alt.Chart(pd.DataFrame()).mark_text() # Empty chart
                        if "Prices" in ep_c:
                            ch_pr = alt.Chart(melt_ep[melt_ep["Metric"] == "Prices"]).mark_bar(color="#40B0A6").encode(
                                x=alt.X("Time:T", title="Time", axis=alt.Axis(format="%H:%M")), 
                                y=alt.Y("Value:Q", title="Price (PHP/kWh)", scale=alt.Scale(zero=True)),
                                tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", alt.Tooltip("Value:Q", format=",.2f")]
                            ).properties(title="Prices")
                        
                        # Combine charts
                        if ch_en.data is not alt.Undefined and ch_pr.data is not alt.Undefined and \
                           any(m in ep_c for m in ["Total_MQ", "Total_BCQ"]) and "Prices" in ep_c :
                            comb_ch = alt.layer(ch_pr, ch_en).resolve_scale(y='independent')
                        elif ch_en.data is not alt.Undefined and any(m in ep_c for m in ["Total_MQ", "Total_BCQ"]): comb_ch = ch_en
                        elif ch_pr.data is not alt.Undefined and "Prices" in ep_c: comb_ch = ch_pr
                        else: comb_ch = alt.Chart(pd.DataFrame()).mark_text(text="No Energy/Price Data for chart.").encode()
                        
                        st.altair_chart(comb_ch.properties(title=f"Metrics for {selected_date_str}").interactive(), use_container_width=True)
                    else: st.info("No MQ, BCQ or Price data for this chart.")
                with chart_tabs[1]:
                    if "WESM" in melt_cols:
                        wesm_d = data[["Time", "WESM"]].dropna(subset=["WESM"])
                        if not wesm_d.empty:
                            ch_wesm = alt.Chart(wesm_d).mark_bar().encode(
                                x=alt.X("Time:T", title="Time", axis=alt.Axis(format="%H:%M")), 
                                y=alt.Y("WESM:Q", title="WESM Balance (kWh)"),
                                color=alt.condition(alt.datum.WESM < 0, alt.value("#ff9900"), alt.value("#4c78a8")), # Orange for import, Blue for export
                                tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), alt.Tooltip("WESM:Q", format=",.2f")]
                            ).properties(title=f"WESM Hourly Balance for {selected_date_str}", height=400).interactive()
                            st.altair_chart(ch_wesm, use_container_width=True)
                            wt = wesm_d["WESM"].sum(); 
                            ws = f"Net Import ({abs(wt):,.2f} kWh)" if wt < 0 else (f"Net Export ({wt:,.2f} kWh)" if wt > 0 else "Balanced (0 kWh)")
                            st.info(f"Daily WESM: {ws}")
                        else: st.info("No WESM data for chart.")
                    else: st.info("WESM data column N/A or all NaN.")
            else: st.warning(f"Plotting columns missing/null for {selected_date_str}.")
        else: st.warning("Time column invalid for charts.")


        # Sankey Chart - for the interval of Max Total MQ
        max_mq_interval_time_str_header = ""
        can_gen_sankey = False
        int_mq_sankey, int_bcq_sankey, int_wesm_sankey_interval = 0.0, 0.0, 0.0

        if "Total_MQ" in data.columns and not data["Total_MQ"].isnull().all() and \
           "Total_BCQ" in data.columns and not data["Total_BCQ"].isnull().all() : # Ensure BCQ is also present
            max_mq_day_val = data["Total_MQ"].max(skipna=True)
            
            # We need positive MQ or positive BCQ to make a meaningful Sankey
            if pd.notna(max_mq_day_val) : # Max MQ can be zero if there's BCQ (export)
                max_mq_row_idx = data["Total_MQ"].idxmax(skipna=True)
                max_mq_row = data.loc[max_mq_row_idx]

                int_mq_sankey = max_mq_row["Total_MQ"]
                if pd.isna(int_mq_sankey): int_mq_sankey = 0.0

                int_bcq_sankey = max_mq_row["Total_BCQ"]
                if pd.isna(int_bcq_sankey): int_bcq_sankey = 0.0
                
                # WESM for this specific interval
                int_wesm_sankey_interval = int_bcq_sankey - int_mq_sankey

                # Condition for generating Sankey:
                # Either MQ > 0, or (MQ=0 and BCQ > 0, implying pure export)
                if int_mq_sankey > 0.001 or (abs(int_mq_sankey) < 0.001 and int_bcq_sankey > 0.001):
                    time_obj = max_mq_row["Time"]
                    if pd.notna(time_obj) and hasattr(time_obj, 'strftime'):
                        max_mq_interval_time_str_header = time_obj.strftime("%H:%M")
                        can_gen_sankey = True
                    else:
                        max_mq_interval_time_str_header = str(time_obj) # Fallback
                        st.warning(f"Time object for max MQ interval is not standard: {time_obj}. Sankey might use this as label.")
                        can_gen_sankey = True # Still try
                else: # Neither MQ > 0 nor (MQ=0 and BCQ >0)
                     can_gen_sankey = False


        if can_gen_sankey:
            st.subheader(f"⚡ Energy Flow for Interval ({max_mq_interval_time_str_header} on {selected_date_str})")
            sankey_fig = create_sankey_chart(int_mq_sankey, int_wesm_sankey_interval, selected_date_str, max_mq_interval_time_str_header)
            if sankey_fig: st.plotly_chart(sankey_fig, use_container_width=True)
            # else: st.info("Sankey chart not generated due to data conditions in create_sankey_chart.") #create_sankey_chart handles its own info messages
        else:
            st.subheader("⚡ Energy Flow Sankey Chart")
            st.info("Sankey chart not generated. Conditions not met (e.g., Max Total MQ is not positive, or relevant BCQ/MQ data for the max MQ interval is zero or unavailable).")


def show_about_page():
    st.header("About This Dashboard")
    st.write("""
    ### Energy Trading Dashboard
    This dashboard provides visualization and analysis of energy trading data, including:
    - Metered quantities (MQ), Bilateral contract quantities (BCQ)
    - WESM (Wholesale Electricity Spot Market) balances calculated as `Total_BCQ - Total_MQ`.
    - Generator contributions sourced from `BCQ_Hourly` table columns for the specific interval.
    - Destination/Load consumption sourced from `MQ_Hourly` table columns for the specific interval.
    - Energy flow visualization (Sankey diagram) for the interval of maximum MQ on the selected day.
    
    Data is sourced from a PostgreSQL database with hourly measurements.
    
    ### Features
    - Interactive date selection with session state persistence.
    - Summary metrics for prices, MQ.
    - Tabbed data tables for hourly figures and daily summaries.
    - Interactive Altair charts for time-series data, including WESM balance.
    - Sankey diagram detailing energy flow for the specific interval of maximum MQ, using actual data from `BCQ_Hourly` (for generation sources) and `MQ_Hourly` (for consumption sinks).
    
    ### WESM Interpretation
    - **WESM Value = Total_BCQ - Total_MQ** (for each interval)
    - **Negative WESM values** (i.e., MQ > BCQ for an interval) indicate a **Net Import** from WESM during that interval. This means local demand (MQ) exceeded local contractual supply (BCQ), and the difference was sourced from WESM.
    - **Positive WESM values** (i.e., BCQ > MQ for an interval) indicate a **Net Export** to WESM during that interval. This means local contractual supply (BCQ) exceeded local demand (MQ), and the surplus was sent to WESM.
    This interpretation is reflected in the WESM chart colors and the Sankey diagram logic.
    
    ### Need Help?
    Contact the system administrator for issues or questions.
    """)
    st.markdown("---"); st.markdown(f"<p style='text-align: center;'>App Version 1.3 (Sankey with DB Data) | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)

if __name__ == "__main__": main()
