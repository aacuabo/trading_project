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

# IMPORTANT: Update the KEYS of this dictionary to your ACTUAL database column names from BCQ_Hourly
GENERATOR_LONG_TO_SHORT_MAP = {
    "FDC_Misamis_Power_Corporation__FDC": 'FDC',
    "GNPower_Kauswagan_Ltd._Co._GNPKLCO": 'GNPK',
    "Power_Sector_Assets_and_Liabilities_Management_Corporation_PSAL": 'PSALM', # Kept short name PSALM, DB col ends _PSAL
    "Sarangani_Energy_Corporation_SEC": 'SEC',
    "Therma_South,_Inc._TSI": 'TSI',
    "Malita_Power_Inc._SMCPC": 'MPI'
}

# IMPORTANT: VERIFY and UPDATE the KEYS of this dictionary to your ACTUAL database column names from MQ_Hourly
DESTINATION_LONG_TO_SHORT_MAP = {
    "14BGN_T1L1_KIDCOTE01_NET": 'M1/M6/M8', # Example: If actual name is C14BGN_T1L1_KIDCOTE01_NET, update key
    "14BGN_T1L1_KIDCOTE02_NET": 'M2',
    "14BGN_T1L1_KIDCOTE03_NET": 'M3',
    "14BGN_T1L1_KIDCOTE04_NET": 'M4',
    "14BGN_T2L1_KIDCOTE05_NET": 'M5',
    "14BGN_T1L1_KIDCOTE08_NET": 'M7',
    "14BGN_T1L1_KIDCOTE10_NET": 'M9',
    "14BGN_T1L1_KIDCSCV01_DEL": 'KIDCSCV01_DEL',
    "14BGN_T1L1_KIDCSCV02_DEL": 'KIDCSCV02_DEL'
    # Add or modify entries here based on your actual MQ_Hourly column names for destinations
}
# Example of how you might need to change DESTINATION_LONG_TO_SHORT_MAP if names were altered:
# DESTINATION_LONG_TO_SHORT_MAP = {
#     "C14BGN_T1L1_KIDCOTE01_NET": 'M1/M6/M8', # Assuming 'C' was prepended or similar
#     "C14BGN_T1L1_KIDCOTE02_NET": 'M2',
#     ... etc.
# }


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
        # These column names might not need explicit double quotes if they are simple identifiers (no spaces, standard chars)
        # However, quoting them is safer.
        query_columns_list = []
        for col_name in GENERATOR_LONG_TO_SHORT_MAP.keys():
            # If column names are simple like FDC_Misamis_Power_Corporation__FDC,
            # they might not strictly need double quotes in PostgreSQL, but it's safer.
            # If they contain spaces or special characters (which yours now don't seem to), they MUST be quoted.
            query_columns_list.append(f'"{col_name}"')
        query_columns_str = ', '.join(query_columns_list)


        query = f"""
            SELECT {query_columns_str}
            FROM "BCQ_Hourly"
            WHERE "Date" = %(selected_date)s AND "Time" = %(interval_time)s;
        """
        params = {"selected_date": selected_date_str, "interval_time": interval_time_db_format}
        # st.write(f"DEBUG SQL Gen: {query} with params {params}") # Uncomment for debugging SQL
        interval_data_df = pd.read_sql(query, engine, params=params)

        if not interval_data_df.empty:
            row = interval_data_df.iloc[0]
            for long_name, short_name in GENERATOR_LONG_TO_SHORT_MAP.items():
                if long_name in row:
                    value = pd.to_numeric(row[long_name], errors='coerce')
                    contributions[short_name] = value if pd.notna(value) and value > 0 else 0.0
        # else:
            # st.info(f"No generator contribution data found in BCQ_Hourly for {selected_date_str} at {interval_time_db_format}.")
        return contributions
    except Exception as e:
        st.error(f"Error fetching Sankey generator contributions: {e}")
        # st.exception(e) # Uncomment for more detailed traceback in Streamlit during development
        return contributions

@st.cache_data(ttl=600)
def fetch_sankey_destination_consumption(selected_date_str: str, interval_time_db_format: str):
    """Fetches actual destination consumption from MQ_Hourly for a specific date and time interval."""
    consumption = {short_name: 0.0 for short_name in DESTINATION_LONG_TO_SHORT_MAP.values()}
    if not DESTINATION_LONG_TO_SHORT_MAP:
        st.warning("Destination mapping is empty. Cannot fetch consumption.")
        return consumption
    try:
        engine = get_sqlalchemy_engine()
        query_columns_list = []
        for col_name in DESTINATION_LONG_TO_SHORT_MAP.keys():
            query_columns_list.append(f'"{col_name}"') # Quote column names
        query_columns_str = ', '.join(query_columns_list)

        query = f"""
            SELECT {query_columns_str}
            FROM "MQ_Hourly"
            WHERE "Date" = %(selected_date)s AND "Time" = %(interval_time)s;
        """
        params = {"selected_date": selected_date_str, "interval_time": interval_time_db_format}
        # st.write(f"DEBUG SQL Dest: {query} with params {params}") # Uncomment for debugging SQL
        interval_data_df = pd.read_sql(query, engine, params=params)

        if not interval_data_df.empty:
            row = interval_data_df.iloc[0]
            for long_name, short_name in DESTINATION_LONG_TO_SHORT_MAP.items():
                if long_name in row:
                    value = pd.to_numeric(row[long_name], errors='coerce')
                    consumption[short_name] = value if pd.notna(value) and value > 0 else 0.0
        # else:
            # st.info(f"No destination consumption data found in MQ_Hourly for {selected_date_str} at {interval_time_db_format}.")
        return consumption
    except Exception as e:
        st.error(f"Error fetching Sankey destination consumption: {e}")
        # st.exception(e) # Uncomment for more detailed traceback in Streamlit during development
        return consumption

def create_sankey_chart(interval_mq_val: float, interval_wesm_val: float, selected_date_str: str, interval_time_hh_mm_str: str):
    if pd.isna(interval_mq_val) or interval_mq_val < 0:
        st.info(f"Max Interval MQ is invalid ({interval_mq_val:,.0f} kWh) for {interval_time_hh_mm_str} on {selected_date_str}. Sankey not generated.")
        return None
    if interval_mq_val == 0 and (pd.isna(interval_wesm_val) or interval_wesm_val <= 0):
         if not (interval_wesm_val > 0) :
            st.info(f"Max Interval MQ is zero and no significant WESM activity for {interval_time_hh_mm_str} on {selected_date_str}. Sankey not generated.")
            return None
            
    if pd.isna(interval_wesm_val):
        interval_wesm_val = 0

    actual_total_bcq_for_interval = interval_mq_val + interval_wesm_val
    actual_total_mq_for_interval = interval_mq_val

    interval_time_db_format = interval_time_hh_mm_str + ":00" # Assumes DB Time is HH:MM:SS string

    generator_contributions = fetch_sankey_generator_contributions(selected_date_str, interval_time_db_format)
    destination_consumptions = fetch_sankey_destination_consumption(selected_date_str, interval_time_db_format)
    
    sankey_node_labels, node_indices, sankey_sources_indices, sankey_targets_indices, sankey_values, node_colors = [], {}, [], [], [], []
    
    def add_node(label, color="grey"):
        if label not in node_indices:
            node_indices[label] = len(sankey_node_labels)
            sankey_node_labels.append(label)
            node_colors.append(color)
        return node_indices[label]

    middle_node_label = f"Distribution Hub ({actual_total_bcq_for_interval:,.0f} kWh)"
    middle_node_idx = add_node(middle_node_label, "orange")

    total_gen_contribution_value = sum(v for v in generator_contributions.values() if pd.notna(v) and v > 0.01)

    if total_gen_contribution_value > 0.01 :
        for short_name, value in generator_contributions.items():
            if value > 0.01:
                percentage_of_total_bcq = (value / actual_total_bcq_for_interval * 100) if actual_total_bcq_for_interval > 0 else 0
                gen_node_label = f"Gen: {short_name} ({value:,.0f} kWh, {percentage_of_total_bcq:.1f}%)"
                gen_node_idx = add_node(gen_node_label, "blue")
                sankey_sources_indices.append(gen_node_idx)
                sankey_targets_indices.append(middle_node_idx)
                sankey_values.append(value)

    if interval_wesm_val < 0:
        import_value = abs(interval_wesm_val)
        if import_value > 0.01:
            # Total supply to hub = local generation (BCQ component from generators) + WESM Import
            # Here, actual_total_bcq_for_interval represents the BCQ part from generators.
            # The total energy handled by the hub on the supply side would be actual_total_bcq_for_interval + import_value.
            # This sum is actually equal to MQ in this case (since BCQ + Import = MQ).
            total_hub_supply = actual_total_bcq_for_interval + import_value 
            percentage = (import_value / total_hub_supply * 100) if total_hub_supply > 0 else 0
            wesm_label = f"WESM Import ({import_value:,.0f} kWh, {percentage:.1f}%)"
            wesm_node_idx = add_node(wesm_label, "red")
            sankey_sources_indices.append(wesm_node_idx)
            sankey_targets_indices.append(middle_node_idx)
            sankey_values.append(import_value)

    if actual_total_mq_for_interval > 0.01 :
        local_demand_node_label = f"Local Demand (MQ) ({actual_total_mq_for_interval:,.0f} kWh)"
        local_demand_node_idx = add_node(local_demand_node_label, "lightgreen")
        sankey_sources_indices.append(middle_node_idx)
        sankey_targets_indices.append(local_demand_node_idx)
        sankey_values.append(actual_total_mq_for_interval)

        for short_name, value in destination_consumptions.items():
            if value > 0.01:
                percentage_of_total_mq = (value / actual_total_mq_for_interval * 100) if actual_total_mq_for_interval > 0 else 0
                dest_node_label = f"Load: {short_name} ({value:,.0f} kWh, {percentage_of_total_mq:.1f}%)"
                dest_node_idx = add_node(dest_node_label, "green")
                sankey_sources_indices.append(local_demand_node_idx)
                sankey_targets_indices.append(dest_node_idx)
                sankey_values.append(value)
    
    if interval_wesm_val > 0:
        export_value = interval_wesm_val
        if export_value > 0.01:
            percentage = (export_value / actual_total_bcq_for_interval) * 100 if actual_total_bcq_for_interval > 0 else 0
            wesm_label = f"WESM Export ({export_value:,.0f} kWh, {percentage:.1f}%)"
            wesm_node_idx = add_node(wesm_label, "purple")
            sankey_sources_indices.append(middle_node_idx)
            sankey_targets_indices.append(wesm_node_idx)
            sankey_values.append(export_value)
            
    if not sankey_values or sum(sankey_values) < 0.1:
        sum_vals = sum(sankey_values) if sankey_values else 0
        st.info(f"Not enough significant flow data for Max MQ interval ({interval_time_hh_mm_str}) on {selected_date_str} to draw Sankey chart. Sum of values: {sum_vals:.2f}")
        return None
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=25, thickness=20, line=dict(color="black", width=0.5), label=sankey_node_labels, color=node_colors),
        link=dict(source=sankey_sources_indices, target=sankey_targets_indices, value=sankey_values,
                  hovertemplate='Flow from %{source.label} to %{target.label}: %{value:,.0f} kWh<extra></extra>') 
    )])
    fig.update_layout(title_text=f"Energy Flow for Interval ({interval_time_hh_mm_str}) on {selected_date_str}", font_size=10, height=700)
    return fig

# --- STREAMLIT UI (main, show_dashboard, show_about_page functions remain the same as in the previous correct version) ---
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
    spacer_left, main_content, spacer_right = st.columns([0.2, 5, 0.2]) 
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
                        
                        line_charts = alt.Chart(pd.DataFrame()) # Placeholder
                        bar_chart = alt.Chart(pd.DataFrame())   # Placeholder

                        energy_metrics = [m for m in ["Total_MQ", "Total_BCQ"] if m in ep_c]
                        if energy_metrics:
                            line_charts = base_chart.transform_filter(
                                alt.FieldOneOfPredicate(field='Metric', oneOf=energy_metrics)
                            ).mark_line(point=True).encode(
                                y=alt.Y("Value:Q", title="Energy (kWh)", scale=alt.Scale(zero=True)),
                                color=alt.Color("Metric:N", legend=alt.Legend(orient='bottom'), 
                                                scale=alt.Scale(domain=['Total_MQ', 'Total_BCQ'], range=['#FFC20A', '#1A85FF'])),
                                tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", alt.Tooltip("Value:Q", format=",.2f")]
                            )

                        if "Prices" in ep_c:
                            bar_chart = base_chart.transform_filter(
                                alt.datum.Metric == "Prices"
                            ).mark_bar(color="#40B0A6").encode(
                                y=alt.Y("Value:Q", title="Price (PHP/kWh)", scale=alt.Scale(zero=True)),
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
                with chart_tabs[1]:
                    if "WESM" in melt_cols:
                        wesm_d = data[["Time", "WESM"]].dropna(subset=["WESM"])
                        if not wesm_d.empty:
                            ch_wesm = alt.Chart(wesm_d).mark_bar().encode(
                                x=alt.X("Time:T", title="Time", axis=alt.Axis(format="%H:%M")), 
                                y=alt.Y("WESM:Q", title="WESM Balance (kWh)"),
                                color=alt.condition(alt.datum.WESM < 0, alt.value("#ff9900"), alt.value("#4c78a8")),
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

        max_mq_interval_time_str_header = ""
        can_gen_sankey = False
        int_mq_sankey, int_bcq_sankey, int_wesm_sankey_interval = 0.0, 0.0, 0.0

        if "Total_MQ" in data.columns and not data["Total_MQ"].isnull().all() and \
           "Total_BCQ" in data.columns and not data["Total_BCQ"].isnull().all() :
            max_mq_day_val = data["Total_MQ"].max(skipna=True) # Can be 0 or negative, but we want the interval
            
            if pd.notna(max_mq_day_val): # If max_mq_day_val is NaN, means all MQ are NaN
                # Find the row corresponding to the maximum Total_MQ for the day
                # If multiple intervals have the same max MQ, idxmax() picks the first.
                max_mq_row_idx = data["Total_MQ"].idxmax(skipna=True) 
                max_mq_row = data.loc[max_mq_row_idx]

                int_mq_sankey = max_mq_row["Total_MQ"]
                if pd.isna(int_mq_sankey): int_mq_sankey = 0.0

                int_bcq_sankey = max_mq_row["Total_BCQ"]
                if pd.isna(int_bcq_sankey): int_bcq_sankey = 0.0
                
                int_wesm_sankey_interval = int_bcq_sankey - int_mq_sankey

                if int_mq_sankey >= 0 or int_bcq_sankey > 0.001: # Allow MQ=0 if BCQ>0 (export)
                    time_obj = max_mq_row["Time"]
                    if pd.notna(time_obj) and hasattr(time_obj, 'strftime'):
                        max_mq_interval_time_str_header = time_obj.strftime("%H:%M")
                        can_gen_sankey = True
                    else:
                        max_mq_interval_time_str_header = str(time_obj)
                        st.warning(f"Time object for max MQ interval is not standard: {time_obj}. Sankey might use this as label.")
                        can_gen_sankey = True # Still try
                else:
                     can_gen_sankey = False
        
        if can_gen_sankey:
            st.subheader(f"⚡ Energy Flow for Interval ({max_mq_interval_time_str_header} on {selected_date_str})")
            sankey_fig = create_sankey_chart(int_mq_sankey, int_wesm_sankey_interval, selected_date_str, max_mq_interval_time_str_header)
            if sankey_fig: st.plotly_chart(sankey_fig, use_container_width=True)
        else:
            st.subheader("⚡ Energy Flow Sankey Chart")
            st.info("Sankey chart not generated. Conditions not met (e.g., Max Total MQ and relevant BCQ for the max MQ interval are zero, negative, or data unavailable).")


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
    
    Data is sourced from a PostgreSQL database with hourly measurements. The Sankey diagram's generator and load breakdowns are now based on actual column data from their respective tables for the specific interval.
    
    ### Features
    - Interactive date selection with session state persistence.
    - Summary metrics for prices, MQ.
    - Tabbed data tables for hourly figures and daily summaries.
    - Interactive Altair charts for time-series data, including WESM balance.
    - Sankey diagram detailing energy flow for the specific interval of maximum MQ, using actual data from `BCQ_Hourly` (for generation sources) and `MQ_Hourly` (for consumption sinks).
    
    ### WESM Interpretation
    - **WESM Value = Total_BCQ - Total_MQ** (for each interval)
    - **Negative WESM values** (i.e., MQ > BCQ for an interval) indicate a **Net Import** from WESM during that interval.
    - **Positive WESM values** (i.e., BCQ > MQ for an interval) indicate a **Net Export** to WESM during that interval.
    This interpretation is reflected in the WESM chart colors and the Sankey diagram logic.
    
    ### Need Help?
    Contact the system administrator for issues or questions.
    """)
    st.markdown("---"); st.markdown(f"<p style='text-align: center;'>App Version 1.4 (DB Column Name Fix) | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)

if __name__ == "__main__": main()
