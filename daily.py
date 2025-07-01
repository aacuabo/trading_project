import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime, date
import altair as alt
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple

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
        db = st.secrets["database"]["db"]
        port = int(st.secrets["database"]["port"])
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
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
                # Ensure time parsing handles various formats gracefully, coercing errors
                df['Time'] = pd.to_datetime(selected_date_str + ' ' + df['Time'].str.strip(), errors='coerce')
                df.dropna(subset=['Time'], inplace=True) # Drop rows where time could not be parsed
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
def fetch_sankey_generator_contributions(selected_date_str: str, interval_time_db_format: str):
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
                        if long_name in GENERATOR_COLUMNS_TO_SCALE:
                            value *= 1000 # Apply scaling
                        contributions[short_name] = value if value > 0 else 0.0
                    else:
                        contributions[short_name] = 0.0
        return contributions
    except Exception as e:
        st.error(f"Error fetching Sankey generator contributions: {e}")
        return contributions

@st.cache_data(ttl=600)
def fetch_sankey_destination_consumption(selected_date_str: str, interval_time_db_format: str):
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
    if pd.isna(interval_mq_val) or interval_mq_val < 0:
        st.info(f"Invalid interval data ({interval_time_hh_mm_str}, {selected_date_str}): MQ = {interval_mq_val:,.0f} kWh")
        return None

    interval_time_db_format = interval_time_hh_mm_str + ":00"

    scaled_generator_contributions = fetch_sankey_generator_contributions(selected_date_str, interval_time_db_format)
    destination_consumptions = fetch_sankey_destination_consumption(selected_date_str, interval_time_db_format)

    sum_scaled_generator_contributions = sum(v for v in scaled_generator_contributions.values() if pd.notna(v))
    actual_total_mq_for_interval = interval_mq_val

    wesm_value_for_sankey = sum_scaled_generator_contributions - actual_total_mq_for_interval

    if sum_scaled_generator_contributions < 0.01 and actual_total_mq_for_interval < 0.01:
        st.info(f"Insufficient flow data for {interval_time_hh_mm_str} on {selected_date_str}")
        return None

    sankey_node_labels, node_indices = [], {}
    sankey_sources_indices, sankey_targets_indices, sankey_values = [], [], []

    COLOR_PALETTE = {
        "junction": "#E69F00",
        "generator": "#0072B2",
        "wesm_import": "#009E73",
        "load": "#D55E00",
        "wesm_export": "#CC79A7"
    }
    node_colors = []

    def add_node(label, color):
        if label not in node_indices:
            node_indices[label] = len(sankey_node_labels)
            sankey_node_labels.append(label)
            node_colors.append(color)
        return node_indices[label]

    total_flow_through_junction = sum_scaled_generator_contributions
    if wesm_value_for_sankey < 0:
        total_flow_through_junction += abs(wesm_value_for_sankey)

    junction_node_label = f"Max Demand ({total_flow_through_junction:,.0f} kWh)"
    junction_node_idx = add_node(junction_node_label, COLOR_PALETTE["junction"])

    for short_name, value in scaled_generator_contributions.items():
        if value > 0.01:
            percentage = (value / sum_scaled_generator_contributions * 100) if sum_scaled_generator_contributions > 0 else 0
            gen_node_label = f"{short_name} ({value:,.0f} kWh, {percentage:.1f}%)"
            gen_node_idx = add_node(gen_node_label, COLOR_PALETTE["generator"])
            sankey_sources_indices.append(gen_node_idx)
            sankey_targets_indices.append(junction_node_idx)
            sankey_values.append(value)

    if wesm_value_for_sankey < 0:
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
        title=dict(
            text=f"Energy Flow: {interval_time_hh_mm_str}, {selected_date_str}",
            font=dict(size=16)
        ),
        font=dict(family="Arial, sans-serif", size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=600,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

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
    st.title("📈 Daily Energy Trading Dashboard")
    show_dashboard()

def show_dashboard():
    st.title("📈 Daily Energy Trading Dashboard")
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
            data['WESM'] = data['Total_BCQ'] - data['Total_MQ']
        else:
            data['WESM'] = pd.NA
            st.warning("Could not calculate WESM as Total_BCQ or Total_MQ is missing or not numeric.")

        st.subheader("Data Overview")
        tbl_tabs = st.tabs(["Summary Metrics", "Hourly Data", "Daily Summary"])

        with tbl_tabs[0]:
            st.markdown('<div style="display: flex; justify-content: center; align-items: center; height: 100%;">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            if "Prices" in data.columns and pd.api.types.is_numeric_dtype(data["Prices"]):
                pv = data["Prices"].dropna();
                col1.metric("Max Price (Php/kWh)", f"{pv.max():,.2f}" if not pv.empty else "N/A")
                col2.metric("Avg Price (Php/kWh)", f"{pv.mean():,.2f}" if not pv.empty else "N/A")
                col3.metric("Min Price (Php/kWh)", f"{pv.min():,.2f}" if not pv.empty else "N/A")
            else: [c.warning("Price N/A") for c in [col1, col2, col3]]

            max_mq_val_display, max_mq_time_display = "N/A", "N/A"
            if "Total_MQ" in data.columns and pd.api.types.is_numeric_dtype(data["Total_MQ"]) and not data["Total_MQ"].isnull().all():
                max_mq_val_for_day = data["Total_MQ"].max(skipna=True)
                if pd.notna(max_mq_val_for_day):
                    max_mq_idx_for_day_display = data["Total_MQ"].idxmax(skipna=True)
                    t_obj_display = data.loc[max_mq_idx_for_day_display, "Time"]
                    t_str_display = t_obj_display.strftime("%H:%M") if pd.notna(t_obj_display) and hasattr(t_obj_display, 'strftime') else "N/A"
                    max_mq_val_display = f"{max_mq_val_for_day:,.2f}"
                    max_mq_time_display = f"at {t_str_display}"
                    col4.metric("Max MQ (kWh)", max_mq_val_display, max_mq_time_display)
                else: col4.info("MQ all NaN.")
            else: col4.warning("MQ N/A")


        with tbl_tabs[1]:
            df_display = data.copy();
            if 'Time' in df_display and pd.api.types.is_datetime64_any_dtype(df_display['Time']):
                df_display['Time'] = df_display['Time'].dt.strftime('%H:%M')
            format_dict = {
                    col: '{:,.2f}' for col in df_display.columns if col != 'Time'
                }
            st.dataframe(df_display.style.format(format_dict, precision=2, na_rep="N/A"),height=300, use_container_width=True)
        with tbl_tabs[2]:
            s_dict = {}
            for c in ["Total_MQ", "Total_BCQ", "WESM"]:
                if c in data and pd.api.types.is_numeric_dtype(data[c]):
                    s_dict[f"{c} (kWh)"] = data[c].sum(skipna=True)
                else:
                    s_dict[f"{c} (kWh)"] = None  # Use None to apply formatting but show "N/A"

            if "Prices" in data and pd.api.types.is_numeric_dtype(data["Prices"]):
                s_dict["Avg Price (PHP/kWh)"] = (
                    data["Prices"].mean(skipna=True)
                    if not data["Prices"].dropna().empty else None
                )
            else:
                s_dict["Avg Price (PHP/kWh)"] = None

            # Build format dict for the summary keys that are numeric
            format_dict = {
                key: "{:,.2f}" for key, val in s_dict.items() if isinstance(val, (int, float))
            }

            # Display the styled summary
            st.dataframe(
                pd.DataFrame([s_dict]).style.format(format_dict, na_rep="N/A"),
                use_container_width=True
            )

        st.subheader("📈 Energy Metrics Over Time (Interactive)")
        if 'Time' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Time']):
            melt_cols = [c for c in ["Total_MQ", "Total_BCQ", "Prices", "WESM"] if c in data and data[c].isnull().sum() < len(data[c])]

            if melt_cols:
                chart_tabs = st.tabs(["Energy & Prices", "WESM Balance"])

                with chart_tabs[0]:

    # Define which columns to consider for energy and price metrics
                        ep_c = [c for c in ["Total_MQ", "Total_BCQ", "Prices"] if c in melt_cols]

                        if ep_c:
                            # Melt the DataFrame to long format for easier Altair plotting
                            # Ensure 'Time' column is datetime
                            data['Time'] = pd.to_datetime(data['Time'])
                            melt_ep = data.melt(id_vars=["Time"], value_vars=ep_c, var_name="Metric", value_name="Value").dropna(subset=['Value'])

                            # Initialize empty charts (as placeholders, will be overwritten if data exists)
                            line_charts_energy = alt.Chart(pd.DataFrame())
                            bar_chart_prices = alt.Chart(pd.DataFrame())

                            # Define energy metrics that are present in the data
                            energy_metrics = [m for m in ["Total_MQ", "Total_BCQ"] if m in ep_c]

                            # Create line charts for energy metrics if they exist
                            if energy_metrics:
                                line_charts_energy = alt.Chart(melt_ep).transform_filter(
                                    alt.FieldOneOfPredicate(field='Metric', oneOf=energy_metrics)
                                ).mark_line(point=True).encode(
                                    x=alt.X("Time:T", timeUnit="hours", title="Hour of Day"), # MODIFIED: Added x-axis for hours
                                    y=alt.Y("Value:Q", title="Energy (kWh)", scale=alt.Scale(zero=True)),
                                    color=alt.Color("Metric:N", legend=alt.Legend(orient='bottom'),
                                                    scale=alt.Scale(domain=['Total_MQ', 'Total_BCQ'], range=['#FFC20A', '#1A85FF'])), # Standard Altair colors
                                    tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", alt.Tooltip("Value:Q", format=",.2f")]
                                )

                            # Create a bar chart for prices if it exists
                            if "Prices" in ep_c:
                                bar_chart_prices = alt.Chart(melt_ep).transform_filter(
                                    alt.datum.Metric == "Prices"
                                ).mark_bar(color="#40B0A6", opacity=0.3).encode( # Standard Altair color
                                    x=alt.X("Time:T", timeUnit="hours", title="Hour of Day"), # MODIFIED: Added x-axis for hours
                                    y=alt.Y("Value:Q", title="Price (PHP/kWh)", scale=alt.Scale(zero=False,domain=[0, 32000])),
                                    tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", alt.Tooltip("Value:Q", format=",.2f")]
                                )

                            # Combine charts based on what data is available
                            if energy_metrics and "Prices" in ep_c:
                                # Layer both energy lines and price bars
                                # Ensure the x-axis title is consistent or handled by Altair's resolution
                                # If both charts define the x-axis identically, it should merge fine.
                                comb_ch = alt.layer(bar_chart_prices, line_charts_energy).resolve_scale(y='independent')
                            elif energy_metrics:
                                # Only show energy lines
                                comb_ch = line_charts_energy
                            elif "Prices" in ep_c:
                                # Only show price bars
                                comb_ch = bar_chart_prices
                            else:
                                # No relevant data for any chart
                                comb_ch = alt.Chart(pd.DataFrame()).mark_text(text="No Energy/Price Data for chart.").encode() # No specific axis needed here

                            # Display the combined chart with a title
                            # Using selected_date_str as per the original fixed comment
                            if hasattr(comb_ch, 'properties'): # Check if comb_ch is an Altair chart
                                st.altair_chart(comb_ch.properties(title=f"Metrics for {selected_date_str}"), use_container_width=True)
                            else: # Should not happen with the current logic, but as a safeguard
                                st.info("Chart could not be generated.")
                        else:
                            # Inform the user if no MQ, BCQ, or Price data is found at all
                            st.info("No MQ, BCQ or Price data for this chart.")

                with chart_tabs[1]:
                    # Check if 'WESM' and 'Prices' columns are available in melt_cols (simulating actual check)
                    wesm_available = "WESM" in melt_cols
                    prices_available = "Prices" in melt_cols
                    charts_to_layer = []

                    # Common X-axis definition for consistent hourly display
                    x_axis_hourly = alt.X("Time:T", timeUnit="hours", title="Hour of Day", axis=alt.Axis(format="%H")) # Format to show only hour

                    if wesm_available:
                        wesm_d = data[["Time", "WESM"]].copy().dropna(subset=["WESM"]) # Use .copy() to avoid SettingWithCopyWarning
                        if not wesm_d.empty:
                            # Calculate absolute WESM for Y-axis magnitude
                            # The transform_calculate is applied to the chart using this data
                            ch_wesm = alt.Chart(wesm_d).transform_calculate(
                                AbsWESM="abs(datum.WESM)"  # Calculate absolute value of WESM
                            ).mark_bar(opacity=0.3).encode(
                                x=x_axis_hourly,
                                y=alt.Y("AbsWESM:Q", title="WESM Volume (kWh)", axis=alt.Axis(titleColor='#800080')), # Purple title
                                color=alt.condition(
                                    alt.datum.WESM > 0,
                                    alt.value('#4CAF50'),  # Green for positive WESM (Export)
                                    alt.value('#F44336')   # Red for negative WESM (Import)
                                ),
                                tooltip=[
                                    alt.Tooltip("Time:T", timeUnit="hours", title="Hour", format="%H"), # Show hour
                                    alt.Tooltip("WESM:Q", format=",.0f", title="WESM (Net)") # Show original WESM value
                                ]
                            )
                            charts_to_layer.append(ch_wesm)

                            # Calculate and display daily WESM summary
                            wt = wesm_d["WESM"].sum()
                            ws = f"Net Import ({abs(wt):,.2f} kWh)" if wt < 0 else (f"Net Export ({wt:,.2f} kWh)" if wt > 0 else "Balanced (0 kWh)")
                            st.info(f"Daily WESM (unscaled): {ws}")
                        else:
                            st.info("No WESM data for WESM chart.")

                    if prices_available:
                        prices_d = data[["Time", "Prices"]].copy().dropna(subset=["Prices"]) # Use .copy()
                        if not prices_d.empty:
                            ch_prices_line = alt.Chart(prices_d).mark_line(
                                point=True,
                                color='green', # Green line for prices
                                strokeDash=[3,3] # Dashed line style
                            ).encode(
                                x=x_axis_hourly,
                                y=alt.Y("Prices:Q", title="Price (PHP/kWh)",
                                        axis=alt.Axis(titleColor='green'), # Green title
                                        scale=alt.Scale(zero=False, domain=[0, 32000])), # Scale does not need to start at zero
                                tooltip=[
                                    alt.Tooltip("Time:T", timeUnit="hours", title="Hour", format="%H"), # Show hour
                                    alt.Tooltip("Prices:Q", format=",.2f", title="Price")
                                ]
                            )
                            charts_to_layer.append(ch_prices_line)
                        else:
                            st.info("No Price data for Price line chart.")

                    if len(charts_to_layer) == 2:
                        combined_wesm_price_chart = alt.layer(*charts_to_layer).resolve_scale(
                            y='independent' # Independent Y-scales for WESM and Prices
                        ).properties(
                            title=f"Hourly WESM Volume & Prices for {selected_date_str}",
                            height=400
                        ).configure_axis( # Apply label angle to all axes in the chart
                            labelAngle=-45
                        )
                        # REMOVED .interactive()
                        st.altair_chart(combined_wesm_price_chart, use_container_width=True)
                    elif len(charts_to_layer) == 1:
                        # Determine title for single chart
                        title_text = ""
                        if wesm_available and not prices_available:
                            title_text = "Hourly WESM Volume"
                        elif prices_available and not wesm_available:
                            title_text = "Hourly Prices"

                        single_chart = charts_to_layer[0].properties(
                            title=f"{title_text} for {selected_date_str}",
                            height=400
                        ).configure_axis( # Apply label angle
                            labelAngle=-45
                        )
                        # REMOVED .interactive()
                        st.altair_chart(single_chart, use_container_width=True)
                    else:
                        st.info("No WESM or Price data available for this tab.")

                    # Add expander for WESM chart explanation if WESM chart was potentially created
                    if wesm_available and any(c.mark == 'bar' for c in charts_to_layer if hasattr(c, 'mark')): # Check if WESM bar chart is in layers
                        with st.expander("Understanding WESM Values on Chart"):
                            st.markdown(
                                "- **Green Bars**: Net Export (Positive WESM values).\n"
                                "- **Red Bars**: Net Import (Negative WESM values).\n"
                                "- *Bar height represents the absolute volume of WESM interaction (kWh).*"
                            )
            else: st.warning(f"Plotting columns missing/null for {selected_date_str}.")
        else: st.warning("Time column invalid for charts.")

        max_mq_interval_time_str_header = ""
        can_gen_sankey = False
        interval_mq_unscaled, interval_bcq_unscaled, interval_wesm_unscaled = 0.0, 0.0, 0.0

        if "Total_MQ" in data.columns and not data["Total_MQ"].isnull().all() and \
           "Total_BCQ" in data.columns:
            max_mq_val_for_day = data["Total_MQ"].max(skipna=True)
            if pd.notna(max_mq_val_for_day):
                sankey_interval_idx = data["Total_MQ"].idxmax(skipna=True)
                sankey_interval_row = data.loc[sankey_interval_idx]
                interval_mq_unscaled = sankey_interval_row["Total_MQ"]
                if pd.isna(interval_mq_unscaled): interval_mq_unscaled = 0.0

                if "Total_BCQ" in sankey_interval_row and pd.notna(sankey_interval_row["Total_BCQ"]):
                    interval_bcq_unscaled = sankey_interval_row["Total_BCQ"]
                else:
                    interval_bcq_unscaled = 0.0

                interval_wesm_unscaled = interval_bcq_unscaled - interval_mq_unscaled
                time_obj = sankey_interval_row["Time"]

                if pd.notna(time_obj) and hasattr(time_obj, 'strftime'):
                    max_mq_interval_time_str_header = time_obj.strftime("%H:%M")
                    if interval_mq_unscaled >= 0 and (interval_mq_unscaled > 0.001 or interval_bcq_unscaled > 0.001):
                         can_gen_sankey = True
                else:
                    max_mq_interval_time_str_header = str(time_obj) if pd.notna(time_obj) else "N/A"
                    if interval_mq_unscaled >= 0 and (interval_mq_unscaled > 0.001 or interval_bcq_unscaled > 0.001):
                         can_gen_sankey = True

        if can_gen_sankey:
            st.subheader(f"⚡ Energy Flow for Interval ({max_mq_interval_time_str_header} on {selected_date_str})")
            sankey_fig = create_sankey_chart(interval_mq_unscaled, interval_wesm_unscaled, selected_date_str, max_mq_interval_time_str_header)
            if sankey_fig: st.plotly_chart(sankey_fig, use_container_width=True)
        else:
            st.subheader("⚡ Energy Flow Sankey Chart")
            st.info("Sankey chart not generated. Conditions for the chosen interval not met (e.g., Max MQ is zero, negative, or essential data for the interval is unavailable).")


def main():
    app_content()

if st.button("Rerun"):
    st.rerun()

if __name__ == "__main__":
    main()
