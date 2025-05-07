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

        # Properly handle datetime conversion
        if 'Time' in df.columns:
            try:
                # First convert to string if not already
                if not pd.api.types.is_string_dtype(df['Time']):
                    df['Time'] = df['Time'].astype(str)
                
                # Create full datetime
                df['Time'] = pd.to_datetime(selected_date_str + ' ' + df['Time'].str.strip(), errors='coerce') # Added .str.strip()
                df.dropna(subset=['Time'], inplace=True)
            except Exception as e:
                st.warning(f"Warning converting time values: {e}. Some time data may be incorrect.")

        # Ensure numeric columns are properly converted
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
def fetch_sankey_generator_contributions(selected_date_str: str, total_generation_to_distribute: float):
    """
    Creates dummy data for generator contributions.
    In a real implementation, this would fetch actual data from the database.
    """
    # Ensure total_generation_to_distribute is not negative for dummy data logic
    total_generation_to_distribute = max(0, total_generation_to_distribute)
    
    contributions = {}
    proportions = { # Ensure these sum close to 1 or normalize if they don't
        'FDC': 0.22, 'GNPK': 0.25, 'PSALM': 0.15,
        'SEC': 0.18, 'TSI': 0.12, 'MPI': 0.08
    }
    # Normalize proportions to sum to 1, to ensure accurate distribution
    sum_proportions = sum(proportions.values())
    if sum_proportions == 0: # Avoid division by zero if all proportions are zero
        if GENERATOR_LONG_TO_SHORT_MAP: # If there are generators, distribute equally
             for short_name in GENERATOR_LONG_TO_SHORT_MAP.values():
                contributions[short_name] = total_generation_to_distribute / len(GENERATOR_LONG_TO_SHORT_MAP)
        return contributions

    normalized_proportions = {k: v / sum_proportions for k, v in proportions.items()}

    for short_name in GENERATOR_LONG_TO_SHORT_MAP.values(): # Iterate over all known generators
        prop = normalized_proportions.get(short_name, 0) # Get proportion, default to 0 if not in map
        variation = 1.0 + (np.random.random() - 0.5) * 0.2  # ±10% random variation
        contributions[short_name] = total_generation_to_distribute * prop * variation
    
    current_sum = sum(contributions.values())
    if current_sum > 0:
        scaling_factor = total_generation_to_distribute / current_sum
        for short_name in contributions:
            contributions[short_name] *= scaling_factor
    elif total_generation_to_distribute > 0 and GENERATOR_LONG_TO_SHORT_MAP: # If sum is 0 but should be >0
        # Fallback to equal distribution if proportions led to zero sum somehow
        for short_name in GENERATOR_LONG_TO_SHORT_MAP.values():
            contributions[short_name] = total_generation_to_distribute / len(GENERATOR_LONG_TO_SHORT_MAP)

    return contributions


@st.cache_data(ttl=600)
def fetch_sankey_destination_consumption(selected_date_str: str, total_mq_to_distribute: float):
    """
    Creates dummy data for destination consumption.
    In a real implementation, this would fetch actual data from the database.
    """
    total_mq_to_distribute = max(0, total_mq_to_distribute) # Ensure non-negative
    consumption = {}
    proportions = { # Ensure these sum close to 1 or normalize
        'M1/M6/M8': 0.18, 'M2': 0.12, 'M3': 0.15, 'M4': 0.11, 'M5': 0.14,
        'M7': 0.09, 'M9': 0.13, 'KIDCSCV01_DEL': 0.04, 'KIDCSCV02_DEL': 0.04
    }
    sum_proportions = sum(proportions.values())
    if sum_proportions == 0:
        if DESTINATION_LONG_TO_SHORT_MAP:
            for short_name in DESTINATION_LONG_TO_SHORT_MAP.values():
                consumption[short_name] = total_mq_to_distribute / len(DESTINATION_LONG_TO_SHORT_MAP)
        return consumption

    normalized_proportions = {k: v / sum_proportions for k, v in proportions.items()}

    for short_name in DESTINATION_LONG_TO_SHORT_MAP.values(): # Iterate over all known destinations
        prop = normalized_proportions.get(short_name, 0)
        variation = 1.0 + (np.random.random() - 0.5) * 0.2
        consumption[short_name] = total_mq_to_distribute * prop * variation
        
    current_sum = sum(consumption.values())
    if current_sum > 0:
        scaling_factor = total_mq_to_distribute / current_sum
        for short_name in consumption:
            consumption[short_name] *= scaling_factor
    elif total_mq_to_distribute > 0 and DESTINATION_LONG_TO_SHORT_MAP:
        for short_name in DESTINATION_LONG_TO_SHORT_MAP.values():
            consumption[short_name] = total_mq_to_distribute / len(DESTINATION_LONG_TO_SHORT_MAP)
            
    return consumption


def create_sankey_chart(data: pd.DataFrame, selected_date_str: str):
    """Creates a Sankey chart showing energy flow, aligning with WESM interpretation."""
    
    if 'Total_MQ' not in data.columns or data['Total_MQ'].isnull().all() or not pd.api.types.is_numeric_dtype(data['Total_MQ']):
        st.warning("Cannot generate Sankey: Total_MQ data invalid or missing.")
        return None
    
    total_mq_sum = data['Total_MQ'].sum()
    if pd.isna(total_mq_sum) or total_mq_sum < 0: # MQ sum should not be negative
        st.info(f"Total MQ is invalid ({total_mq_sum:,.0f} kWh) for {selected_date_str}. Cannot generate Sankey chart.")
        return None
    # If total_mq_sum is exactly 0, we might still show flows if WESM causes generation
    if total_mq_sum == 0 and ('WESM' not in data.columns or pd.isna(data['WESM'].sum()) or data['WESM'].sum() == 0) :
        st.info(f"Total MQ is zero and no WESM activity for {selected_date_str}. Sankey chart not generated.")
        return None

    wesm_daily_sum = 0 # Default if WESM data is unavailable
    if 'WESM' in data.columns and pd.api.types.is_numeric_dtype(data['WESM']) and not data['WESM'].isnull().all():
        wesm_daily_sum = data['WESM'].sum()
        if pd.isna(wesm_daily_sum): wesm_daily_sum = 0 # handle case where sum results in NaN (e.g. all NaNs)
    
    # This is the amount local generators (or similar non-WESM sources) need to produce
    actual_local_generation_needed = total_mq_sum + wesm_daily_sum 
    
    # This is the amount consumed by local destinations (which is the Total MQ)
    actual_consumption_by_local_destinations = total_mq_sum

    generator_contributions = fetch_sankey_generator_contributions(selected_date_str, actual_local_generation_needed)
    destination_consumptions = fetch_sankey_destination_consumption(selected_date_str, actual_consumption_by_local_destinations)

    sankey_node_labels = []
    node_indices = {}
    sankey_sources_indices = []
    sankey_targets_indices = []
    sankey_values = []
    node_colors = []
    
    def add_node(label, color="grey"):
        if label not in node_indices:
            node_indices[label] = len(sankey_node_labels)
            sankey_node_labels.append(label)
            node_colors.append(color)
        return node_indices[label]

    # Middle Node: Total MQ
    middle_node_label = f"Total Daily MQ ({total_mq_sum:,.0f} kWh)"
    middle_node_idx = add_node(middle_node_label, "orange")

    # --- SOURCES feeding into MQ Node ---
    # Denominator for source percentages (should ideally sum up to total_mq_sum)
    # total_input_to_mq = sum(v for v in generator_contributions.values() if v > 0) + \
    #                     (abs(wesm_daily_sum) if wesm_daily_sum < 0 else 0)
    # Using total_mq_sum as the reference for percentages makes more sense for flows into/out of MQ.
    denominator_for_percentages = total_mq_sum if total_mq_sum > 0 else 1 # Avoid division by zero

    # Local Generators
    for short_name, value in generator_contributions.items():
        if value > 0.01: # Threshold to avoid tiny flows
            percentage = (value / denominator_for_percentages) * 100
            gen_node_label = f"{short_name} ({value:,.0f} kWh, {percentage:.1f}%)"
            gen_node_idx = add_node(gen_node_label, "blue")
            sankey_sources_indices.append(gen_node_idx)
            sankey_targets_indices.append(middle_node_idx)
            sankey_values.append(value)

    # WESM Import (if WESM is negative, it's an import)
    if wesm_daily_sum < 0:
        import_value = abs(wesm_daily_sum)
        if import_value > 0.01:
            percentage = (import_value / denominator_for_percentages) * 100
            wesm_label = f"WESM Import ({import_value:,.0f} kWh, {percentage:.1f}%)"
            wesm_node_idx = add_node(wesm_label, "red") # Imports are red
            sankey_sources_indices.append(wesm_node_idx)
            sankey_targets_indices.append(middle_node_idx)
            sankey_values.append(import_value)

    # --- DESTINATIONS fed from MQ Node ---
    # Local Consumer Destinations
    for short_name, value in destination_consumptions.items():
        if value > 0.01: # Threshold
            percentage = (value / denominator_for_percentages) * 100
            dest_node_label = f"{short_name} ({value:,.0f} kWh, {percentage:.1f}%)"
            dest_node_idx = add_node(dest_node_label, "green")
            sankey_sources_indices.append(middle_node_idx)
            sankey_targets_indices.append(dest_node_idx)
            sankey_values.append(value)

    # WESM Export (if WESM is positive, it's an export)
    if wesm_daily_sum > 0:
        export_value = wesm_daily_sum
        if export_value > 0.01:
            percentage = (export_value / denominator_for_percentages) * 100
            wesm_label = f"WESM Export ({export_value:,.0f} kWh, {percentage:.1f}%)"
            wesm_node_idx = add_node(wesm_label, "purple") # Exports are purple
            sankey_sources_indices.append(middle_node_idx)
            sankey_targets_indices.append(wesm_node_idx)
            sankey_values.append(export_value)
            
    if not sankey_values or sum(sankey_values) < 0.1: # Increased threshold for sum
        st.info(f"Not enough significant data to draw Sankey chart for {selected_date_str}.")
        return None
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=25, thickness=20,
            line=dict(color="black", width=0.5),
            label=sankey_node_labels,
            color=node_colors
        ),
        link=dict(
            source=sankey_sources_indices,
            target=sankey_targets_indices,
            value=sankey_values,
        )
    )])
    
    fig.update_layout(
        title_text=f"Energy Flow for {selected_date_str}",
        font_size=10, height=600
    )
    return fig


# --- STREAMLIT UI ---
def main():
    st.title("📊 Daily Energy Trading Dashboard")
    
    st.sidebar.header("Navigation")
    page_options = ["Dashboard", "About"]
    # Ensure "Dashboard" is default if session state is not set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
        
    page = st.sidebar.radio("Go to", page_options, index=page_options.index(st.session_state.current_page), key="nav_radio")
    st.session_state.current_page = page # Update session state on change

    if page == "About":
        show_about_page()
    else: # Dashboard
        show_dashboard()


def show_dashboard():
    """Shows the main dashboard page"""
    spacer_left, main_content, spacer_right = st.columns([0.5, 4, 0.5]) # Adjusted to [0.5, 4, 0.5]
    
    with main_content:
        available_dates = fetch_available_dates()
        
        if not available_dates:
            st.error("No available dates found. Check database connection and data.")
            st.stop()
        
        min_available_date = min(available_dates)
        max_available_date = max(available_dates)
        
        # Maintain selected date in session state if it exists and is valid
        if 'selected_date' not in st.session_state or \
           not (min_available_date <= st.session_state.selected_date <= max_available_date):
            st.session_state.selected_date = max_available_date # Default to the latest available date

        selected_date = st.date_input(
            "Select date",
            value=st.session_state.selected_date,
            min_value=min_available_date,
            max_value=max_available_date,
            key="date_picker" 
        )
        st.session_state.selected_date = selected_date # Update session state

        if selected_date not in available_dates:
            st.warning(f"Data may not be available for the exact date selected: {selected_date}. Displaying data for the closest available date if applicable.")
        
        selected_date_str = selected_date.strftime('%Y-%m-%d')
        data = fetch_data(selected_date_str)
        
        if data.empty:
            st.warning(f"No data available for selected date: {selected_date_str}.")
            return
            
        # Calculate WESM
        if all(col in data.columns for col in ["Total_BCQ", "Total_MQ"]) and \
           pd.api.types.is_numeric_dtype(data["Total_BCQ"]) and pd.api.types.is_numeric_dtype(data["Total_MQ"]):
            data['WESM'] = data['Total_BCQ'] - data['Total_MQ']
        else:
            st.warning("WESM column not calculated: Total_BCQ or Total_MQ are missing or not numeric.")
            data['WESM'] = pd.NA  # Ensure column exists
        
        # Display metrics
        st.subheader("Daily Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        # Price metrics
        if "Prices" in data.columns and not data["Prices"].empty and pd.api.types.is_numeric_dtype(data["Prices"]):
            prices_valid = data["Prices"].dropna()
            if not prices_valid.empty:
                max_price = prices_valid.max()
                avg_price = prices_valid.mean()
                min_price = prices_valid.min()
                col1.metric(label="Max Price (PHP/kWh)", value=f"{max_price:,.2f}")
                col2.metric(label="Avg Price (PHP/kWh)", value=f"{avg_price:,.2f}")
                col3.metric(label="Min Price (PHP/kWh)", value=f"{min_price:,.2f}")
            else:
                col1.metric(label="Max Price (PHP/kWh)", value="N/A")
                col2.metric(label="Avg Price (PHP/kWh)", value="N/A")
                col3.metric(label="Min Price (PHP/kWh)", value="N/A")
        else:
            col1.warning("Price data missing/invalid")
            col2.warning("Avg Price data missing/invalid")
            col3.warning("Min Price data missing/invalid")
        
        # MQ metrics
        if "Total_MQ" in data.columns and "Time" in data.columns and \
           not data["Total_MQ"].empty and pd.api.types.is_numeric_dtype(data["Total_MQ"]) and \
           not data["Total_MQ"].isnull().all():
            max_mq_value = data["Total_MQ"].max(skipna=True)
            if pd.notna(max_mq_value):
                max_mq_row_index = data["Total_MQ"].idxmax()
                max_mq_time = data.loc[max_mq_row_index, "Time"]
                max_mq_time_str = max_mq_time.strftime("%H:%M") if pd.api.types.is_datetime64_any_dtype(max_mq_time) else str(max_mq_time)
                col4.metric(label="Max Total MQ (kWh)", value=f"{max_mq_value:,.2f}")
                col4.write(f"at {max_mq_time_str}")
            else:
                col4.info("Total_MQ data is all NaN.")
        else:
            col4.warning("Max MQ data missing/invalid")
        
        # Data table with tabs
        st.subheader("Data Tables")
        table_tabs = st.tabs(["Hourly Data", "Daily Summary"]) # Renamed to avoid conflict
        
        with table_tabs[0]:
            if not data.empty:
                display_data = data.copy()
                if 'Time' in display_data.columns and pd.api.types.is_datetime64_any_dtype(display_data['Time']):
                    display_data['Time'] = display_data['Time'].dt.strftime('%H:%M')
                
                st.dataframe(display_data.style.format(precision=2, na_rep="N/A"), use_container_width=True)
            else:
                st.info("No hourly data available")
                
        with table_tabs[1]:
            if not data.empty:
                summary_dict = {}
                for col in ["Total_MQ", "Total_BCQ", "WESM"]:
                    if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                        summary_dict[f"{col} (kWh)"] = data[col].sum(skipna=True)
                    else:
                        summary_dict[f"{col} (kWh)"] = "N/A"
                if "Prices" in data.columns and pd.api.types.is_numeric_dtype(data["Prices"]):
                     summary_dict["Average Price (PHP/kWh)"] = data["Prices"].mean(skipna=True)
                else:
                    summary_dict["Average Price (PHP/kWh)"] = "N/A"

                summary_df = pd.DataFrame([summary_dict])
                st.dataframe(summary_df.style.format(precision=2, na_rep="N/A"), use_container_width=True)
            else:
                st.info("No data available for summary")
        
        # Charts
        st.subheader("📈 Energy Metrics Over Time (Interactive)")
        if 'Time' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Time']):
            columns_to_melt = ["Total_MQ", "Total_BCQ", "Prices", "WESM"]
            existing_cols_to_melt = [col for col in columns_to_melt if col in data.columns and not data[col].isnull().all()]
            
            if existing_cols_to_melt:
                chart_tabs = st.tabs(["Energy & Prices", "WESM Balance"]) # Renamed
                
                with chart_tabs[0]:
                    ep_cols = [col for col in ["Total_MQ", "Total_BCQ", "Prices"] if col in existing_cols_to_melt]
                    if ep_cols:
                        melted_data_ep = data.melt(
                            id_vars=["Time"],
                            value_vars=ep_cols,
                            var_name="Metric", value_name="Value"
                        )
                        melted_data_ep.dropna(subset=['Value'], inplace=True)
                        
                        # Chart for MQ and BCQ
                        energy_metrics = [m for m in ["Total_MQ", "Total_BCQ"] if m in ep_cols]
                        chart_energy = alt.Chart(pd.DataFrame({'A' : []})).mark_text() # Empty chart default
                        if energy_metrics:
                            chart_energy = alt.Chart(melted_data_ep[melted_data_ep["Metric"].isin(energy_metrics)]).mark_line(point=True).encode(
                                x=alt.X("Time:T", axis=alt.Axis(title="Time", format="%H:%M")),
                                y=alt.Y("Value:Q", title="Energy (kWh)", axis=alt.Axis(titleColor="#1A85FF"), scale=alt.Scale(zero=True)),
                                color=alt.Color("Metric:N", legend=alt.Legend(title="Metric", orient='bottom'),
                                                scale=alt.Scale(domain=['Total_MQ', 'Total_BCQ'], range=['#FFC20A', '#1A85FF'])),
                                tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", "Value:Q"]
                            ).properties(title="Energy Metrics")
                        
                        # Chart for Prices
                        chart_price = alt.Chart(pd.DataFrame({'A' : []})).mark_text() # Empty chart default
                        if "Prices" in ep_cols:
                            chart_price = alt.Chart(melted_data_ep[melted_data_ep["Metric"] == "Prices"]).mark_bar(color="#40B0A6").encode(
                                x=alt.X("Time:T", axis=alt.Axis(title="Time", format="%H:%M")),
                                y=alt.Y("Value:Q", title="Price (PHP/kWh)", axis=alt.Axis(titleColor="#40B0A6"), scale=alt.Scale(zero=True)),
                                tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", "Value:Q"]
                            ).properties(title="Prices")
                        
                        # Combine charts if both exist
                        if energy_metrics and "Prices" in ep_cols:
                            combined_chart = alt.layer(chart_price, chart_energy).resolve_scale(y='independent').properties(
                                title=f"Energy Metrics and Prices for {selected_date_str}"
                            ).interactive()
                        elif energy_metrics: combined_chart = chart_energy.properties(title=f"Energy Metrics for {selected_date_str}").interactive()
                        elif "Prices" in ep_cols: combined_chart = chart_price.properties(title=f"Prices for {selected_date_str}").interactive()
                        else: combined_chart = alt.Chart(pd.DataFrame()).mark_text(text="No data for Energy/Price chart.").properties(title="No Data")
                        
                        st.altair_chart(combined_chart, use_container_width=True)
                    else:
                        st.info("No MQ, BCQ or Price data for this chart.")
                
                with chart_tabs[1]:
                    if "WESM" in existing_cols_to_melt:
                        wesm_data = data[["Time", "WESM"]].copy()
                        wesm_data.dropna(subset=["WESM"], inplace=True)
                        
                        if not wesm_data.empty:
                            wesm_chart = alt.Chart(wesm_data).mark_bar().encode(
                                x=alt.X("Time:T", axis=alt.Axis(title="Time", format="%H:%M")),
                                y=alt.Y("WESM:Q", title="WESM Balance (kWh)"),
                                color=alt.condition(
                                    alt.datum.WESM < 0, # Negative WESM is Import (user clarification)
                                    alt.value("#ff9900"),  # Orange for Import
                                    alt.value("#4c78a8")   # Blue for Export (positive WESM)
                                ),
                                tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "WESM:Q"]
                            ).properties(
                                title=f"WESM Hourly Balance for {selected_date_str}",
                                height=400
                            ).interactive()
                            
                            st.altair_chart(wesm_chart, use_container_width=True)
                            
                            wesm_total = wesm_data["WESM"].sum()
                            # Corrected WESM summary based on user clarification
                            if wesm_total < 0: wesm_status = f"Net Import from WESM ({abs(wesm_total):,.2f} kWh)"
                            elif wesm_total > 0: wesm_status = f"Net Export to WESM ({wesm_total:,.2f} kWh)"
                            else: wesm_status = f"Balanced WESM (0 kWh)"
                            st.info(f"Daily WESM Balance: {wesm_status}")
                        else:
                            st.info("No WESM data for chart.")
                    else:
                        st.info("WESM data column not available.")
            else:
                st.warning(f"Required columns for plotting are missing or all null in data for {selected_date_str}.")
        else:
            st.warning("Time column not in expected datetime format or data is empty for charts.")
        
        # Sankey Chart
        st.subheader("⚡ Daily Energy Flow Sankey Chart")
        sankey_fig = create_sankey_chart(data.copy(), selected_date_str) # Pass a copy of data
        if sankey_fig:
            st.plotly_chart(sankey_fig, use_container_width=True)

def show_about_page():
    """Shows information about the dashboard"""
    st.header("About This Dashboard")
    
    st.write("""
    ### Energy Trading Dashboard
    
    This dashboard provides visualization and analysis of energy trading data, including:
    
    - Metered quantities (MQ)
    - Bilateral contract quantities (BCQ)
    - WESM (Wholesale Electricity Spot Market) balances
    - Generator contributions (currently illustrative using dummy data)
    - Energy flow visualization
    
    The data is sourced from a PostgreSQL database with hourly energy measurements.
    
    ### Features
    
    - Interactive date selection with session state persistence.
    - Summary metrics for prices, MQ.
    - Tabbed data tables for hourly figures and daily summaries.
    - Interactive Altair charts for time-series data, including a dedicated WESM balance chart.
    - Energy flow Sankey diagram to visualize sources, MQ, and destinations.
    
    ### WESM Interpretation
    
    - **WESM = Total_BCQ - Total_MQ**
    - **Negative WESM values** (MQ > BCQ) indicate a **Net Import** from WESM.
    - **Positive WESM values** (BCQ > MQ) indicate a **Net Export** to WESM.
    
    This interpretation is reflected in the WESM chart colors and the Sankey diagram logic.
    
    ### Need Help?
    
    For any issues or questions about this dashboard, please contact the system administrator.
    """)
    st.markdown("---")
    st.markdown(f"<p style='text-align: center;'>App Version 1.1 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
