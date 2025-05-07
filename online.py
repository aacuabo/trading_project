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
        # Changed INNER JOINs to LEFT JOINs to ensure we get MQ data even if other tables are missing data
        
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
                df['Time'] = pd.to_datetime(selected_date_str + ' ' + df['Time'], errors='coerce')
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
# Define mappings
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
def fetch_sankey_generator_contributions(selected_date_str: str, total_mq: float):
    """
    Creates dummy data for generator contributions that sum up to the Total MQ value.
    In a real implementation, this would fetch actual data from the database.
    
    Args:
        selected_date_str: The selected date as string
        total_mq: The total MQ value to distribute among generators
    
    Returns:
        Dictionary mapping generator short names to their contribution values
    """
    try:
        # Since Generator_Daily_Output table doesn't exist yet, 
        # we'll use dummy data that's proportional to the total MQ
        contributions = {}
        # Distribute the total MQ among generators with varying proportions
        # to make the visualization more interesting
        proportions = {
            'FDC': 0.22,
            'GNPK': 0.25,
            'PSALM': 0.15,
            'SEC': 0.18,
            'TSI': 0.12,
            'MPI': 0.08
        }
        
        # Apply some random variation to make it look more realistic
        for short_name, proportion in proportions.items():
            # Add ±10% random variation
            variation = 1.0 + (np.random.random() - 0.5) * 0.2
            contributions[short_name] = total_mq * proportion * variation
        
        # Normalize to ensure they sum to total_mq
        current_sum = sum(contributions.values())
        if current_sum > 0:
            scaling_factor = total_mq / current_sum
            for short_name in contributions:
                contributions[short_name] = contributions[short_name] * scaling_factor
        
        return contributions
    except Exception as e:
        st.warning(f"Error creating generator data: {e}")
        # Even more basic fallback
        contributions = {}
        for i, short_name in enumerate(GENERATOR_LONG_TO_SHORT_MAP.values()):
            contributions[short_name] = total_mq / len(GENERATOR_LONG_TO_SHORT_MAP)
        return contributions


@st.cache_data(ttl=600)
def fetch_sankey_destination_consumption(selected_date_str: str, total_mq_to_distribute: float):
    """
    Creates dummy data for destination consumption that sums up to the total MQ value.
    In a real implementation, this would fetch actual data from the database.
    
    Args:
        selected_date_str: The selected date as string
        total_mq_to_distribute: The total MQ value to distribute among destinations
    
    Returns:
        Dictionary mapping destination short names to their consumption values
    """
    try:
        # Since Destination_Daily_Consumption table doesn't exist yet,
        # we'll use dummy data that's proportional to the total MQ
        consumption = {}
        
        # Define proportions for each destination
        proportions = {
            'M1/M6/M8': 0.18,
            'M2': 0.12,
            'M3': 0.15,
            'M4': 0.11,
            'M5': 0.14,
            'M7': 0.09,
            'M9': 0.13,
            'KIDCSCV01_DEL': 0.04,
            'KIDCSCV02_DEL': 0.04
        }
        
        # Apply some random variation to make it look more realistic
        for short_name, proportion in proportions.items():
            # Add ±10% random variation
            variation = 1.0 + (np.random.random() - 0.5) * 0.2
            consumption[short_name] = total_mq_to_distribute * proportion * variation
        
        # Normalize to ensure they sum to total_mq_to_distribute
        current_sum = sum(consumption.values())
        if current_sum > 0:
            scaling_factor = total_mq_to_distribute / current_sum
            for short_name in consumption:
                consumption[short_name] = consumption[short_name] * scaling_factor
                
        return consumption
    except Exception as e:
        st.warning(f"Error creating destination data: {e}")
        # Even more basic fallback
        consumption = {}
        num_destinations = len(DESTINATION_LONG_TO_SHORT_MAP)
        if num_destinations > 0:
            for short_name in DESTINATION_LONG_TO_SHORT_MAP.values():
                consumption[short_name] = total_mq_to_distribute / num_destinations
                
        return consumption


def create_sankey_chart(data: pd.DataFrame, selected_date_str: str):
    """Creates a Sankey chart showing energy flow from generators through MQ to destinations.
    
    This implementation handles WESM as a source (either positive or negative contribution)
    and adds percentage values to all labels.
    """
    if 'Total_MQ' not in data.columns or 'WESM' not in data.columns:
        st.warning("Cannot generate Sankey: Required columns missing")
        return None
        
    if data['Total_MQ'].isnull().all() or not pd.api.types.is_numeric_dtype(data['Total_MQ']):
        st.warning("Cannot generate Sankey: MQ data invalid")
        return None
    
    # Calculate total MQ
    total_mq_sum = data['Total_MQ'].sum()
    if pd.isna(total_mq_sum) or total_mq_sum == 0:
        st.info(f"Total MQ is zero or N/A for {selected_date_str}. Cannot generate Sankey chart.")
        return None
        
    # Calculate WESM contribution
    wesm_daily_sum = data['WESM'].sum()
    
    # Calculate total generation needed (MQ minus WESM if WESM is negative, or just MQ if WESM is positive)
    if wesm_daily_sum > 0:
        # If WESM is negative (we're exporting), we need to generate more than MQ
        total_generation_needed = total_mq_sum - wesm_daily_sum  # Note: subtracting a negative adds to the total
    else:
        # If WESM is positive (we're importing), WESM contributes to the MQ total
        total_generation_needed = total_mq_sum - wesm_daily_sum  # We need to generate less than MQ
    
    # Initialize Sankey data structures
    sankey_node_labels = []
    node_indices = {}  # Map label to index
    sankey_sources_indices = []
    sankey_targets_indices = []
    sankey_values = []
    node_colors = []  # Node colors
    
    # Helper function to add nodes
    def add_node(label, color="grey"):
        if label not in node_indices:
            node_indices[label] = len(sankey_node_labels)
            sankey_node_labels.append(label)
            node_colors.append(color)
        return node_indices[label]
    
    # 1. Middle Node: Total MQ
    middle_node_label = f"Total Daily MQ ({total_mq_sum:,.0f} kWh)"
    middle_node_idx = add_node(middle_node_label, "orange")
    
    # 2. Source Nodes (Generators & WESM)
    
    # Get generator contributions based on what we need to generate
    if total_generation_needed > 0:
        generator_contributions = fetch_sankey_generator_contributions(selected_date_str, total_generation_needed)
    else:
        # In case total_generation_needed is negative or zero (shouldn't happen in practice)
        generator_contributions = fetch_sankey_generator_contributions(selected_date_str, total_mq_sum)
    
    # Calculate total input for percentage calculations
    total_input = sum(generator_contributions.values())
    if wesm_daily_sum > 0:
        total_input += wesm_daily_sum
    
    # Add generator nodes with percentages
    for short_name, value in generator_contributions.items():
        if value > 0:
            percentage = (value / total_input) * 100
            gen_node_label = f"{short_name} ({value:,.0f} kWh, {percentage:.1f}%)"
            gen_node_idx = add_node(gen_node_label, "blue")
            sankey_sources_indices.append(gen_node_idx)
            sankey_targets_indices.append(middle_node_idx)
            sankey_values.append(value)
    
    # WESM Contribution - Always handle as a source
    if wesm_daily_sum != 0:  # If WESM is non-zero
        # For WESM as source (positive = import)
        if wesm_daily_sum > 0:
            percentage = (wesm_daily_sum / total_input) * 100
            wesm_label = f"WESM Import ({wesm_daily_sum:,.0f} kWh, {percentage:.1f}%)"
            wesm_node_idx = add_node(wesm_label, "red")
            sankey_sources_indices.append(wesm_node_idx)
            sankey_targets_indices.append(middle_node_idx)
            sankey_values.append(wesm_daily_sum)
        else:  # For WESM as source but negative value (export)
            wesm_export = abs(wesm_daily_sum)
            percentage = (wesm_export / total_input) * 100 if total_input > 0 else 0
            wesm_label = f"WESM Export ({wesm_export:,.0f} kWh, {percentage:.1f}%)"
            wesm_node_idx = add_node(wesm_label, "purple")
            # For export, add this as a destination (going from MQ to WESM)
            sankey_sources_indices.append(middle_node_idx)
            sankey_targets_indices.append(wesm_node_idx)
            sankey_values.append(wesm_export)
    
    # 3. Destination Nodes
    # Calculate MQ for consumer destinations (should be = total_mq_sum)
    destination_consumptions = fetch_sankey_destination_consumption(
        selected_date_str, 
        total_mq_sum if wesm_daily_sum >= 0 else (total_mq_sum - wesm_daily_sum)
    )
    
    # Calculate total output for percentage calculations
    total_output = sum(destination_consumptions.values())
    if wesm_daily_sum < 0:
        total_output += abs(wesm_daily_sum)
    
    # Add destination nodes with percentages
    for short_name, value in destination_consumptions.items():
        if value > 0:
            percentage = (value / total_output) * 100
            dest_node_label = f"{short_name} ({value:,.0f} kWh, {percentage:.1f}%)"
            dest_node_idx = add_node(dest_node_label, "green")
            sankey_sources_indices.append(middle_node_idx)
            sankey_targets_indices.append(dest_node_idx)
            sankey_values.append(value)
    
    # Check for data sufficiency
    if not sankey_values or sum(sankey_values) == 0:
        st.info(f"Not enough data to draw Sankey chart for {selected_date_str}.")
        return None
    
    # Create the Sankey figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=25,
            thickness=20,
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
        font_size=10,
        height=600
    )
    
    return fig


# --- STREAMLIT UI ---
def main():
    st.title("📊 Daily Energy Trading Dashboard")
    
    # Add sidebar for navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "About"])
    
    if page == "About":
        show_about_page()
    else:
        show_dashboard()


def show_dashboard():
    """Shows the main dashboard page"""
    spacer_left, main_content, spacer_right = st.columns([0.5, 4, 0.5])
    
    with main_content:
        available_dates = fetch_available_dates()
        
        if not available_dates:
            st.error("No available dates found. Check database connection and data.")
            st.stop()
        
        min_available_date = min(available_dates)
        max_available_date = max(available_dates)
        default_date = max_available_date  # Default to the latest available date
        
        selected_date = st.date_input(
            "Select date",
            value=default_date,
            min_value=min_available_date,
            max_value=max_available_date,
        )
        
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
            max_price = data["Prices"].max(skipna=True)
            avg_price = data["Prices"].mean(skipna=True)
            min_price = data["Prices"].min(skipna=True)
            
            col1.metric(label="Max Price (PHP/kWh)", value=f"{max_price:,.2f}" if pd.notna(max_price) else "N/A")
            col2.metric(label="Avg Price (PHP/kWh)", value=f"{avg_price:,.2f}" if pd.notna(avg_price) else "N/A")
            col3.metric(label="Min Price (PHP/kWh)", value=f"{min_price:,.2f}" if pd.notna(min_price) else "N/A")
        else:
            col1.warning("Price data not available")
            col2.warning("Avg Price data not available")
            col3.warning("Min Price data not available")
        
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
            col4.warning("Max MQ data not available")
        
        # Data table with tabs
        st.subheader("Data Tables")
        tabs = st.tabs(["Hourly Data", "Daily Summary"])
        
        with tabs[0]:
            if not data.empty:
                # Format datetime for display
                display_data = data.copy()
                if 'Time' in display_data.columns and pd.api.types.is_datetime64_any_dtype(display_data['Time']):
                    display_data['Time'] = display_data['Time'].dt.strftime('%H:%M')
                
                st.dataframe(display_data, use_container_width=True)
            else:
                st.info("No hourly data available")
                
        with tabs[1]:
            if not data.empty:
                # Create daily summary 
                summary = {
                    "Total MQ (kWh)": data["Total_MQ"].sum() if "Total_MQ" in data.columns else "N/A",
                    "Total BCQ (kWh)": data["Total_BCQ"].sum() if "Total_BCQ" in data.columns else "N/A",
                    "Net WESM (kWh)": data["WESM"].sum() if "WESM" in data.columns else "N/A",
                    "Average Price (PHP/kWh)": data["Prices"].mean() if "Prices" in data.columns else "N/A"
                }
                summary_df = pd.DataFrame([summary])
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.info("No data available for summary")
        
        # Charts
        st.subheader("📈 Energy Metrics Over Time (Interactive)")
        if 'Time' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Time']):
            columns_to_melt = ["Total_MQ", "Total_BCQ", "Prices", "WESM"]
            existing_cols_to_melt = [col for col in columns_to_melt if col in data.columns and not data[col].isnull().all()]
            
            if existing_cols_to_melt:
                tabs = st.tabs(["Energy & Prices", "WESM Balance"])
                
                with tabs[0]:
                    melted_data = data.melt(
                        id_vars=["Time"],
                        value_vars=[col for col in ["Total_MQ", "Total_BCQ", "Prices"] if col in existing_cols_to_melt],
                        var_name="Metric",
                        value_name="Value"
                    )
                    melted_data.dropna(subset=['Value'], inplace=True)
                    
                    # Chart for MQ and BCQ
                    energy_metrics = [m for m in ["Total_MQ", "Total_BCQ"] if m in existing_cols_to_melt]
                    if energy_metrics:
                        chart_energy = alt.Chart(melted_data[melted_data["Metric"].isin(energy_metrics)]).mark_line(point=True).encode(
                            x=alt.X("Time:T", axis=alt.Axis(title="Time", format="%H:%M")),
                            y=alt.Y("Value:Q", title="Energy (kWh)", axis=alt.Axis(titleColor="#1A85FF"), scale=alt.Scale(zero=True)),
                            color=alt.Color("Metric:N", legend=alt.Legend(title="Metric", orient='bottom'),
                                           scale=alt.Scale(domain=['Total_MQ', 'Total_BCQ'], range=['#FFC20A', '#1A85FF'])),
                            tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", "Value:Q"]
                        ).properties(title="Energy Metrics")
                    else:
                        chart_energy = alt.Chart(pd.DataFrame()).mark_text()
                    
                    # Chart for Prices
                    if "Prices" in existing_cols_to_melt:
                        chart_price = alt.Chart(melted_data[melted_data["Metric"] == "Prices"]).mark_bar(color="#40B0A6").encode(
                            x=alt.X("Time:T", axis=alt.Axis(title="Time", format="%H:%M")),
                            y=alt.Y("Value:Q", title="Price (PHP/kWh)", axis=alt.Axis(titleColor="#40B0A6"), scale=alt.Scale(zero=True)),
                            tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "Metric:N", "Value:Q"]
                        ).properties(title="Prices")
                    else:
                        chart_price = alt.Chart(pd.DataFrame()).mark_text()
                    
                    # Combine charts if both exist
                    if energy_metrics and "Prices" in existing_cols_to_melt:
                        combined_chart = alt.layer(chart_price, chart_energy).resolve_scale(y='independent').properties(
                            title=f"Energy Metrics and Prices for {selected_date_str}"
                        ).interactive()
                    elif energy_metrics:
                        combined_chart = chart_energy.properties(title=f"Energy Metrics for {selected_date_str}").interactive()
                    elif "Prices" in existing_cols_to_melt:
                        combined_chart = chart_price.properties(title=f"Prices for {selected_date_str}").interactive()
                    else:
                        combined_chart = alt.Chart(pd.DataFrame()).mark_text(text="No data to plot for selected metrics.").properties(title="No Data")
                    
                    st.altair_chart(combined_chart, use_container_width=True)
                
                with tabs[1]:
                    if "WESM" in existing_cols_to_melt:
                        wesm_data = data[["Time", "WESM"]].copy()
                        wesm_data.dropna(subset=["WESM"], inplace=True)
                        
                        if not wesm_data.empty:
                            # Create WESM bar chart
                            wesm_chart = alt.Chart(wesm_data).mark_bar().encode(
                                x=alt.X("Time:T", axis=alt.Axis(title="Time", format="%H:%M")),
                                y=alt.Y("WESM:Q", title="WESM Balance (kWh)"),
                                color=alt.condition(
                                    alt.datum.WESM > 0,
                                    alt.value("#ff9900"),  # orange for positive (net import)
                                    alt.value("#4c78a8")   # blue for negative (net export)
                                ),
                                tooltip=[alt.Tooltip("Time:T", format="%Y-%m-%d %H:%M"), "WESM:Q"]
                            ).properties(
                                title=f"WESM Balance for {selected_date_str}",
                                height=400
                            ).interactive()
                            
                            st.altair_chart(wesm_chart, use_container_width=True)
                            
                            # Add summary text
                            wesm_total = wesm_data["WESM"].sum()
                            wesm_status = "Net Import from WESM" if wesm_total > 0 else "Net Export to WESM"
                            st.info(f"Daily WESM Balance: {wesm_total:,.2f} kWh ({wesm_status})")
                        else:
                            st.info("No WESM data available")
                    else:
                        st.info("WESM data not available")
            else:
                st.warning(f"Required columns for plotting are missing or all null in data for {selected_date_str}.")
        else:
            st.warning("Time column not in expected datetime format or data is empty.")
        
        # Sankey Chart
        st.subheader("⚡ Daily Energy Flow Sankey Chart")
        sankey_fig = create_sankey_chart(data, selected_date_str)
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
    - Generator contributions
    - Energy flow visualization
    
    The data is sourced from a PostgreSQL database with hourly energy measurements.
    
    ### Features
    
    - Interactive date selection
    - Summary metrics
    - Hour-by-hour data tables
    - Interactive charts
    - Energy flow Sankey diagram
    
    ### Need Help?
    
    For any issues or questions about this dashboard, please contact the system administrator.
    """)


if __name__ == "__main__":
    main()
