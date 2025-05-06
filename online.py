import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import altair as alt # Import Altair
# import seaborn # Keep this import if you intend to use other seaborn features

# Configure Matplotlib for Streamlit (only if still using Matplotlib for other plots)
# plt.rcParams["figure.figsize"] = (12, 6)
# try:
#     plt.style.use("seaborn-v0_8")
# except OSError:
#      plt.style.use("classic")


# --- DATABASE CONFIGURATION ---
def get_sqlalchemy_engine():
    try:
        # Read database credentials from secrets.toml
        user = st.secrets["database"]["user"]
        password = st.secrets["database"]["password"]
        host = st.secrets["database"]["host"]
        db = st.secrets["database"]["db"]
        # Ensure port is an integer
        port = int(st.secrets["database"]["port"])

        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
        return create_engine(url)
    except KeyError as e:
        st.error(f"Error loading database credentials: {e}. Make sure your .streamlit/secrets.toml file is correctly configured with [database] section and keys: user, password, host, db, port.")
        st.stop() # Stop the app if secrets are not found
    except ValueError:
         st.error("Error: Database port in secrets.toml is not a valid integer.")
         st.stop()
    except Exception as e:
        st.error(f"Error creating database engine: {e}")
        st.stop() # Stop the app if engine creation fails


# --- LOAD DATA ---
def fetch_data(selected_date_str):
    try:
        engine = get_sqlalchemy_engine() # Get the engine using the updated function
        query = """
            SELECT mq."Time", mq."Total_MQ", bcq."Total_BCQ", p."Prices"
            FROM "MQ_Hourly" AS mq
            JOIN "BCQ_Hourly" AS bcq ON mq."Date" = bcq."Date" AND mq."Time" = bcq."Time"
            JOIN "Prices_Hourly" AS p ON mq."Date" = p."Date" AND mq."Time" = p."Time"
            WHERE mq."Date" = %s
            ORDER BY
                mq."Time"
        """
        # --- Pass parameters as a list containing a tuple ---
        df = pd.read_sql(query, engine, params=[(selected_date_str,)])

        # --- Convert 'Time' column to datetime objects ---
        # Altair works best with datetime objects for temporal axes.
        if not df.empty and 'Time' in df.columns:
            # Assuming "Time" column contains time strings like "HH:MM:SS" or similar
            # Combine with the selected date to create full datetime objects
            # Ensure the Time column is treated as string before concatenation
            df['Time'] = pd.to_datetime(selected_date_str + ' ' + df['Time'].astype(str))


        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Removed the fetch_generator_data function


# --- STREAMLIT UI ---
st.title("📊 Daily Energy Trading Dashboard")
selected_date = st.date_input("Select date", datetime.today())

# Format the selected date to 'YYYY-MM-DD' string
selected_date_str = selected_date.strftime('%Y-%m-%d')

# --- FETCH AND DISPLAY DATA ---
# The fetching functions now call get_sqlalchemy_engine internally,
# which reads from st.secrets
data = fetch_data(selected_date_str)
# Removed the call to fetch_generator_data
# generator_data = fetch_generator_data(selected_date_str)


if not data.empty:
    st.subheader("Hourly Summary")
    st.dataframe(data) # Display fetched data for debugging

    # --- PLOT DATA USING ALTAIR FOR INTERACTIVITY ---
    st.subheader("📈 Energy Metrics Over Time (Interactive)")

    # Melt the DataFrame for Altair - necessary for plotting multiple metrics
    # on the same or twin axes easily with color encoding.
    # Ensure 'Time' is a datetime type before melting
    if 'Time' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Time']):
        melted_data = data.melt(
            "Time",
            ["Total_MQ", "Total_BCQ", "Prices"],
            "Metric",
            "Value"
        )

        # Create charts for the left y-axis (MQ and BCQ) - as lines
        chart_energy = alt.Chart(melted_data[melted_data["Metric"].isin(["Total_MQ", "Total_BCQ"])]).mark_line(point=True).encode(
            x=alt.X("Time", axis=alt.Axis(title="Time", format="%H:%M")), # Format time axis
            # --- Align zero for the energy axis - 'zero=True' goes inside alt.Scale() ---
            y=alt.Y("Value", title="Energy (kWh)", axis=alt.Axis(titleColor="tab:blue"), scale=alt.Scale(zero=True)),
            # --- Use a commonly supported colorblind-friendly scheme ---
            color=alt.Color("Metric", legend=alt.Legend(title="Metric"), scale=alt.Scale(scheme='category10')), # Use 'category10' palette
            tooltip=[alt.Tooltip("Time", format="%Y-%m-%d %H:%M"), "Metric", "Value"] # Add tooltips with formatted time
        ).properties(
             title="Energy Metrics" # Title for this layer's legend
        )

        # Create chart for the right y-axis (Prices) - as bars
        # --- FIX: Change the color of the price bars to apple green ---
        chart_price = alt.Chart(melted_data[melted_data["Metric"] == "Prices"]).mark_bar(color="#40B0A6").encode(
            x=alt.X("Time", axis=alt.Axis(title="")), # Empty title as it's shared
            # --- Align zero for the price axis - 'zero=True' goes inside alt.Scale() ---
            y=alt.Y("Value", title="Price (PHP/kWh)", axis=alt.Axis(titleColor="tab:red"), scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip("Time", format="%Y-%m-%d %H:%M"), "Metric", "Value"] # Add tooltips with formatted time
        ).properties(
             title="Prices" # Title for this layer's legend
        )

        # Combine the charts with independent y-axes
        # --- Reverse the layering order to put bars behind lines ---
        # List chart_price first to draw its bars at the bottom, then chart_energy lines on top.
        combined_chart = alt.layer(chart_price, chart_energy).resolve_scale(
            y='independent' # Allow y-axes to have different scales
        ).properties(
            title=f"Energy Metrics and Prices for {selected_date_str}"
        )


        # Display the chart in Streamlit
        st.altair_chart(combined_chart, use_container_width=True)
    else:
        st.warning("Time column is not in the expected datetime format for plotting or is empty.")


else:
    st.warning(f"No data available for selected date: {selected_date_str}.")

# Removed the section that displayed generator_data
# if not generator_data.empty:
#     st.subheader("🔌 Generator BCQ Summary")
#     st.dataframe(generator_data)
