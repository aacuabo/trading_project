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
    """Establishes and returns a SQLAlchemy database engine using Streamlit secrets."""
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
    """Fetches hourly MQ, BCQ, and Prices data for a selected date."""
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
        # This format [(value,)] is often required by database adapters for single positional parameters.
        df = pd.read_sql(query, engine, params=[(selected_date_str,)])

        # --- Convert 'Time' column to datetime objects for Altair ---
        # Altair works best with datetime objects for temporal axes.
        if not df.empty and 'Time' in df.columns:
            # Assuming "Time" column contains time strings like "HH:MM:SS" or similar
            # Combine with the selected date to create full datetime objects
            # Ensure the Time column is treated as string before concatenation
            try:
                # We need to handle potential None or NaN values in the 'Time' column before astype(str)
                df['Time'] = df['Time'].fillna('').astype(str)
                df['Time'] = pd.to_datetime(selected_date_str + ' ' + df['Time'], errors='coerce')
                # Drop rows where datetime conversion failed
                df.dropna(subset=['Time'], inplace=True)
            except Exception as e:
                st.error(f"Error converting 'Time' column to datetime: {e}. Please check the format of the 'Time' column in your database.")
                return pd.DataFrame() # Return empty DataFrame on error


        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Removed the fetch_generator_data function as requested


# --- STREAMLIT UI ---
st.title("📊 Daily Energy Trading Dashboard")

# Date input for user to select the date
selected_date = st.date_input("Select date", datetime.today())

# Format the selected date to 'YYYY-MM-DD' string for the SQL query
selected_date_str = selected_date.strftime('%Y-%m-%d')

# --- FETCH AND DISPLAY DATA ---
# Fetch the hourly data for the selected date
data = fetch_data(selected_date_str)


if not data.empty:
    # --- Display Daily Summary Metrics as Cards (Centered) ---
    st.subheader("Daily Summary Metrics")

    # Create columns for centering the metrics
    # Use 5 columns: 2 empty spacers on the sides, 3 for the metrics in the middle
    # The ratio [1, 1, 1, 1, 1] gives equal width, adjust as needed for centering.
    col_spacer_left, col1, col2, col3, col_spacer_right = st.columns([1, 1, 1, 1, 1])

    # Display Maximum Price in the first data column (col1)
    if "Prices" in data.columns:
        max_price = data["Prices"].max()
        col1.metric(label="Maximum Price (PHP/kWh)", value=f"{max_price:,.2f}") # Format for readability
    else:
        col1.warning("Max Price data not available.")

    # Display Average Price in the second data column (col2)
    if "Prices" in data.columns: # Check again for safety, though already checked for max
        avg_price = data["Prices"].mean()
        col2.metric(label="Average Price (PHP/kWh)", value=f"{avg_price:,.2f}") # Format for readability
    else:
        col2.warning("Avg Price data not available.")


    # Display Maximum Total MQ and corresponding time in the third data column (col3)
    if "Total_MQ" in data.columns and "Time" in data.columns and not data["Total_MQ"].empty:
        # Find the row with the maximum Total_MQ value
        max_mq_value = data["Total_MQ"].max()
        # Check if the column is all NaNs or empty, idxmax would raise an error
        if pd.notnull(max_mq_value) and not data["Total_MQ"].isnull().all(): # Also check if all values are null
             max_mq_row_index = data["Total_MQ"].idxmax()
             # Get the corresponding Time value from that row
             max_mq_time = data.loc[max_mq_row_index, "Time"]

             # Format the time for display
             if pd.api.types.is_datetime64_any_dtype(max_mq_time):
                  max_mq_time_str = max_mq_time.strftime("%H:%M")
             else:
                  # Handle cases where max_mq_time might not be a datetime object
                  max_mq_time_str = str(max_mq_time) # Display as string if not datetime

             col3.metric(label="Maximum Total MQ (kWh)", value=f"{max_mq_value:,.2f}")
             col3.write(f"at {max_mq_time_str}") # Display time below the metric
        else:
             col3.info("Total_MQ data is all zero/null or not applicable for max metric.")

    else:
         col3.warning("Max MQ or Time data not available or empty.")


    st.subheader("Hourly Summary")
    # Display the fetched data as a table
    st.dataframe(data)

    # --- PLOT DATA USING ALTAIR FOR INTERACTIVITY ---
    st.subheader("📈 Energy Metrics Over Time (Interactive)")

    # --- DEBUGGING: Check columns before melting ---
    # Use this to see the actual column names returned by the database.
    # You can comment this line out once you've confirmed the column names
    # and updated the columns_to_melt list below.
    # st.write("Columns in data DataFrame:", data.columns.tolist()) # Use .tolist() for clearer display

    # Melt the DataFrame for Altair - necessary for plotting multiple metrics
    # on the same or twin axes easily with color encoding.
    # Ensure 'Time' is a datetime type before melting
    if 'Time' in data.columns and pd.api.types.is_datetime64_any_dtype(data['Time']):

        # --- IMPORTANT: VERIFY AND UPDATE COLUMN NAMES HERE if necessary ---
        # The list below must contain the EXACT names of the columns in your
        # 'data' DataFrame that you want to plot (Total_MQ, Total_BCQ, Prices).
        # Check the output of the commented-out "st.write" line above if you
        # are still facing KeyError during melting.
        columns_to_melt = ["Total_MQ", "Total_BCQ", "Prices"] # <-- **VERIFY/UPDATE THESE NAMES if needed**

        # Check if all columns to melt exist in the DataFrame
        if all(col in data.columns for col in columns_to_melt):
            melted_data = data.melt(
                id_vars=["Time"], # Use id_vars to specify identifier columns
                value_vars=columns_to_melt, # Columns to unpivot
                var_name="Metric", # Name for the new column holding metric names
                value_name="Value" # Name for the new column holding metric values
            )

            # Debugging: Display melted data
            # st.subheader("Melted Data for Plotting")
            # st.dataframe(melted_data)
            # st.write("Melted Data Types:", melted_data.dtypes)


            # Create charts for the left y-axis (MQ and BCQ) - as lines
            chart_energy = alt.Chart(melted_data[melted_data["Metric"].isin(["Total_MQ", "Total_BCQ"])]).mark_line(point=True).encode(
                x=alt.X("Time", axis=alt.Axis(title="Time", format="%H:%M")), # Format time axis
                # --- Align zero for the energy axis - 'zero=True' goes inside alt.Scale() ---
                y=alt.Y("Value", title="Energy (kWh)", axis=alt.Axis(titleColor="tab:blue"), scale=alt.Scale(zero=True)),
                # --- Use a commonly supported colorblind-friendly scheme ---
                # --- Move legend to the bottom ---
                color=alt.Color("Metric", legend=alt.Legend(title="Metric", orient='bottom')), # Use 'category10' palette
                tooltip=[alt.Tooltip("Time", format="%Y-%m-%d %H:%M"), "Metric", "Value"] # Add tooltips with formatted time
            ).properties(
                 title="Energy Metrics" # Title for this layer's legend
            )

            # Create chart for the right y-axis (Prices) - as bars
            # --- Change the color of the price bars to apple green (#40B0A6 is a pleasant shade) ---
            chart_price = alt.Chart(melted_data[melted_data["Metric"] == "Prices"])\
                .mark_bar(color="#40B0A6")\
                .encode(
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
            ).interactive() # Add interactivity for zooming and panning

            # Display the chart in Streamlit
            st.altair_chart(combined_chart, use_container_width=True)
        else:
            # If columns are missing for plotting, display an informative warning
            missing_cols = [col for col in columns_to_melt if col not in data.columns]
            st.warning(f"Data fetched but required columns for plotting are missing: {missing_cols}. Check your database tables ('MQ_Hourly', 'BCQ_Hourly', 'Prices_Hourly') and SQL query result.")


    else:
        st.warning("Time column is not in the expected datetime format for plotting or data is empty after fetch.")


else:
    st.warning(f"No data available for selected date: {selected_date_str}.")

# Removed the section that displayed generator_data as requested
# if not generator_data.empty:
#     st.subheader("🔌 Generator BCQ Summary")
#     st.dataframe(generator_data)
