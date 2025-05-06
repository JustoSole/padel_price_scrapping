import streamlit as st
import gspread
import pandas as pd
import plotly.express as px

# --- Constants ---
# Google Sheets Configuration (Copied from test_conection_gsheet.py)

# --- Google Sheets Connection ---
@st.cache_resource # Cache the connection resource
def get_gspread_client():
    """Initializes and returns the gspread client."""
    try:
        # Reads credentials dictionary directly from st.secrets
        # Assumes secrets are stored under a top-level key like [google_sheets] in secrets.toml
        # Ensure your secrets file has a structure like:
        # [google_sheets]
        # type = "service_account"
        # ... (all other fields from SERVICE_ACCOUNT_INFO)
        creds = st.secrets["google_sheets"]
        # Ensure private_key is read correctly (Streamlit might handle newlines automatically)
        # If you encounter key errors, double-check the secrets.toml format.
        gc = gspread.service_account_from_dict(creds)
        return gc
    except KeyError:
        st.error("❌ Google Sheets credentials (`[google_sheets]`) not found in Streamlit Secrets. Please configure secrets.toml locally or add secrets in the Streamlit Cloud app settings.")
        return None

@st.cache_data(ttl=600) # Cache data for 10 minutes
def fetch_data(_gc: gspread.Client):
    """Fetches data from the Google Sheet and returns a Pandas DataFrame."""
    if _gc is None:
        return pd.DataFrame() # Return empty DataFrame if connection failed
    try:
        # Get sheet key and name from secrets
        # Assumes keys like 'spreadsheet_key' and 'worksheet_name' exist under [google_sheets]
        try:
            spreadsheet_key = st.secrets["google_sheets"]["spreadsheet_key"]
            worksheet_name = st.secrets["google_sheets"]["worksheet_name"]
        except KeyError as e:
            st.error(f"❌ Secret Key Error: Missing '{e}' under `[google_sheets]` in Streamlit Secrets.")
            return pd.DataFrame()

        sh = _gc.open_by_key(spreadsheet_key)
        # Attempt to get worksheet by name, fall back to index 0 if not found
        try:
            worksheet = sh.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            st.warning(f"⚠️ Worksheet '{worksheet_name}' not found. Reading the first sheet.")
            worksheet = sh.get_worksheet(0)

        records = worksheet.get_all_records(head=1) # Assumes header is on row 1
        df = pd.DataFrame(records)
        # Basic data cleaning: Replace empty strings with NaN, attempt numeric conversion
        df.replace('', pd.NA, inplace=True)
        for col in df.columns:
            # Identify numeric and percentage columns
            numeric_cols = ['LOWEST PRICE (B2C)', 'MAP', 'RC Suggested Price', 'Unit Cost',
                            'Total Stock (QTY)', 'total Stock sold (last 12 M) (Qty)', 'Stock Days on Hand',
                            'RACKET CENTRAL', 'JUST PADDLES', 'PADEL USA', 'CASAS PADEL', 'FROMUTH',
                            'PICKLEBALL CENTRAL'
                           ]
            percentage_cols = ['RC Marginal Contribution (%)', 'B2B Marginal Contribution (%)']
            
            if col in numeric_cols or col in percentage_cols:
                 # Clean currency/percentage symbols if present before converting
                 if df[col].dtype == 'object':
                    # Store original Series temporarily
                    original_series = df[col].astype(str)
                    # Remove symbols
                    cleaned_series = original_series.str.replace(r'[$,%]', '', regex=True).str.strip()
                    # Handle potential empty strings after cleaning -> NaN
                    cleaned_series.replace('', pd.NA, inplace=True)
                    # Convert to numeric
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    # If it's a percentage column, divide by 100
                    if col in percentage_cols:
                         numeric_series = numeric_series / 100.0
                         
                    df[col] = numeric_series
            elif pd.api.types.is_numeric_dtype(df[col]):
                # If already numeric, check if it's a percentage column that might need division
                # (This handles cases where Sheets might export % as numbers > 1)
                if col in percentage_cols and df[col].abs().max() > 1: # Heuristic: if max abs value > 1, assume it needs division
                    st.info(f"Column '{col}' seems to be numeric percentage > 1, dividing by 100.")
                    df[col] = df[col] / 100.0
        # else: keep non-numeric, non-percentage columns as they are
                
        return df
    except gspread.exceptions.APIError as e:
        st.error(f"❌ Google API Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return pd.DataFrame()

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Pricing & Competition Dashboard")
st.title("📊 Pricing & Competition Dashboard")
st.caption("This dashboard provides insights into product pricing, stock levels, margins, and competitor pricing based on data from Google Sheets.")

# Fetch data
gc = get_gspread_client()
df_original = fetch_data(gc)

if df_original.empty:
    st.warning("Could not load data from Google Sheets.")
else:
    st.success(f"✅ Data loaded successfully from Google Sheet!")

    df = df_original.copy() # Work with a copy for filtering

    # --- Define global constants for columns ---
    ALL_COMPETITOR_COLUMNS = sorted(['JUST PADDLES', 'PADEL USA', 'CASAS PADEL', 'FROMUTH', 'PICKLEBALL CENTRAL'])
    RC_PRICE_COLUMN = 'RACKET CENTRAL'

    # --- Sidebar Filters ---
    st.sidebar.header("Filters")

    # EAN Filter
    ean_search = st.sidebar.text_input("Search by EAN:", placeholder="Enter EAN...")

    # SKU Filter
    sku_search = st.sidebar.text_input("Search by SKU:", placeholder="Enter SKU...")

    # Text Search Filter
    search_term = st.sidebar.text_input("Search by Item Name:", placeholder="Enter item name...")

    # Categorical Filters
    brands = st.sidebar.multiselect(
        "Select Brand(s):",
        options=sorted(df["BRAND"].unique()), # Sort options
    )
    # Add 'All' option helper
    def get_unique_options(column_name):
        options = sorted(df[column_name].unique())
        return options

    sports_options = get_unique_options("SPORT")
    sports = st.sidebar.multiselect(
        "Select Sport(s):",
        options=sports_options,
    )

    type1_options = get_unique_options("TYPE1")
    type1s = st.sidebar.multiselect(
        "Select Type1(s):",
        options=type1_options,
    )

    # New Competitor Filter for Co-occurrence
    selected_competitors_for_coccurrence_filter = st.sidebar.multiselect(
        "Show Products Priced by RC & Selected Competitor(s):",
        options=ALL_COMPETITOR_COLUMNS,
        help="Filters data to show only products where Racket Central AND at least one of the selected competitors have a price."
    )

    # Apply Categorical Filters
    # Handle empty selections - if empty, select all
    if not brands: brands = df["BRAND"].unique()
    if not sports: sports = df["SPORT"].unique()
    if not type1s: type1s = df["TYPE1"].unique()

    df_filtered = df[
        df["BRAND"].isin(brands) &
        df["SPORT"].isin(sports) &
        df["TYPE1"].isin(type1s)
    ]

    # Numerical Filters (using sliders, handle potential NaNs and empty filtered data)
    def add_range_slider(column_name):
        if df_filtered.empty or df_filtered[column_name].isnull().all():
            st.sidebar.warning(f"No data or only missing values for '{column_name}' in current selection.")
            return None, None

        min_val = df_filtered[column_name].min()
        max_val = df_filtered[column_name].max()

        # Use slightly wider range for slider if min/max are the same or very close
        if pd.isna(min_val) or pd.isna(max_val):
             st.sidebar.warning(f"Cannot create slider for '{column_name}' due to missing values.")
             return None, None # Return None to indicate filter shouldn't be applied

        # Determine step and range
        step = 1
        if isinstance(min_val, float) or isinstance(max_val, float):
            diff = max_val - min_val
            if diff < 1 and diff > 0:
                step = 0.01
            elif diff == 0:
                 step = 0.1 # For single float value
            else:
                 step = (max_val - min_val) / 100
                 step = max(step, 0.01) # Ensure a minimum step

        # Handle case where min == max
        if min_val == max_val:
             range_min = min_val - step * 5 # Adjust range slightly
             range_max = max_val + step * 5
        else:
             range_min = min_val
             range_max = max_val

        # Ensure range_min < range_max for slider
        if range_min >= range_max:
             range_max = range_min + step

        try:
            selected_range = st.sidebar.slider(
                f"Select {column_name} Range:",
                    min_value=float(range_min),
                max_value=float(range_max),
                value=(float(range_min), float(range_max)),
                    step=float(step) if step > 0 else None # step can't be 0
            )
            return selected_range
        except Exception as e:
            
            st.sidebar.error(f"Error creating slider for {column_name}: {e}")
            return None, None

    # price_range = add_range_slider("LOWEST PRICE (B2C)")
    # map_range = add_range_slider("MAP")
    # rc_price_range = add_range_slider("RC Suggested Price")
    # cost_range = add_range_slider("Unit Cost")
    # stock_range = add_range_slider("Total Stock (QTY)")
    # stock_days_range = add_range_slider("Stock Days on Hand")

    # --- Apply Numerical Filters ---
    # Store original filtered df before numerical filters for comparison/KPIs if needed
    df_pre_numeric_filter = df_filtered.copy()

    # Apply filters safely, checking if range is valid
    # Use a temporary variable to chain filters
    df_final_filtered = df_filtered.copy()

    # if price_range and price_range[0] is not None and price_range[1] is not None:
    #      df_final_filtered = df_final_filtered[df_final_filtered["LOWEST PRICE (B2C)"].between(price_range[0], price_range[1], inclusive='both') | df_final_filtered["LOWEST PRICE (B2C)"].isnull()]
    # if map_range and map_range[0] is not None and map_range[1] is not None:
    #      df_final_filtered = df_final_filtered[df_final_filtered["MAP"].between(map_range[0], map_range[1], inclusive='both') | df_final_filtered["MAP"].isnull()]
    # if rc_price_range and rc_price_range[0] is not None and rc_price_range[1] is not None:
    #     df_final_filtered = df_final_filtered[df_final_filtered["RC Suggested Price"].between(rc_price_range[0], rc_price_range[1], inclusive='both') | df_final_filtered["RC Suggested Price"].isnull()]
    # if cost_range and cost_range[0] is not None and cost_range[1] is not None:
    #     df_final_filtered = df_final_filtered[df_final_filtered["Unit Cost"].between(cost_range[0], cost_range[1], inclusive='both') | df_final_filtered["Unit Cost"].isnull()]
    # if stock_range and stock_range[0] is not None and stock_range[1] is not None:
    #     df_final_filtered = df_final_filtered[df_final_filtered["Total Stock (QTY)"].between(stock_range[0], stock_range[1], inclusive='both') | df_final_filtered["Total Stock (QTY)"].isnull()]
    # if stock_days_range and stock_days_range[0] is not None and stock_days_range[1] is not None:
    #     df_final_filtered = df_final_filtered[df_final_filtered["Stock Days on Hand"].between(stock_days_range[0], stock_days_range[1], inclusive='both') | df_final_filtered["Stock Days on Hand"].isnull()]

    # --- Apply Text Search Filters ---
    # Apply EAN Filter (assuming EAN column exists and is string/object type)
    if ean_search and 'EAN' in df_final_filtered.columns:
        # Convert EAN column to string to ensure consistent comparison
        df_final_filtered = df_final_filtered[df_final_filtered['EAN'].astype(str).str.contains(ean_search, case=False, na=False)]

    # Apply SKU Filter (assuming SKU column exists)
    if sku_search and 'SKU' in df_final_filtered.columns:
        df_final_filtered = df_final_filtered[df_final_filtered['SKU'].astype(str).str.contains(sku_search, case=False, na=False)]

    # Apply Item Name Filter
    if search_term:
         df_final_filtered = df_final_filtered[df_final_filtered['ITEM NAME'].str.contains(search_term, case=False, na=False)]

    # Apply Co-occurrence Filter based on selected competitors
    if selected_competitors_for_coccurrence_filter:
        rows_to_keep_coccurrence = []
        for index, row in df_final_filtered.iterrows():
            rc_price_is_valid = pd.notna(pd.to_numeric(row.get(RC_PRICE_COLUMN), errors='coerce'))
            if not rc_price_is_valid:
                continue

            at_least_one_selected_competitor_has_price = False
            for comp_col in selected_competitors_for_coccurrence_filter:
                if comp_col in row and pd.notna(pd.to_numeric(row[comp_col], errors='coerce')):
                    at_least_one_selected_competitor_has_price = True
                    break
            
            if at_least_one_selected_competitor_has_price:
                rows_to_keep_coccurrence.append(index)
        
        df_final_filtered = df_final_filtered.loc[rows_to_keep_coccurrence]

    # --- Calculate Price Comparison KPIs ---
    total_comparable_products_kpi = 0
    rc_is_lowest_count_kpi = 0
    rc_is_higher_count_kpi = 0
    percentage_lowest_kpi = 0.0
    percentage_higher_kpi = 0.0
    available_competitors_for_kpi_count = 0

    if not df_final_filtered.empty:
        # Determine which competitors to consider for these main KPIs
        competitors_to_consider_for_main_kpi = []
        if selected_competitors_for_coccurrence_filter: # User has specifically filtered by competitors
            competitors_to_consider_for_main_kpi = [
                col for col in selected_competitors_for_coccurrence_filter
                if col in df_final_filtered.columns and not df_final_filtered[col].isnull().all()
            ]
        else: # No specific competitor filter, so consider all available in the (already filtered) data
            competitors_to_consider_for_main_kpi = [
                col for col in ALL_COMPETITOR_COLUMNS
                if col in df_final_filtered.columns and not df_final_filtered[col].isnull().all()
            ]
        available_competitors_for_kpi_count = len(competitors_to_consider_for_main_kpi)

        if RC_PRICE_COLUMN in df_final_filtered.columns and not df_final_filtered[RC_PRICE_COLUMN].isnull().all() and competitors_to_consider_for_main_kpi:
            for index, row in df_final_filtered.iterrows(): # df_final_filtered is already pre-filtered by co-occurrence if selected
                rc_price_val = pd.to_numeric(row[RC_PRICE_COLUMN], errors='coerce')
                # RC price validity is already ensured if co-occurrence filter was active. 
                # If not active, we might have rows with RC NaN, but the next check handles it.
                if pd.isna(rc_price_val):
                    continue

                current_competitor_prices_for_kpi = []
                for comp_col in competitors_to_consider_for_main_kpi:
                    comp_price = pd.to_numeric(row.get(comp_col), errors='coerce') # Use .get() for safety
                    if pd.notna(comp_price):
                        current_competitor_prices_for_kpi.append(comp_price)

                if not current_competitor_prices_for_kpi: # No prices from *considered* competitors for this row
                    continue 

                total_comparable_products_kpi += 1 # This row is comparable based on RC and *considered* competitors

                min_comp_price = min(current_competitor_prices_for_kpi)
                if rc_price_val <= min_comp_price:
                    rc_is_lowest_count_kpi += 1
                
                is_higher_this_row = False
                for comp_price_val_check in current_competitor_prices_for_kpi:
                    if rc_price_val > comp_price_val_check:
                        is_higher_this_row = True
                        break
                if is_higher_this_row:
                    rc_is_higher_count_kpi += 1
            
            if total_comparable_products_kpi > 0:
                percentage_lowest_kpi = (rc_is_lowest_count_kpi / total_comparable_products_kpi) * 100
                percentage_higher_kpi = (rc_is_higher_count_kpi / total_comparable_products_kpi) * 100

    # --- Main Display ---
    st.header("📊 Dashboard Overview")

    if df_final_filtered.empty:
        st.warning("No data matches the selected filters.")
    else:
        # --- KPIs ---
        st.subheader("Key Performance Indicators")
        st.caption("High-level metrics calculated from the currently filtered data.")
        kpi_cols = st.columns(4)
        total_products = len(df_final_filtered)
        avg_b2c_price = df_final_filtered['LOWEST PRICE (B2C)'].mean()
        avg_rc_price = df_final_filtered['RACKET CENTRAL'].mean()
        avg_cost = df_final_filtered['Unit Cost'].mean()
        avg_stock_days = df_final_filtered['Stock Days on Hand'].mean()
        total_stock_qty = df_final_filtered['Total Stock (QTY)'].sum()
        # Calculate total stock value safely, handling NaNs
        df_final_filtered['Stock Value'] = df_final_filtered['Total Stock (QTY)'] * df_final_filtered['Unit Cost']
        total_stock_value = df_final_filtered['Stock Value'].sum()
        avg_margin_b2c = df_final_filtered['RC Marginal Contribution (%)'].mean()

        kpi_cols[0].metric("Total Products", f"{total_products:,}")
        kpi_cols[1].metric("Avg. B2C Price", f"${avg_b2c_price:,.2f}" if pd.notna(avg_b2c_price) else "N/A")
        kpi_cols[2].metric("Avg. RC Price", f"${avg_rc_price:,.2f}" if pd.notna(avg_rc_price) else "N/A")
        kpi_cols[3].metric("Avg. Unit Cost", f"${avg_cost:,.2f}" if pd.notna(avg_cost) else "N/A")

        kpi_cols = st.columns(4) # New row for more KPIs
        kpi_cols[0].metric("Total Stock Qty", f"{total_stock_qty:,.0f}" if pd.notna(total_stock_qty) else "N/A")
        kpi_cols[1].metric("Total Stock Value", f"${total_stock_value:,.2f}" if pd.notna(total_stock_value) else "N/A")
        kpi_cols[2].metric("Avg. Stock Days", f"{avg_stock_days:,.1f}" if pd.notna(avg_stock_days) else "N/A")
        kpi_cols[3].metric("Avg. RC Margin %", f"{avg_margin_b2c:.1%}" if pd.notna(avg_margin_b2c) else "N/A")

        # Calculate and add Total Sales (12M) KPI
        # Use 'RACKET CENTRAL' as the price column
        df_final_filtered['Sales Value (12M)'] = df_final_filtered['total Stock sold (last 12 M) (Qty)'] * df_final_filtered['RACKET CENTRAL'] # Corrected column name
        total_sales_12m = df_final_filtered['Sales Value (12M)'].sum()

        # Add a new row for the sales KPI
        kpi_cols_row3 = st.columns(4) # Use 4 columns for alignment, place KPI in the first
        kpi_cols_row3[0].metric("Total Sales (12M)", f"${total_sales_12m:,.2f}" if pd.notna(total_sales_12m) else "N/A")

        # Add Price Competitiveness KPIs
        st.subheader("Price Competitiveness")
        kpi_comp_row = st.columns(3)
        kpi_comp_row[0].metric(
            "Total Comparable Products",
            f"{total_comparable_products_kpi}"
        )
        kpi_comp_row[1].metric(
            f"RC Lowest Priced",
            f"{rc_is_lowest_count_kpi} ({percentage_lowest_kpi:.1f}%)"
        )
        kpi_comp_row[2].metric(
            f"RC Higher Priced",
            f"{rc_is_higher_count_kpi} ({percentage_higher_kpi:.1f}%)"
        )
        if selected_competitors_for_coccurrence_filter:
            st.caption(f"Competitiveness KPIs based on products also priced by {available_competitors_for_kpi_count} selected competitor(s).")
        elif available_competitors_for_kpi_count > 0:
            st.caption(f"Competitiveness KPIs based on {available_competitors_for_kpi_count} active competitor(s) in the filtered data.")
        else:
            st.caption("No active competitor data in the filtered selection for these KPIs.")

        st.divider()

        # --- Data Table and Export ---
        st.subheader("Filtered Data")
        st.caption("Detailed product data based on the selected filters. Use the download button to export.")

        # --- Add 'RC Cheaper?' column to df_final_filtered for display and potential export ---
        if not df_final_filtered.empty:
            rc_cheaper_status_list = []
            for index, row in df_final_filtered.iterrows():
                rc_price_val = pd.to_numeric(row.get(RC_PRICE_COLUMN), errors='coerce')
                status = "N/A"  # Default status

                if pd.notna(rc_price_val):
                    competitor_prices_for_row = []
                    for comp_col in ALL_COMPETITOR_COLUMNS:
                        if comp_col in row:
                            comp_price = pd.to_numeric(row[comp_col], errors='coerce')
                            if pd.notna(comp_price):
                                competitor_prices_for_row.append(comp_price)
                    
                    if competitor_prices_for_row:  # If there are competitor prices
                        min_competitor_price = min(competitor_prices_for_row)
                        if rc_price_val < min_competitor_price:
                            status = "Cheaper"
                        elif rc_price_val == min_competitor_price:
                            status = "Equal"
                        else: # rc_price_val > min_competitor_price
                            status = "Higher"
                    # else: RC price exists, but no competitor prices, status remains "N/A"
                # else: RC price is NaN, status remains "N/A"
                rc_cheaper_status_list.append(status)
            df_final_filtered['RC Cheaper?'] = rc_cheaper_status_list
        else:
            # Ensure the column exists even if df_final_filtered is empty, for consistent downstream processing
            if 'RC Cheaper?' not in df_final_filtered.columns:
                 df_final_filtered['RC Cheaper?'] = pd.Series(dtype='object')

        # --- Prepare df_for_ui for the main table in the UI ---
        # Columns to explicitly hide from the main display table in the UI
        cols_to_hide_in_ui = ['BRAND', 'SPORT', 'TYPE1']
        # Columns to always drop from any display/export (e.g. temporary KPI calculation helpers)
        cols_to_always_drop = ['Stock Value', 'Sales Value (12M)'] 

        # Start with all columns from df_final_filtered (which now includes 'RC Cheaper?')
        potential_ui_cols = df_final_filtered.columns.tolist()
        
        # Remove columns that should never be shown or were temporary
        actual_ui_cols = [col for col in potential_ui_cols if col not in cols_to_always_drop]
        # Further refine for UI: remove columns designated to be hidden in this specific table
        actual_ui_cols = [col for col in actual_ui_cols if col not in cols_to_hide_in_ui]

        # Reorder to place 'RC Cheaper?' after 'ITEM NAME' if both exist
        if 'ITEM NAME' in actual_ui_cols and 'RC Cheaper?' in actual_ui_cols:
            actual_ui_cols.remove('RC Cheaper?')
            try:
                item_name_idx = actual_ui_cols.index('ITEM NAME')
                actual_ui_cols.insert(item_name_idx + 1, 'RC Cheaper?')
            except ValueError:
                # If 'ITEM NAME' isn't in actual_ui_cols for some reason, just append 'RC Cheaper?' at the end
                actual_ui_cols.append('RC Cheaper?')
        elif 'RC Cheaper?' not in actual_ui_cols and 'RC Cheaper?' in df_final_filtered.columns.tolist() : # if it got filtered out but should be there
            actual_ui_cols.append('RC Cheaper?')

        # Create the DataFrame for UI display
        df_for_ui = df_final_filtered[actual_ui_cols] if actual_ui_cols and not df_final_filtered.empty else pd.DataFrame(columns=actual_ui_cols)
        
        # Define styling function for the 'RC Cheaper?' column
        def style_rc_cheaper(val):
            if val == "Cheaper": # Changed from "Yes"
                return 'background-color: lightgreen; color: darkgreen; font-weight: bold;'
            elif val == "Higher": # Changed from "No"
                return 'background-color: lightcoral; color: darkred; font-weight: bold;'
            elif val == "Equal": # New status
                return 'background-color: lightyellow; color: #B8860B; font-weight: bold;' # DarkGoldenrod
            else:  # For "N/A"
                return '' # No special style for N/A

        if not df_for_ui.empty:
            if 'RC Cheaper?' in df_for_ui.columns:
                st.dataframe(
                    df_for_ui.style.applymap(style_rc_cheaper, subset=['RC Cheaper?']),
                    use_container_width=True,
                    hide_index=True
                )
            else: # Display without styling if 'RC Cheaper?' column isn't in the final UI df
                st.dataframe(df_for_ui, use_container_width=True, hide_index=True)
        else:
            st.info("No data to display in the main table based on current filters.")

        # Function to convert DF to CSV
        # The CSV export should use df_final_filtered (which includes 'RC Cheaper?' and original columns like BRAND, SPORT, TYPE1)
        # but drop the temporary calculation columns.
        @st.cache_data # Cache conversion
        def convert_df_to_csv(df_to_convert):
            # df_to_convert is df_final_filtered
            df_to_export = df_to_convert.drop(columns=cols_to_always_drop, errors='ignore')
            return df_to_export.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(df_final_filtered) # Pass the df_final_filtered (now with 'RC Cheaper?')

        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name='filtered_pricing_data.csv',
            mime='text/csv',
        )

        st.divider()

        # --- Alerts ---
        st.subheader("⚠️ Alerts & Highlights")
        st.caption("Highlights potential issues based on predefined thresholds (e.g., high/low stock, pricing below cost).")
        # Remove the columns layout
        # alert_cols = st.columns(3)

        # High Stock Days
        HIGH_STOCK_DAYS = 180 # Example threshold
        df_high_stock = df_final_filtered[df_final_filtered['Stock Days on Hand'] > HIGH_STOCK_DAYS]
        # Display directly without column
        st.warning(f"High Stock ({len(df_high_stock)} items > {HIGH_STOCK_DAYS} days)")
        if not df_high_stock.empty:
            # Removed height=200
            st.dataframe(df_high_stock[['BRAND', 'ITEM NAME', 'Stock Days on Hand']].sort_values('Stock Days on Hand', ascending=False), use_container_width=True, hide_index=True)
        else:
            st.info("No items with high stock days.")
        st.divider() # Add divider between alerts

        # Low Stock Days
        LOW_STOCK_DAYS = 30 # Example threshold
        df_low_stock = df_final_filtered[df_final_filtered['Stock Days on Hand'] < LOW_STOCK_DAYS]
        # Display directly without column
        st.warning(f"Low Stock ({len(df_low_stock)} items < {LOW_STOCK_DAYS} days)")
        if not df_low_stock.empty:
             # Reverted .head(50) and removed height limit
             st.dataframe(df_low_stock[['BRAND', 'ITEM NAME', 'Stock Days on Hand']].sort_values('Stock Days on Hand'), use_container_width=True, hide_index=True)
        else:
            st.info("No items with low stock days.")
        st.divider() # Add divider between alerts

        # Price Below Cost
        df_below_cost = df_final_filtered[df_final_filtered['RACKET CENTRAL'] < df_final_filtered['Unit Cost']]
        # Display directly without column
        st.error(f"Price Below Cost ({len(df_below_cost)} items)")
        if not df_below_cost.empty:
            # Removed height=200
            st.dataframe(df_below_cost[['BRAND', 'ITEM NAME', 'Unit Cost', 'RACKET CENTRAL']], use_container_width=True, hide_index=True)
        else:
            st.info("No items priced below cost.")

        # Price Below MAP (Optional - if MAP exists and is relevant)
        # df_below_map = df_final_filtered[df_final_filtered['LOWEST PRICE (B2C)'] < df_final_filtered['MAP']]
        # st.warning(f"Price Below MAP ({len(df_below_map)} items)")
        # st.dataframe(df_below_map[['BRAND', 'ITEM NAME', 'MAP', 'LOWEST PRICE (B2C)']])

        st.divider()

        # --- Visualizations ---
        st.subheader("📊 Visualizations")
        st.caption("Visual representations of the filtered data across different categories.")
        tab2, tab3, tab_compare = st.tabs(["Stock Analysis", "Margins & Costs", "Price Comparison"])

        with tab2:
            st.markdown("#### Stock Analysis")
            st.caption("Visualizations related to stock quantity and days on hand.")
            col1, col2 = st.columns(2)
            with col1:
                if not df_final_filtered['Total Stock (QTY)'].isnull().all():
                    stock_by_brand = df_final_filtered.groupby('BRAND')['Total Stock (QTY)'].sum().reset_index().sort_values('Total Stock (QTY)', ascending=False)
                    fig_stock_brand = px.bar(stock_by_brand, x='BRAND', y='Total Stock (QTY)', title='Total Stock Quantity by Brand')
                    st.plotly_chart(fig_stock_brand, use_container_width=True)
                else: st.info("No Stock Quantity data to display.")
            with col2:
                if not df_final_filtered['Stock Days on Hand'].isnull().all():
                    fig_stock_days = px.histogram(df_final_filtered, x="Stock Days on Hand", title="Stock Days on Hand Distribution", nbins=30)
                    st.plotly_chart(fig_stock_days, use_container_width=True)
                else: st.info("No Stock Days on Hand data to display.")

        with tab3:
            st.markdown("#### Margins & Costs")
            st.caption("Visualizations comparing costs, prices, and profit margins.")
            col1, col2 = st.columns(2)
            with col1:
                if 'RC Marginal Contribution (%)' in df_final_filtered.columns and not df_final_filtered['RC Marginal Contribution (%)'].isnull().all():
                     margin_by_brand = df_final_filtered.groupby('BRAND')['RC Marginal Contribution (%)'].mean().reset_index().sort_values('RC Marginal Contribution (%)', ascending=False)
                     fig_margin_brand = px.bar(margin_by_brand, x='BRAND', y='RC Marginal Contribution (%)', title='Average RC Margin % by Brand', labels={'RC Marginal Contribution (%)':'Avg. Margin (%)'})
                     fig_margin_brand.update_layout(yaxis_tickformat='.1%')
                     st.plotly_chart(fig_margin_brand, use_container_width=True)
                else: st.info("No Margin data to display.")
            with col2:
                 if not df_final_filtered['Unit Cost'].isnull().all() and not df_final_filtered['RACKET CENTRAL'].isnull().all():
                    fig_cost_price = px.scatter(df_final_filtered, x="Unit Cost", y="RACKET CENTRAL",
                                                title="Unit Cost vs. RACKET CENTRAL Price",
                                                color="BRAND", hover_data=['ITEM NAME']) # Use 'ITEM NAME'
                    st.plotly_chart(fig_cost_price, use_container_width=True)
                 else: st.info("Insufficient Cost or RACKET CENTRAL Price data for scatter plot.")

        # --- Price Comparison Tab ---
        with tab_compare:
            st.markdown("#### Competitor Price Comparison")
            st.caption("Compares 'RACKET CENTRAL' prices against available competitor data.")

            # Define potential competitor columns
            competitor_cols = ['JUST PADDLES', 'PADEL USA', 'CASAS PADEL', 'FROMUTH', 'PICKLEBALL CENTRAL']
            rc_price_col = 'RACKET CENTRAL' # Our price column

            # Identify competitors actually present in the filtered data
            available_competitors = [col for col in competitor_cols if col in df_final_filtered.columns and not df_final_filtered[col].isnull().all()]

            if not available_competitors:
                st.info("No competitor pricing data available in the current selection for comparison.")
            elif rc_price_col not in df_final_filtered.columns or df_final_filtered[rc_price_col].isnull().all():
                 st.info(f"No '{rc_price_col}' pricing data available in the current selection for comparison.")
            else:
                # Create DataFrame for comparison
                cols_to_compare = ['BRAND', 'ITEM NAME', rc_price_col] + available_competitors
                df_comparison = df_final_filtered[cols_to_compare].copy()

                # --- Calculate Differences and KPIs ---
                st.markdown("**Summary vs Competitors:**")
                kpi_compare_cols = st.columns(len(available_competitors))

                for i, competitor in enumerate(available_competitors):
                    diff_col_name = f"Diff vs {competitor}"
                    # Calculate difference (RC - Competitor). Negative means RC is cheaper.
                    # Ensure both columns are numeric before subtraction
                    rc_numeric = pd.to_numeric(df_comparison[rc_price_col], errors='coerce')
                    competitor_numeric = pd.to_numeric(df_comparison[competitor], errors='coerce')
                    df_comparison[diff_col_name] = rc_numeric - competitor_numeric

                    # Calculate KPIs for this competitor
                    valid_comparisons = df_comparison[diff_col_name].notna()
                    total_comparable = valid_comparisons.sum()

                    if total_comparable > 0:
                        rc_cheaper_count = (df_comparison[diff_col_name][valid_comparisons] < 0).sum()
                        rc_more_expensive_count = (df_comparison[diff_col_name][valid_comparisons] > 0).sum()
                        # rc_same_price_count = (df_comparison[diff_col_name][valid_comparisons] == 0).sum() # Not explicitly used in metric

                        avg_diff = df_comparison[diff_col_name][valid_comparisons].mean()
                        
                        delta_text = f"Avg Diff: ${avg_diff:+.2f}"
                        delta_color = "inverse" if avg_diff > 0 else "normal" # 'inverse' if RC is more expensive on avg

                        with kpi_compare_cols[i]:
                            st.metric(
                                label=f"vs {competitor} ({total_comparable} comparable)",
                                value=f"RC Cheaper: {rc_cheaper_count}",
                                delta=delta_text,
                                delta_color=delta_color
                            )
                            st.caption(f"RC More Expensive: {rc_more_expensive_count}")
                    else:
                         with kpi_compare_cols[i]:
                             st.info(f"No comparable prices found for {competitor}.")
                
                st.divider()

                # --- Overall Price Position Analysis ---
                # This section is now moved to the main KPIs above.
                # st.markdown("**Overall Price Position:**")
                # ... (original code for overall price position was here) ...
                # st.divider()

                # --- Display Detailed Table ---
                # Filter out rows where ALL competitor prices are NaN
                if available_competitors: # Ensure there are competitors to check
                    df_comparison = df_comparison.dropna(subset=available_competitors, how='all')

                st.markdown("**Detailed Price Comparison:**")
                st.caption(f"'{rc_price_col}' is our price. 'Diff vs [Competitor]' shows '{rc_price_col}' - '[Competitor Price]'. Negative values mean '{rc_price_col}' is cheaper.")
                # Display the potentially filtered DataFrame
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)


    # --- Display Original Data (Optional Expander) ---
    with st.expander("Show Original Full Data"):
       st.dataframe(df_original, use_container_width=True, hide_index=True)

# --- Add source information ---
st.caption("ℹ️ **Data Source Note:** Competitor pricing data is obtained weekly (Thursdays, 8 AM US Central Time) by scraping competitor websites and matching products using their EAN codes.")

# --- Add instructions to run ---
# st.sidebar.info("To run this dashboard: `streamlit run dashboard.py`") 