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
        # --- Ensure 'Year/Model' is present as a column (if not, try to add as empty) ---
        if 'Year/Model' not in df.columns:
            df['Year/Model'] = pd.NA
        # Basic data cleaning: Replace empty strings with NaN, attempt numeric conversion
        df.replace('', pd.NA, inplace=True) # B: Converts empty strings to pd.NA globally.
                                            # If 'Year/Model' had empty strings, they become pd.NA.

        # Convert 'Year/Model' to string type for consistent filtering
        if 'Year/Model' in df.columns:
            df['Year/Model'] = df['Year/Model'].astype(str)
            # Replace the string '<NA>' (which results from pd.NA.astype(str)) back to pd.NA
            # if you want to treat them as actual missing values for other purposes
            # or for cleaner display in filter options if desired.
            # For the current filtering mechanism, leaving them as "<NA>" is also fine
            # as get_unique_options_sim will also treat it as a string option.
            # Let's convert "<NA>" string back to pd.NA for general data consistency, 
            # get_unique_options_sim will handle it by converting to string again for options if NA exists.
            df.loc[df['Year/Model'] == '<NA>', 'Year/Model'] = pd.NA

        for col in df.columns: # C: This loop handles numeric/percentage conversions for other columns.
                               # 'Year/Model' is not in numeric_cols or percentage_cols, so it's skipped.
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

# Helper function to get unique sorted options for dashboard filters
def get_unique_options_dashboard(df_source, column_name): # Modified to take df_source
    if column_name not in df_source.columns:
        return []
    return sorted(df_source[column_name].astype(str).unique())

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="Pricing & Competition Dashboard")
st.title("📊 Pricing & Competition Dashboard")
st.caption("This dashboard provides insights into product pricing, stock levels, margins, and competitor pricing based on data from Google Sheets.")

# Fetch data
gc = get_gspread_client()
df_original = fetch_data(gc)

if df_original.empty:
    st.warning("Could not load data from Google Sheets. Please check the connection and sheet configuration.")
else:
    st.success(f"✅ Data loaded successfully from Google Sheet! Contains {len(df_original)} rows.")

    # --- Define global constants for columns ---
    ALL_COMPETITOR_COLUMNS = sorted(['JUST PADDLES', 'PADEL USA', 'CASAS PADEL', 'FROMUTH', 'PICKLEBALL CENTRAL'])
    RC_PRICE_COLUMN = 'RACKET CENTRAL'
    # Assuming RACKET_CENTRAL_B2B_PRICE_COLUMN is a column in your sheet, e.g., 'RC B2B Price'
    # If it's named differently, adjust this constant.
    # If it needs to be calculated, that logic would go into fetch_data or here.
    RC_B2B_PRICE_COLUMN = 'RACKET CENTRAL B2B' # Placeholder, ensure this column exists in your data source

    # --- Main Application Tabs ---
    tab_dashboard, tab_simulator, tab_completeness = st.tabs(["📊 Dashboard Overview", "⚙️ Price Simulator", "🔍 Data Completeness Analysis"])

    with tab_dashboard:
        st.header("📋 Dashboard Filters & Overview")
        df = df_original.copy() # Work with a copy for filtering for this tab

        # --- Sidebar Filters (for Dashboard Overview tab) ---
        st.sidebar.header("Dashboard Filters")

        # EAN Filter
        ean_search = st.sidebar.text_input("Search by EAN:", placeholder="Enter EAN...", key="dashboard_ean")

        # SKU Filter
        sku_search = st.sidebar.text_input("Search by SKU:", placeholder="Enter SKU...", key="dashboard_sku")

        # Text Search Filter
        search_term = st.sidebar.text_input("Search by Item Name:", placeholder="Enter item name...", key="dashboard_item_name")

        # Categorical Filters
        brands_options_dashboard = sorted(df["BRAND"].unique())
        brands_dashboard = st.sidebar.multiselect(
            "Select Brand(s):",
            options=brands_options_dashboard,
            key="dashboard_brands"
        )

        sports_options_dashboard = get_unique_options_dashboard(df, "SPORT")
        sports_dashboard = st.sidebar.multiselect(
            "Select Sport(s):",
            options=sports_options_dashboard,
            key="dashboard_sports"
        )

        type1_options_dashboard = get_unique_options_dashboard(df, "TYPE1")
        type1s_dashboard = st.sidebar.multiselect(
            "Select Type1(s):",
            options=type1_options_dashboard,
            key="dashboard_type1s"
        )
        
        year_model_options_dashboard = get_unique_options_dashboard(df, "Year/Model")
        year_models_dashboard = st.sidebar.multiselect(
            "Select Year/Model(s):",
            options=year_model_options_dashboard,
            key="dashboard_year_models"
        )
        
        selected_competitors_for_coccurrence_filter = st.sidebar.multiselect(
            "Show Products Priced by RC & Selected Competitor(s):",
            options=ALL_COMPETITOR_COLUMNS,
            help="Filters data to show only products where Racket Central AND at least one of the selected competitors have a price.",
            key="dashboard_cooccurrence"
        )

        # Apply Categorical Filters
        if not brands_dashboard: brands_dashboard = df["BRAND"].unique()
        if not sports_dashboard: sports_dashboard = get_unique_options_dashboard(df, "SPORT") # Use all if empty
        if not type1s_dashboard: type1s_dashboard = get_unique_options_dashboard(df, "TYPE1") # Use all if empty
        if not year_models_dashboard: year_models_dashboard = get_unique_options_dashboard(df, "Year/Model") # Use all if empty
        
        df_filtered = df[
            df["BRAND"].isin(brands_dashboard) &
            df["SPORT"].isin(sports_dashboard) &
            df["TYPE1"].isin(type1s_dashboard) &
            df["Year/Model"].isin(year_models_dashboard) # Added Year/Model filter
        ]
        
        # Numerical Filters (using sliders, handle potential NaNs and empty filtered data)
        # (Keep existing add_range_slider and its applications as they were for the dashboard tab)
        def add_range_slider(column_name, key_suffix=""): # Added key_suffix for unique keys
            if df_filtered.empty or column_name not in df_filtered.columns or df_filtered[column_name].isnull().all():
                st.sidebar.warning(f"No data or only missing values for '{column_name}' in current selection.")
                return None, None

            min_val = df_filtered[column_name].min()
            max_val = df_filtered[column_name].max()

            if pd.isna(min_val) or pd.isna(max_val):
                 st.sidebar.warning(f"Cannot create slider for '{column_name}' due to missing values.")
                 return None, None

            step = 1
            if isinstance(min_val, float) or isinstance(max_val, float):
                diff = max_val - min_val
                if diff < 1 and diff > 0: step = 0.01
                elif diff == 0: step = 0.1
                else:
                    step = (max_val - min_val) / 100
                    step = max(step, 0.01)
            
            if min_val == max_val:
                 range_min_slider, range_max_slider = min_val - step * 5, max_val + step * 5
            else:
                 range_min_slider, range_max_slider = min_val, max_val
            
            if range_min_slider >= range_max_slider: range_max_slider = range_min_slider + step

            try:
                selected_range = st.sidebar.slider(
                    f"Select {column_name} Range:",
                    min_value=float(range_min_slider),
                    max_value=float(range_max_slider),
                    value=(float(range_min_slider), float(range_max_slider)),
                    step=float(step) if step > 0 else None,
                    key=f"slider_{column_name.replace(' ', '_')}_{key_suffix}" # Unique key
                )
                return selected_range
            except Exception as e:
                st.sidebar.error(f"Error creating slider for {column_name}: {e}")
                return None, None

        # --- Apply Text Search Filters ---
        df_final_filtered = df_filtered.copy() # Start with categorically filtered data

        if ean_search and 'EAN' in df_final_filtered.columns:
            df_final_filtered = df_final_filtered[df_final_filtered['EAN'].astype(str).str.contains(ean_search, case=False, na=False)]
        if sku_search and 'SKU' in df_final_filtered.columns:
            df_final_filtered = df_final_filtered[df_final_filtered['SKU'].astype(str).str.contains(sku_search, case=False, na=False)]
        if search_term and 'ITEM NAME' in df_final_filtered.columns:
             df_final_filtered = df_final_filtered[df_final_filtered['ITEM NAME'].str.contains(search_term, case=False, na=False)]

        # Apply Co-occurrence Filter
        if selected_competitors_for_coccurrence_filter:
            rows_to_keep_coccurrence = []
            for index, row in df_final_filtered.iterrows():
                rc_price_is_valid = pd.notna(pd.to_numeric(row.get(RC_PRICE_COLUMN), errors='coerce'))
                if not rc_price_is_valid: continue
                at_least_one_selected_competitor_has_price = False
                for comp_col in selected_competitors_for_coccurrence_filter:
                    if comp_col in row and pd.notna(pd.to_numeric(row[comp_col], errors='coerce')):
                        at_least_one_selected_competitor_has_price = True
                        break
                if at_least_one_selected_competitor_has_price:
                    rows_to_keep_coccurrence.append(index)
            df_final_filtered = df_final_filtered.loc[rows_to_keep_coccurrence] if rows_to_keep_coccurrence else df_final_filtered.iloc[0:0]


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
                        comp_price = pd.to_numeric(row.get(comp_col), errors='coerce')
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
                    st.dataframe(df_comparison.drop(columns=[col for col in df_comparison.columns if 'Diff vs' in col], errors='ignore'), use_container_width=True, hide_index=True) # Display without diff columns for cleanliness here, diffs used in metrics


        # --- Display Original Data (Optional Expander) ---
        with st.expander("Show Original Full Data (Unfiltered)"):
           st.dataframe(df_original, use_container_width=True, hide_index=True)

    # --- SIMULATOR TAB ---
    with tab_simulator:
        st.header("⚙️ Price Simulator")
        st.caption("Select products and apply discounts to simulate pricing strategies and check compliance with business rules.")

        if 'staged_scenario_items' not in st.session_state:
            st.session_state.staged_scenario_items = [] 
        if 'staged_scenario_kpis' not in st.session_state:
            st.session_state.staged_scenario_kpis = {}

        def get_delta_string_for_metric(new_val, old_val, kind="percentage"):
            """
            Calculates and formats a delta string for st.metric.
            Returns None if new_val or old_val is NA, or if delta is 0.
            kind can be "percentage", "count".
            """
            if pd.notna(new_val) and pd.notna(old_val):
                delta = new_val - old_val
                if delta == 0:
                    return None  # No delta string if change is zero
                
                if kind == "percentage":
                    return f"{delta:+.2f}%"
                elif kind == "count":
                    return f"{delta:+,}"
                # Add other kinds here if needed, e.g., currency
                # elif kind == "currency":
                #     return f"${delta:+, .2f}" 
                else:  # Default numeric format if kind is not recognized or for general numbers
                    return f"{delta:+.2f}"
            return None # If new_val or old_val is NA, no delta.

        if df_original.empty:
            st.warning("Original data is not available for the simulator.")
        else:
            # df_sim = df_original.copy() # We'll use df_final_filtered from the dashboard tab
            df_sim_filtered = df_final_filtered.copy() # Use the globally filtered DataFrame

            # --- Simulator Specific Filters --- (No longer needed here)
            # st.subheader("1. Filter Products for Simulation")
            # sim_cols = st.columns(4)
            
            # def get_unique_options_sim(column_name):
            #     if column_name not in df_sim.columns: # df_sim is now df_sim_filtered
            #         st.warning(f"Column '{column_name}' not found for filtering in simulator.")
            #         return []
            #     return sorted(df_sim[column_name].astype(str).dropna().unique())

            # with sim_cols[0]:
            #     sim_brands_options = get_unique_options_sim("BRAND")
            #     sim_brands = st.multiselect("Brand(s):", sim_brands_options, key="sim_brands")
            # with sim_cols[1]:
            #     sim_sports_options = get_unique_options_sim("SPORT")
            #     sim_sports = st.multiselect("Sport(s):", sim_sports_options, key="sim_sports")
            # with sim_cols[2]:
            #     sim_type1s_options = get_unique_options_sim("TYPE1")
            #     sim_type1s = st.multiselect("Type1(s):", sim_type1s_options, key="sim_type1s")
            # with sim_cols[3]:
            #     sim_yearmodel_options = get_unique_options_sim("Year/Model") # Assuming 'Year/Model' column exists
            #     sim_yearmodel = st.multiselect("Year/Model:", sim_yearmodel_options, key="sim_yearmodel")

            # Apply simulator filters (No longer needed as df_sim_filtered is already filtered by sidebar)
            # df_sim_filtered = df_sim.copy()
            # if sim_brands: df_sim_filtered = df_sim_filtered[df_sim_filtered["BRAND"].isin(sim_brands)]
            # if sim_sports: df_sim_filtered = df_sim_filtered[df_sim_filtered["SPORT"].isin(sim_sports)]
            # if sim_type1s: df_sim_filtered = df_sim_filtered[df_sim_filtered["TYPE1"].isin(sim_type1s)]
            # if sim_yearmodel and "Year/Model" in df_sim_filtered.columns:
            #      df_sim_filtered = df_sim_filtered[df_sim_filtered["Year/Model"].isin(sim_yearmodel)]
            
            # if df_sim_filtered.empty and (sim_brands or sim_sports or sim_type1s or sim_yearmodel) :
            #     st.info("No products match the simulator filters.")
            # el
            if not df_sim_filtered.empty :
                st.success(f"{len(df_sim_filtered)} products selected for simulation from sidebar filters.")
                
                # --- Discount Inputs ---
                st.subheader("2. Apply Discounts (%)")
                discount_cols = st.columns(2)
                with discount_cols[0]:
                    discount_rc = st.number_input(f"Discount on '{RC_PRICE_COLUMN}' (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, format="%.1f", key="sim_discount_rc")
                with discount_cols[1]:
                    # Check if RC_B2B_PRICE_COLUMN exists before creating input
                    if RC_B2B_PRICE_COLUMN in df_sim_filtered.columns:
                        discount_rc_b2b = st.number_input(f"Discount on '{RC_B2B_PRICE_COLUMN}' (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, format="%.1f", key="sim_discount_rc_b2b")
                    else:
                        st.warning(f"Column '{RC_B2B_PRICE_COLUMN}' not found. B2B discount cannot be applied.")
                        discount_rc_b2b = 0.0 # Default to 0 if column not found

                # --- Perform Calculations and Rule Checks ---
                df_results = df_sim_filtered.copy()
                
                # Ensure necessary original price, cost, and MAP columns are numeric
                columns_to_make_numeric = [RC_PRICE_COLUMN, 'Unit Cost', 'MAP'] + ALL_COMPETITOR_COLUMNS
                if RC_B2B_PRICE_COLUMN in df_results.columns: # Check if original B2B price col exists in data
                    columns_to_make_numeric.append(RC_B2B_PRICE_COLUMN)

                for col in columns_to_make_numeric:
                    if col in df_results.columns:
                        df_results[col] = pd.to_numeric(df_results[col], errors='coerce')
                
                # --- Calculate New Prices ---
                # Initialize New Price columns to pd.NA to ensure they exist
                df_results['New RC Price'] = pd.NA
                df_results['New RC B2B Price'] = pd.NA

                if RC_PRICE_COLUMN in df_results.columns:
                    # Apply discount if original price is not NA
                    mask_rc_price_not_na = df_results[RC_PRICE_COLUMN].notna()
                    df_results.loc[mask_rc_price_not_na, 'New RC Price'] = \
                        df_results.loc[mask_rc_price_not_na, RC_PRICE_COLUMN] * (1 - discount_rc / 100)
                
                if RC_B2B_PRICE_COLUMN in df_results.columns:
                    mask_rc_b2b_price_not_na = df_results[RC_B2B_PRICE_COLUMN].notna()
                    df_results.loc[mask_rc_b2b_price_not_na, 'New RC B2B Price'] = \
                        df_results.loc[mask_rc_b2b_price_not_na, RC_B2B_PRICE_COLUMN] * (1 - discount_rc_b2b / 100)
                # --- End New Price Calculation ---


                # --- START: Revised Margin Calculations ---
                cost_col = 'Unit Cost'
                df_results['Current RC Margin (%)'] = pd.NA
                df_results['Current RC B2B Margin (%)'] = pd.NA
                df_results['New RC Margin (%)'] = pd.NA
                df_results['New RC B2B Margin (%)'] = pd.NA

                if cost_col in df_results.columns and df_results[cost_col].notna().any():
                    numeric_cost = df_results[cost_col] 
                    cost_is_valid_mask = numeric_cost.notna() # Cost must exist

                    # Current RC Margin: ((Price - Cost) / Price) * 100
                    if RC_PRICE_COLUMN in df_results.columns:
                        current_rc_price = df_results[RC_PRICE_COLUMN] 
                        price_is_valid_and_nonzero_mask = current_rc_price.notna() & (current_rc_price != 0)
                        final_current_rc_mask = cost_is_valid_mask & price_is_valid_and_nonzero_mask
                        if final_current_rc_mask.any():
                            df_results.loc[final_current_rc_mask, 'Current RC Margin (%)'] = \
                                ((current_rc_price[final_current_rc_mask] - numeric_cost[final_current_rc_mask]) / current_rc_price[final_current_rc_mask]) * 100

                    # Current RC B2B Margin: ((Price - Cost) / Price) * 100
                    if RC_B2B_PRICE_COLUMN in df_results.columns:
                        current_rc_b2b_price = df_results[RC_B2B_PRICE_COLUMN] 
                        price_is_valid_and_nonzero_mask_b2b = current_rc_b2b_price.notna() & (current_rc_b2b_price != 0)
                        final_current_rc_b2b_mask = cost_is_valid_mask & price_is_valid_and_nonzero_mask_b2b
                        if final_current_rc_b2b_mask.any():
                            df_results.loc[final_current_rc_b2b_mask, 'Current RC B2B Margin (%)'] = \
                                ((current_rc_b2b_price[final_current_rc_b2b_mask] - numeric_cost[final_current_rc_b2b_mask]) / current_rc_b2b_price[final_current_rc_b2b_mask]) * 100
                    
                    # New RC Margin: ((New Price - Cost) / New Price) * 100
                    new_rc_price_numeric = df_results['New RC Price'] 
                    new_price_is_valid_and_nonzero_mask = new_rc_price_numeric.notna() & (new_rc_price_numeric != 0)
                    final_new_rc_mask = cost_is_valid_mask & new_price_is_valid_and_nonzero_mask
                    if final_new_rc_mask.any():
                        df_results.loc[final_new_rc_mask, 'New RC Margin (%)'] = \
                            ((new_rc_price_numeric[final_new_rc_mask] - numeric_cost[final_new_rc_mask]) / new_rc_price_numeric[final_new_rc_mask]) * 100

                    # New RC B2B Margin: ((New Price - Cost) / New Price) * 100
                    new_rc_b2b_price_numeric = df_results['New RC B2B Price']
                    new_b2b_price_is_valid_and_nonzero_mask = new_rc_b2b_price_numeric.notna() & (new_rc_b2b_price_numeric != 0)
                    final_new_rc_b2b_mask = cost_is_valid_mask & new_b2b_price_is_valid_and_nonzero_mask # Corrected this line
                    if final_new_rc_b2b_mask.any(): # Corrected this line to use the correct mask
                        df_results.loc[final_new_rc_b2b_mask, 'New RC B2B Margin (%)'] = \
                            ((new_rc_b2b_price_numeric[final_new_rc_b2b_mask] - numeric_cost[final_new_rc_b2b_mask]) / new_rc_b2b_price_numeric[final_new_rc_b2b_mask]) * 100
                else:
                    st.warning(f"'{cost_col}' column not found or contains no valid data. Margins cannot be calculated accurately.")
                # --- END: Revised Margin Calculations ---

                # Competitor Prices & MAP
                B2C_COMPETITOR_COLUMNS = [col for col in ALL_COMPETITOR_COLUMNS if col != 'FROMUTH' and col in df_results.columns]
                # B2B_COMPETITOR_COLUMNS = [col for col in ALL_COMPETITOR_COLUMNS if col in df_results.columns] # FROMUTH is included here if present
                B2B_COMPETITOR_COLUMNS = ['FROMUTH'] if 'FROMUTH' in df_results.columns else []

                if B2C_COMPETITOR_COLUMNS:
                    df_results['Min B2C Competitor Price'] = df_results[B2C_COMPETITOR_COLUMNS].min(axis=1, skipna=True)
                else:
                    df_results['Min B2C Competitor Price'] = pd.NA
                
                if B2B_COMPETITOR_COLUMNS:
                    df_results['Min B2B Competitor Price'] = df_results[B2B_COMPETITOR_COLUMNS].min(axis=1, skipna=True)
                else:
                    df_results['Min B2B Competitor Price'] = pd.NA
                

                # --- Business Rule Checks ---
                # Rule 1: New price <= Min_Competitor_Price
                df_results['RC Meets Rule 1'] = df_results['New RC Price'] <= df_results['Min B2C Competitor Price']
                df_results.loc[df_results['Min B2C Competitor Price'].isna(), 'RC Meets Rule 1'] = "N/A (No B2C Comp)"

                df_results['RC B2B Meets Rule 1'] = df_results['New RC B2B Price'] <= df_results['Min B2B Competitor Price']
                df_results.loc[df_results['Min B2B Competitor Price'].isna(), 'RC B2B Meets Rule 1'] = "N/A (No B2B Comp)"


                # Rule 2: New price >= MAP
                map_col = 'MAP'
                if map_col in df_results.columns:
                    df_results['RC Meets Rule 2'] = df_results['New RC Price'] >= df_results[map_col]
                    df_results['RC B2B Meets Rule 2'] = df_results['New RC B2B Price'] >= df_results[map_col]
                    # Handle cases where MAP is NaN -> rule could be considered met or N/A. For now, if MAP is NaN, comparison is False.
                    df_results.loc[df_results[map_col].isna(), ['RC Meets Rule 2', 'RC B2B Meets Rule 2']] = "N/A (No MAP)"
                else:
                    st.warning(f"'{map_col}' column not found. Rule 2 (MAP compliance) cannot be fully checked.")
                    df_results['RC Meets Rule 2'] = "N/A (No MAP Col)"
                    df_results['RC B2B Meets Rule 2'] = "N/A (No MAP Col)"

                # Rule 3: Margin >= 20% ( (New Price - Cost) / Cost >= 0.20 )
                # Rule 3 is actually: New Margin (Gross Margin) >= 20%
                df_results['RC Meets Rule 3'] = df_results['New RC Margin (%)'] >= 20.0
                df_results['RC B2B Meets Rule 3'] = df_results['New RC B2B Margin (%)'] >= 20.0
                if cost_col not in df_results.columns: # Should be if 'New RC Margin (%)' is NA, but cost check is a proxy
                     df_results['RC Meets Rule 3'] = "N/A (No Cost Col for Margin)" # Or "N/A (Margin Not Calc)"
                     df_results['RC B2B Meets Rule 3'] = "N/A (No Cost Col for Margin)"


                # --- START: Current State Rule Checks ---
                # Current Rule 1: Original price <= Min_Competitor_Price
                df_results['Current RC Meets Rule 1'] = pd.NA
                if RC_PRICE_COLUMN in df_results.columns:
                    df_results['Current RC Meets Rule 1'] = df_results[RC_PRICE_COLUMN] <= df_results['Min B2C Competitor Price']
                    df_results.loc[df_results['Min B2C Competitor Price'].isna(), 'Current RC Meets Rule 1'] = "N/A (No B2C Comp)"

                df_results['Current RC B2B Meets Rule 1'] = pd.NA
                if RC_B2B_PRICE_COLUMN in df_results.columns:
                    df_results['Current RC B2B Meets Rule 1'] = df_results[RC_B2B_PRICE_COLUMN] <= df_results['Min B2B Competitor Price']
                    df_results.loc[df_results['Min B2B Competitor Price'].isna(), 'Current RC B2B Meets Rule 1'] = "N/A (No B2B Comp)"

                # Current Rule 2: Original price >= MAP
                if map_col in df_results.columns:
                    df_results['Current RC Meets Rule 2'] = pd.NA
                    if RC_PRICE_COLUMN in df_results.columns:
                        df_results['Current RC Meets Rule 2'] = df_results[RC_PRICE_COLUMN] >= df_results[map_col]
                        df_results.loc[df_results[map_col].isna() & df_results[RC_PRICE_COLUMN].notna(), 'Current RC Meets Rule 2'] = "N/A (No MAP)"
                    
                    df_results['Current RC B2B Meets Rule 2'] = pd.NA
                    if RC_B2B_PRICE_COLUMN in df_results.columns:
                        df_results['Current RC B2B Meets Rule 2'] = df_results[RC_B2B_PRICE_COLUMN] >= df_results[map_col]
                        df_results.loc[df_results[map_col].isna() & df_results[RC_B2B_PRICE_COLUMN].notna(), 'Current RC B2B Meets Rule 2'] = "N/A (No MAP)"
                else:
                    df_results['Current RC Meets Rule 2'] = "N/A (No MAP Col)"
                    df_results['Current RC B2B Meets Rule 2'] = "N/A (No MAP Col)"

                # Current Rule 3: Current Margin >= 20%
                df_results['Current RC Meets Rule 3'] = pd.NA
                if 'Current RC Margin (%)' in df_results.columns:
                    df_results['Current RC Meets Rule 3'] = df_results['Current RC Margin (%)'] >= 20.0
                    # If Current RC Margin is NA (e.g. due to no cost), this comparison results in NA, which is fine for styling.
                    # We can make it explicit if needed for the check_current_overall_compliance function.
                    df_results.loc[df_results['Current RC Margin (%)'].isna(), 'Current RC Meets Rule 3'] = "N/A (Margin Not Calc)"


                df_results['Current RC B2B Meets Rule 3'] = pd.NA
                if 'Current RC B2B Margin (%)' in df_results.columns:
                    df_results['Current RC B2B Meets Rule 3'] = df_results['Current RC B2B Margin (%)'] >= 20.0
                    df_results.loc[df_results['Current RC B2B Margin (%)'].isna(), 'Current RC B2B Meets Rule 3'] = "N/A (Margin Not Calc)"
                # --- END: Current State Rule Checks ---


                # Overall Compliance (Simulated/New State)
                def check_overall_compliance(row, price_type_prefix): # price_type_prefix is 'RC' or 'RC B2B'
                    # Uses 'New [PREFIX] Price' and '[PREFIX] Meets Rule X'
                    new_price_col = f'New {price_type_prefix} Price'
                    rule1_key = f'{price_type_prefix} Meets Rule 1'
                    rule2_key = f'{price_type_prefix} Meets Rule 2'
                    rule3_key = f'{price_type_prefix} Meets Rule 3'

                    # If the new price itself couldn't be calculated (e.g. no original price), it cannot be compliant.
                    if new_price_col not in row or pd.isna(row[new_price_col]):
                        return False # Or "N/A (No New Price)" - False is simpler for overall boolean

                    rule1_met_val = row.get(rule1_key)
                    rule2_met_val = row.get(rule2_key)
                    rule3_met_val = row.get(rule3_key)

                    # Check for missing rule columns (should not happen if logic is correct)
                    if rule1_met_val is None or rule2_met_val is None or rule3_met_val is None:
                        return "N/A (Rule Data Missing)"

                    is_rule1_pass = rule1_met_val is True or (isinstance(rule1_met_val, str) and "N/A (No" in rule1_met_val and "Comp)" in rule1_met_val)
                    is_rule2_pass = rule2_met_val is True or (isinstance(rule2_met_val, str) and "N/A (No" in rule2_met_val and "MAP)" in rule2_met_val)
                    is_rule3_pass = rule3_met_val is True or (isinstance(rule3_met_val, str) and "N/A (No" in rule3_met_val and "Margin)" in rule3_met_val)
                    
                    # If any rule is definitively False, overall is False
                    if rule1_met_val is False or rule2_met_val is False or rule3_met_val is False:
                        return False

                    # If all rules that could be checked are True or N/A (non-blocking N/A), then it's compliant
                    return is_rule1_pass and is_rule2_pass and is_rule3_pass

                df_results['RC Overall Compliant'] = df_results.apply(lambda row: check_overall_compliance(row, 'RC'), axis=1)
                if RC_B2B_PRICE_COLUMN in df_sim_filtered.columns : # Check if original B2B column exists to warrant B2B simulation
                     df_results['RC B2B Overall Compliant'] = df_results.apply(lambda row: check_overall_compliance(row, 'RC B2B'), axis=1)
                else:
                     df_results['RC B2B Overall Compliant'] = False # Or pd.NA

                # --- START: Current Overall Compliance ---
                def check_current_overall_compliance(row, price_type_prefix): # price_type_prefix is 'RC' or 'RC B2B'
                    original_price_col = RC_PRICE_COLUMN if price_type_prefix == 'RC' else RC_B2B_PRICE_COLUMN
                    rule1_key = f'Current {price_type_prefix} Meets Rule 1'
                    rule2_key = f'Current {price_type_prefix} Meets Rule 2'
                    rule3_key = f'Current {price_type_prefix} Meets Rule 3'

                    # If the original price itself is not available, current compliance cannot be determined for rules.
                    # However, overall compliance might be False if one rule is False, or N/A if rules are N/A.
                    # Let's check rule status first.
                    if original_price_col not in row or pd.isna(row[original_price_col]):
                         # If price is NA, but some rule might be definitively false due to other data (unlikely with current setup)
                         # For now, if original price is NA, it's hard to be "compliant".
                         # Let's see if any rule is False.
                         pass # Continue to check rules status


                    rule1_met_val = row.get(rule1_key)
                    rule2_met_val = row.get(rule2_key)
                    rule3_met_val = row.get(rule3_key)
                    
                    # Check for missing rule columns
                    if rule1_met_val is None or rule2_met_val is None or rule3_met_val is None:
                        return "N/A (Rule Data Missing)" # Should indicate an issue if this happens.

                    is_rule1_pass = rule1_met_val is True or (isinstance(rule1_met_val, str) and "N/A (No" in rule1_met_val and "Comp)" in rule1_met_val)
                    is_rule2_pass = rule2_met_val is True or (isinstance(rule2_met_val, str) and "N/A (No" in rule2_met_val and "MAP)" in rule2_met_val)
                    is_rule3_pass = rule3_met_val is True or (isinstance(rule3_met_val, str) and "N/A (No" in rule3_met_val and "Margin)" in rule3_met_val)

                    # If any rule is definitively False, overall is False
                    if rule1_met_val is False or rule2_met_val is False or rule3_met_val is False:
                        return False
                        
                    # If original price is NA, and no rule was False (i.e. all are True or N/A), then overall is N/A not True
                    if (original_price_col not in row or pd.isna(row[original_price_col])) and \
                       (rule1_met_val is not False and rule2_met_val is not False and rule3_met_val is not False):
                        return "N/A (No Original Price)"


                    return is_rule1_pass and is_rule2_pass and is_rule3_pass

                df_results['Current RC Overall Compliant'] = pd.NA
                if RC_PRICE_COLUMN in df_results.columns: # Only if original RC price exists
                    df_results['Current RC Overall Compliant'] = df_results.apply(lambda row: check_current_overall_compliance(row, 'RC'), axis=1)

                df_results['Current RC B2B Overall Compliant'] = pd.NA
                if RC_B2B_PRICE_COLUMN in df_results.columns: # Only if original B2B price exists
                     df_results['Current RC B2B Overall Compliant'] = df_results.apply(lambda row: check_current_overall_compliance(row, 'RC B2B'), axis=1)
                # --- END: Current Overall Compliance ---


                # --- Display Results Table ---
                st.subheader("3. Simulation Results")

                # --- Footnotes & Definitions Expander ---
                with st.expander("ℹ️ View Calculation Formulas and Business Rule Definitions", expanded=False):
                    st.markdown("#### Key Metric Calculations (for an individual item):")
                    st.markdown(f"The 'Avg.' KPIs shown in the metrics display are averages of these item-level calculations.")
                    st.markdown(f"1.  **`RC Price`**: `Original '{RC_PRICE_COLUMN}' Price * (1 - Discount % / 100)` - The B2C price after discount.")
                    if RC_B2B_PRICE_COLUMN in df_sim_filtered.columns and RC_B2B_PRICE_COLUMN in df_results.columns:
                        st.markdown(f"2.  **`RC B2B Price`**: `Original '{RC_B2B_PRICE_COLUMN}' Price * (1 - Discount % / 100)` - The B2B price after discount.")
                    st.markdown(f"3.  **`Gross Margin (%)`**: `((Price - Unit Cost) / Price) * 100%` - Profit relative to selling price.")
                    st.markdown(f"4.  **`Markup on Cost (%)`**: `((Price - Unit Cost) / Unit Cost) * 100%` - Price increase from unit cost.")
                    st.markdown(f"5.  **`Markup on MAP (%)`**: `((Price - MAP) / MAP) * 100%` - Price difference from MAP, relative to MAP.")
                    st.markdown("--- ")

                    st.markdown("#### Business Rule Definitions (for an individual item):")
                    st.markdown(f"'Met' status indicates if the rule is satisfied for that item. A rule is considered 'Met' (non-blocking N/A) if prerequisite data like competitor prices, MAP, or cost is missing.")
                    
                    b2c_competitors_in_data = sorted([col for col in ALL_COMPETITOR_COLUMNS if col != 'FROMUTH' and col in df_results.columns and df_results[col].notna().any()])
                    b2b_competitors_in_data = ['FROMUTH'] if 'FROMUTH' in df_results.columns and df_results['FROMUTH'].notna().any() else []
                    b2c_competitors_list_display = ", ".join(b2c_competitors_in_data) if b2c_competitors_in_data else "None in selection"
                    b2b_competitors_list_display = "FROMUTH" if b2b_competitors_in_data else "FROMUTH not in selection"

                    st.markdown(f"1.  **`Rule 1: vs Comp Met`**")
                    st.markdown(f"    - **Condition:** `Price ≤ Minimum Competitor Price`")
                    st.markdown(f"    - **Brief:** Price is at/below lowest relevant competitor. B2C Comps: {b2c_competitors_list_display}. B2B Comp: {b2b_competitors_list_display}.")

                    st.markdown(f"2.  **`Rule 2: vs MAP Met`**")
                    st.markdown(f"    - **Condition:** `Price ≥ MAP`")
                    st.markdown(f"    - **Brief:** Price is at/above MAP." )

                    st.markdown(f"3.  **`Rule 3: Margin ≥ 20% Met`**")
                    st.markdown(f"    - **Condition:** Item's `Gross Margin (%) ≥ 20%`")
                    st.markdown(f"    - **Brief:** Gross margin is at least 20%.")

                    st.markdown(f"4.  **`Overall Compliant Met`** (e.g., 'Overall B2C Compliant Met')")
                    st.markdown(f"    - **Explanation:** Indicates if all applicable rules (1, 2, and 3) for that price type (B2C or B2B) are satisfied. A rule is 'satisfied' for overall compliance if its condition is True OR if it's 'N/A' (blank in table due to missing prerequisite data, which is a non-blocking N/A). If any individual rule is definitively 'False' (red in table), then 'Overall Compliant Met' is 'False'. Otherwise (all rules are True or non-blocking N/A), it's 'True'.")
                    st.markdown("--- ")
                    
                    st.markdown("**Rule Column Display Key:**")
                    st.markdown("- <span style='background-color:lightgreen;color:darkgreen;font-weight:bold;padding:2px 5px;border-radius:3px;'>True</span>: The business rule is satisfied.", unsafe_allow_html=True)
                    st.markdown("- <span style='background-color:lightcoral;color:darkred;font-weight:bold;padding:2px 5px;border-radius:3px;'>False</span>: The business rule is not satisfied.", unsafe_allow_html=True)
                    st.markdown("- **Blank/Default**: Not Applicable (e.g., essential data like MAP, Cost, or Competitor Price was missing) or the new price could not be calculated.")
                    st.caption("Note: 'N/A' in the underlying data (e.g., 'N/A (No Comp)') results in a Blank/Default display for that rule. An 'Overall Compliant' status of 'N/A' means one or more rules could not be definitively evaluated. A False in 'Overall Compliant' means at least one rule was definitively False, or a new price could not be calculated.")
                
                st.markdown("--- ") # Separator before tables

                # --- Helper function to transform and style rule columns ---
                def format_and_style_rules_df(df_for_styling, rule_column_names_to_style):
                    # df_for_styling is a DataFrame (ideally a copy) that contains the rule columns to be styled.
                    # rule_column_names_to_style is a list of the names of these rule columns as they appear in df_for_styling.
                    
                    for rule_col_name in rule_column_names_to_style:
                        if rule_col_name in df_for_styling.columns:
                            # Ensure the column is of object dtype to handle mixed types (True, False, strings like "N/A...")
                            df_for_styling[rule_col_name] = df_for_styling[rule_col_name].astype(object)

                            # Store original N/A string types to preserve them
                            na_string_mask = df_for_styling[rule_col_name].apply(lambda x: isinstance(x, str) and "N/A" in x)
                            na_strings = df_for_styling.loc[na_string_mask, rule_col_name].copy() # Use .copy() to avoid SettingWithCopyWarning

                            # Convert booleans to "True"/"False" strings, keep others as is for now
                            df_for_styling[rule_col_name] = df_for_styling[rule_col_name].apply(
                                lambda x: "True" if x is True else ("False" if x is False else x)
                            )
                            # Restore original N/A strings
                            df_for_styling.loc[na_string_mask, rule_col_name] = na_strings
                            
                            # Convert any remaining pd.NA or None to empty strings for blank display
                            # Also convert any non-"True"/"False"/"N/A..." strings to blank if necessary (though unlikely with current logic)
                            mask_to_blank = df_for_styling[rule_col_name].apply(lambda x: pd.isna(x) or x not in ["True", "False"] and not (isinstance(x, str) and "N/A" in x))
                            df_for_styling.loc[mask_to_blank, rule_col_name] = ""

                    def style_rule_cell(val):
                        if val == "True":
                            return 'background-color:lightgreen;color:darkgreen;font-weight:bold;'
                        elif val == "False":
                            return 'background-color:lightcoral;color:darkred;font-weight:bold;'
                        # Blank for N/A strings or actual empty strings from pd.NA/None
                        elif val == "" or (isinstance(val, str) and "N/A" in val): 
                            return '' 
                        return ''

                    final_columns_to_style_in_df = [col for col in rule_column_names_to_style if col in df_for_styling.columns]
                    
                    if final_columns_to_style_in_df:
                        return df_for_styling.style.applymap(style_rule_cell, subset=final_columns_to_style_in_df)
                    return df_for_styling.style # Should not happen if rule_column_names_to_style is not empty and columns exist

                # --- B2C Simulation Table ---
                st.subheader("📈 B2C Simulation Results")
                
                b2c_original_rule_cols = ['RC Meets Rule 1', 'RC Meets Rule 2', 'RC Meets Rule 3', 'RC Overall Compliant']
                b2c_display_rule_names = ['Rule 1: vs B2C Comp', 'Rule 2: vs MAP', 'Rule 3: Margin ≥ 20%', 'Overall B2C Compliant']
                b2c_current_rule_metric_cols = ['Current RC Meets Rule 1', 'Current RC Meets Rule 2', 'Current RC Meets Rule 3', 'Current RC Overall Compliant'] # Added for KPI calculation

                base_cols_b2c = ['ITEM NAME', 'SKU'] # Added SKU
                if "Year/Model" in df_results.columns and st.session_state.get("dashboard_year_models"):
                    base_cols_b2c.append("Year/Model")
                
                # current_b2c_rule_cols are sourced from df_results for KPI calculation;
                # their inclusion in df_b2c_for_styling (via actual_cols_b2c_present) is now ensured if they are in df_results.
                # Display in the main table is managed by column_config and format_and_style_rules_df.

                cols_to_select_b2c = base_cols_b2c + [
                    RC_PRICE_COLUMN, 'Current RC Margin (%)', 'New RC Price', 'New RC Margin (%)'
                ] + b2c_original_rule_cols + b2c_current_rule_metric_cols + [ # Added current rule columns for KPI calculation
                    'Min B2C Competitor Price', 'MAP', 'Unit Cost'
                ]
                
                actual_cols_b2c_present = [col for col in cols_to_select_b2c if col in df_results.columns]
                # df_b2c_for_styling is used for KPI calculation AND styling. format_and_style_rules_df will modify it.
                df_b2c_for_styling = df_results[actual_cols_b2c_present].copy() 

                if not df_b2c_for_styling.empty:
                    # --- Calculate Current B2C KPIs (Averages) ---
                    avg_current_gross_margin_b2c = pd.to_numeric(df_b2c_for_styling['Current RC Margin (%)'], errors='coerce').mean()
                    
                    avg_current_markup_cost_b2c = pd.NA
                    if RC_PRICE_COLUMN in df_b2c_for_styling and 'Unit Cost' in df_b2c_for_styling:
                        original_price_b2c = pd.to_numeric(df_b2c_for_styling[RC_PRICE_COLUMN], errors='coerce')
                        unit_cost_b2c = pd.to_numeric(df_b2c_for_styling['Unit Cost'], errors='coerce')
                        valid_markup_cost_mask_current_b2c = unit_cost_b2c.notna() & (unit_cost_b2c != 0) & original_price_b2c.notna()
                        if valid_markup_cost_mask_current_b2c.any():
                            markup_cost_current_b2c = ((original_price_b2c[valid_markup_cost_mask_current_b2c] - unit_cost_b2c[valid_markup_cost_mask_current_b2c]) / unit_cost_b2c[valid_markup_cost_mask_current_b2c]) * 100
                            avg_current_markup_cost_b2c = markup_cost_current_b2c.mean()

                    avg_current_markup_map_b2c = pd.NA
                    if RC_PRICE_COLUMN in df_b2c_for_styling and 'MAP' in df_b2c_for_styling:
                        original_price_b2c = pd.to_numeric(df_b2c_for_styling[RC_PRICE_COLUMN], errors='coerce')
                        map_price_b2c = pd.to_numeric(df_b2c_for_styling['MAP'], errors='coerce')
                        valid_markup_map_mask_current_b2c = map_price_b2c.notna() & (map_price_b2c != 0) & original_price_b2c.notna()
                        if valid_markup_map_mask_current_b2c.any():
                            markup_map_current_b2c = ((original_price_b2c[valid_markup_map_mask_current_b2c] - map_price_b2c[valid_markup_map_mask_current_b2c]) / map_price_b2c[valid_markup_map_mask_current_b2c]) * 100
                            avg_current_markup_map_b2c = markup_map_current_b2c.mean()
                    
                    current_rule_1_met_b2c = (df_b2c_for_styling['Current RC Meets Rule 1'] == True).sum() if 'Current RC Meets Rule 1' in df_b2c_for_styling else 0
                    current_rule_2_met_b2c = (df_b2c_for_styling['Current RC Meets Rule 2'] == True).sum() if 'Current RC Meets Rule 2' in df_b2c_for_styling else 0
                    current_rule_3_met_b2c = (df_b2c_for_styling['Current RC Meets Rule 3'] == True).sum() if 'Current RC Meets Rule 3' in df_b2c_for_styling else 0
                    current_overall_compliant_b2c = (df_b2c_for_styling['Current RC Overall Compliant'] == True).sum() if 'Current RC Overall Compliant' in df_b2c_for_styling else 0
                    # --- End Current B2C KPIs ---

                    # --- Calculate Simulated B2C KPIs (Averages) ---
                    avg_sim_gross_margin_b2c = pd.to_numeric(df_b2c_for_styling['New RC Margin (%)'], errors='coerce').mean() # Renamed from avg_margin_b2c
                    
                    avg_sim_markup_cost_b2c = pd.NA # Renamed from avg_markup_cost_b2c
                    if 'New RC Price' in df_b2c_for_styling and 'Unit Cost' in df_b2c_for_styling:
                        new_price_b2c = pd.to_numeric(df_b2c_for_styling['New RC Price'], errors='coerce')
                        unit_cost_b2c_sim = pd.to_numeric(df_b2c_for_styling['Unit Cost'], errors='coerce') # Ensure a different variable if used above
                        valid_markup_cost_mask_sim_b2c = unit_cost_b2c_sim.notna() & (unit_cost_b2c_sim != 0) & new_price_b2c.notna()
                        if valid_markup_cost_mask_sim_b2c.any():
                            markup_cost_sim_b2c = ((new_price_b2c[valid_markup_cost_mask_sim_b2c] - unit_cost_b2c_sim[valid_markup_cost_mask_sim_b2c]) / unit_cost_b2c_sim[valid_markup_cost_mask_sim_b2c]) * 100
                            avg_sim_markup_cost_b2c = markup_cost_sim_b2c.mean()

                    avg_sim_markup_map_b2c = pd.NA # Renamed from avg_markup_map_b2c
                    if 'New RC Price' in df_b2c_for_styling and 'MAP' in df_b2c_for_styling:
                        new_price_b2c_sim = pd.to_numeric(df_b2c_for_styling['New RC Price'], errors='coerce')
                        map_price_b2c_sim = pd.to_numeric(df_b2c_for_styling['MAP'], errors='coerce')
                        valid_markup_map_mask_sim_b2c = map_price_b2c_sim.notna() & (map_price_b2c_sim != 0) & new_price_b2c_sim.notna()
                        if valid_markup_map_mask_sim_b2c.any():
                            markup_map_sim_b2c = ((new_price_b2c_sim[valid_markup_map_mask_sim_b2c] - map_price_b2c_sim[valid_markup_map_mask_sim_b2c]) / map_price_b2c_sim[valid_markup_map_mask_sim_b2c]) * 100
                            avg_sim_markup_map_b2c = markup_map_sim_b2c.mean()

                    sim_rule_1_met_b2c = (df_b2c_for_styling['RC Meets Rule 1'] == True).sum() if 'RC Meets Rule 1' in df_b2c_for_styling else 0
                    sim_rule_2_met_b2c = (df_b2c_for_styling['RC Meets Rule 2'] == True).sum() if 'RC Meets Rule 2' in df_b2c_for_styling else 0
                    sim_rule_3_met_b2c = (df_b2c_for_styling['RC Meets Rule 3'] == True).sum() if 'RC Meets Rule 3' in df_b2c_for_styling else 0
                    sim_overall_compliant_b2c = (df_b2c_for_styling['RC Overall Compliant'] == True).sum() if 'RC Overall Compliant' in df_b2c_for_styling else 0
                    # --- End Simulated B2C KPIs ---

                    # --- UI Structure for B2C KPIs (Consolidated) ---
                    with st.expander("📊 B2C Simulation: KPIs & Impact vs. Current", expanded=True):
                        kpi_sim_row1_b2c = st.columns(3)
                        # Calculate deltas for simulated KPIs
                        # delta_gross_margin_b2c = avg_sim_gross_margin_b2c - avg_current_gross_margin_b2c if pd.notna(avg_sim_gross_margin_b2c) and pd.notna(avg_current_gross_margin_b2c) else None
                        # delta_markup_cost_b2c = avg_sim_markup_cost_b2c - avg_current_markup_cost_b2c if pd.notna(avg_sim_markup_cost_b2c) and pd.notna(avg_current_markup_cost_b2c) else None
                        # delta_markup_map_b2c = avg_sim_markup_map_b2c - avg_current_markup_map_b2c if pd.notna(avg_sim_markup_map_b2c) and pd.notna(avg_current_markup_map_b2c) else None

                        kpi_sim_row1_b2c[0].metric("Avg. Gross Margin", 
                                                 f"{avg_sim_gross_margin_b2c:.2f}%" if pd.notna(avg_sim_gross_margin_b2c) else "N/A",
                                                 delta=get_delta_string_for_metric(avg_sim_gross_margin_b2c, avg_current_gross_margin_b2c, kind="percentage"),
                                                 help="Simulated Average: ((New RC Price - Unit Cost) / New RC Price) * 100")
                        kpi_sim_row1_b2c[1].metric("Avg. Markup on Cost", 
                                                 f"{avg_sim_markup_cost_b2c:.2f}%" if pd.notna(avg_sim_markup_cost_b2c) else "N/A",
                                                 delta=get_delta_string_for_metric(avg_sim_markup_cost_b2c, avg_current_markup_cost_b2c, kind="percentage"),
                                                 help="Simulated Average: ((New RC Price - Unit Cost) / Unit Cost) * 100")
                        kpi_sim_row1_b2c[2].metric("Avg. Markup on MAP", 
                                                 f"{avg_sim_markup_map_b2c:.2f}%" if pd.notna(avg_sim_markup_map_b2c) else "N/A",
                                                 delta=get_delta_string_for_metric(avg_sim_markup_map_b2c, avg_current_markup_map_b2c, kind="percentage"),
                                                 help="Simulated Average: ((New RC Price - MAP) / MAP) * 100")
                        
                        kpi_sim_row2_b2c = st.columns(4)
                        # Calculate deltas for rule compliance counts
                        # delta_rule1_met_b2c = sim_rule_1_met_b2c - current_rule_1_met_b2c if pd.notna(sim_rule_1_met_b2c) and pd.notna(current_rule_1_met_b2c) else None
                        # delta_rule2_met_b2c = sim_rule_2_met_b2c - current_rule_2_met_b2c if pd.notna(sim_rule_2_met_b2c) and pd.notna(current_rule_2_met_b2c) else None
                        # delta_rule3_met_b2c = sim_rule_3_met_b2c - current_rule_3_met_b2c if pd.notna(sim_rule_3_met_b2c) and pd.notna(current_rule_3_met_b2c) else None
                        # delta_overall_compliant_b2c = sim_overall_compliant_b2c - current_overall_compliant_b2c if pd.notna(sim_overall_compliant_b2c) and pd.notna(current_overall_compliant_b2c) else None

                        kpi_sim_row2_b2c[0].metric(f"{b2c_display_rule_names[0]} Met", f"{sim_rule_1_met_b2c:,}",
                                                 delta=get_delta_string_for_metric(sim_rule_1_met_b2c, current_rule_1_met_b2c, kind="count"))
                        kpi_sim_row2_b2c[1].metric(f"{b2c_display_rule_names[1]} Met", f"{sim_rule_2_met_b2c:,}",
                                                 delta=get_delta_string_for_metric(sim_rule_2_met_b2c, current_rule_2_met_b2c, kind="count"))
                        kpi_sim_row2_b2c[2].metric(f"{b2c_display_rule_names[2]} Met", f"{sim_rule_3_met_b2c:,}",
                                                 delta=get_delta_string_for_metric(sim_rule_3_met_b2c, current_rule_3_met_b2c, kind="count"))
                        kpi_sim_row2_b2c[3].metric(f"{b2c_display_rule_names[3]} Met", f"{sim_overall_compliant_b2c:,}",
                                                 delta=get_delta_string_for_metric(sim_overall_compliant_b2c, current_overall_compliant_b2c, kind="count"))
                    st.markdown("--- ") # Visual separator after KPIs

                    # Identify which of the original rule columns are actually present in the selected df for styling
                    active_b2c_original_rule_cols = [col for col in b2c_original_rule_cols if col in df_b2c_for_styling.columns]
                    
                    # Prepare DataFrame for display by removing current rule columns
                    cols_for_b2c_display = [col for col in df_b2c_for_styling.columns if col not in b2c_current_rule_metric_cols]
                    df_b2c_for_display_table = df_b2c_for_styling[cols_for_b2c_display].copy()

                    b2c_styler = format_and_style_rules_df(df_b2c_for_display_table, active_b2c_original_rule_cols)
                    
                    column_config_b2c = {original: st.column_config.TextColumn(new_display_name) 
                                         for original, new_display_name in zip(b2c_original_rule_cols, b2c_display_rule_names) 
                                         if original in df_b2c_for_display_table.columns} # only configure present columns
                    # Add renaming for New RC Price
                    if 'New RC Price' in df_b2c_for_display_table.columns:
                        column_config_b2c['New RC Price'] = st.column_config.NumberColumn("New RC Price (after discount)", format="%.2f")
                    # Configure other numeric columns for B2C if they exist
                    if RC_PRICE_COLUMN in df_b2c_for_display_table.columns:
                        column_config_b2c[RC_PRICE_COLUMN] = st.column_config.NumberColumn(format="%.2f")
                    if 'Current RC Margin (%)' in df_b2c_for_display_table.columns:
                        column_config_b2c['Current RC Margin (%)'] = st.column_config.NumberColumn(format="%.2f%%")
                    if 'New RC Margin (%)' in df_b2c_for_display_table.columns:
                        column_config_b2c['New RC Margin (%)'] = st.column_config.NumberColumn(format="%.2f%%")
                    if 'Min B2C Competitor Price' in df_b2c_for_display_table.columns:
                        column_config_b2c['Min B2C Competitor Price'] = st.column_config.NumberColumn(format="%.2f")
                    if 'MAP' in df_b2c_for_display_table.columns:
                        column_config_b2c['MAP'] = st.column_config.NumberColumn(format="%.2f")
                    if 'Unit Cost' in df_b2c_for_display_table.columns:
                        column_config_b2c['Unit Cost'] = st.column_config.NumberColumn(format="%.2f")

                    st.dataframe(b2c_styler, use_container_width=True, hide_index=True, column_config=column_config_b2c)

                    # Button to add B2C results to scenario
                    if not df_b2c_for_display_table.empty:
                        if st.button("➕ Add B2C Preview to Scenario", key="add_b2c_to_scenario"):
                            # Store relevant info: df, discount applied, type
                            # We need to store the state of df_results for these items *with the current discount applied*
                            # Also capture the discount_rc itself for display
                            # Ensure we are taking a snapshot of the data as it is at this point
                            # The df_b2c_for_display_table has the right columns for display, 
                            # but for calculation integrity and to avoid re-calculating rules, we might need more columns or the original df_results slice
                            # For simplicity in display and management, let's store what's in df_b2c_for_display_table plus essential identifiers and the discount.
                            
                            # Create a copy of the relevant slice from df_results to ensure all calculation columns are there
                            # and then select only the columns that were meant for display in the original B2C table, plus discount info.
                            # df_results already contains 'New RC Price', 'New RC Margin (%)' and rule columns based on current discount_rc
                            
                            # Get the ITEM NAMEs from the current b2c display table to filter df_results
                            items_in_current_b2c_view = df_b2c_for_display_table['ITEM NAME'].unique()
                            df_to_stage_b2c = df_results[df_results['ITEM NAME'].isin(items_in_current_b2c_view)].copy()
                            df_to_stage_b2c['Applied Discount (%)'] = discount_rc
                            df_to_stage_b2c['Scenario Type'] = 'B2C'
                            
                            # Re-select/re-order columns similar to df_b2c_for_display_table for consistency in the scenario view
                            # plus the new 'Applied Discount (%)' and 'Scenario Type'
                            # Also, ensure all original rule columns for B2C are present for potential later aggregation or detailed view
                            cols_to_keep_for_staging_b2c = ([col for col in cols_for_b2c_display if col in df_to_stage_b2c.columns] 
                                + ['Applied Discount (%)', 'Scenario Type'] + [col for col in b2c_original_rule_cols if col in df_to_stage_b2c.columns])
                            # Remove duplicates while preserving order if any column was in both lists                           
                            seen = set()
                            cols_to_keep_for_staging_b2c_unique = [x for x in cols_to_keep_for_staging_b2c if not (x in seen or seen.add(x))]
                            st.session_state.staged_scenario_items.append(df_to_stage_b2c[cols_to_keep_for_staging_b2c_unique])
                            st.success(f"{len(df_to_stage_b2c)} B2C items added to scenario with {discount_rc}% discount")
                        else:
                            st.info("No B2C simulation data to display based on current selections and applied discounts")

                # --- B2B Simulation Table ---
                if RC_B2B_PRICE_COLUMN in df_results.columns and 'New RC B2B Price' in df_results.columns:
                    st.markdown("--- ") # Separator
                    st.subheader("📈 B2B Simulation Results")
                    
                    b2b_original_rule_cols = ['RC B2B Meets Rule 1', 'RC B2B Meets Rule 2', 'RC B2B Meets Rule 3', 'RC B2B Overall Compliant']
                    b2b_display_rule_names = ['Rule 1: vs B2B Comp', 'Rule 2: vs MAP', 'Rule 3: Margin ≥ 20%', 'Overall B2B Compliant']
                    b2b_current_rule_metric_cols = ['Current RC B2B Meets Rule 1', 'Current RC B2B Meets Rule 2', 'Current RC B2B Meets Rule 3', 'Current RC B2B Overall Compliant'] # Added for KPI calculation

                    base_cols_b2b = ['ITEM NAME', 'SKU'] # Added SKU
                    if "Year/Model" in df_results.columns and st.session_state.get("dashboard_year_models"):
                        base_cols_b2b.append("Year/Model")

                    # current_b2b_rule_cols are sourced from df_results for KPI calculation;
                    # their inclusion in df_b2b_for_styling (via actual_cols_b2b_present) is now ensured if they are in df_results.
                    # Display in the main table is managed by column_config and format_and_style_rules_df.

                    cols_to_select_b2b = base_cols_b2b + [ # Ensuring this uses base_cols_b2b
                        RC_B2B_PRICE_COLUMN, 'Current RC B2B Margin (%)', 'New RC B2B Price', 'New RC B2B Margin (%)'
                    ] + b2b_original_rule_cols + b2b_current_rule_metric_cols + [ # Added current rule columns for KPI calculation
                        'Min B2B Competitor Price', 'MAP', 'Unit Cost'
                    ]

                    actual_cols_b2b_present = [col for col in cols_to_select_b2b if col in df_results.columns]
                    df_b2b_for_styling = df_results[actual_cols_b2b_present].copy()
                    
                    if not df_b2b_for_styling.empty:
                        # --- Calculate Current B2B KPIs (Averages) ---
                        avg_current_gross_margin_b2b = pd.to_numeric(df_b2b_for_styling['Current RC B2B Margin (%)'], errors='coerce').mean()

                        avg_current_markup_cost_b2b = pd.NA
                        if RC_B2B_PRICE_COLUMN in df_b2b_for_styling and 'Unit Cost' in df_b2b_for_styling:
                            original_price_b2b = pd.to_numeric(df_b2b_for_styling[RC_B2B_PRICE_COLUMN], errors='coerce')
                            unit_cost_b2b = pd.to_numeric(df_b2b_for_styling['Unit Cost'], errors='coerce')
                            valid_markup_cost_mask_current_b2b = unit_cost_b2b.notna() & (unit_cost_b2b != 0) & original_price_b2b.notna()
                            if valid_markup_cost_mask_current_b2b.any():
                                markup_cost_current_b2b = ((original_price_b2b[valid_markup_cost_mask_current_b2b] - unit_cost_b2b[valid_markup_cost_mask_current_b2b]) / unit_cost_b2b[valid_markup_cost_mask_current_b2b]) * 100
                                avg_current_markup_cost_b2b = markup_cost_current_b2b.mean()
                        
                        avg_current_markup_map_b2b = pd.NA
                        if RC_B2B_PRICE_COLUMN in df_b2b_for_styling and 'MAP' in df_b2b_for_styling:
                            original_price_b2b = pd.to_numeric(df_b2b_for_styling[RC_B2B_PRICE_COLUMN], errors='coerce')
                            map_price_b2b = pd.to_numeric(df_b2b_for_styling['MAP'], errors='coerce')
                            valid_markup_map_mask_current_b2b = map_price_b2b.notna() & (map_price_b2b != 0) & original_price_b2b.notna()
                            if valid_markup_map_mask_current_b2b.any():
                                markup_map_current_b2b = ((original_price_b2b[valid_markup_map_mask_current_b2b] - map_price_b2b[valid_markup_map_mask_current_b2b]) / map_price_b2b[valid_markup_map_mask_current_b2b]) * 100
                                avg_current_markup_map_b2b = markup_map_current_b2b.mean()

                        current_rule_1_met_b2b = (df_b2b_for_styling['Current RC B2B Meets Rule 1'] == True).sum() if 'Current RC B2B Meets Rule 1' in df_b2b_for_styling else 0
                        current_rule_2_met_b2b = (df_b2b_for_styling['Current RC B2B Meets Rule 2'] == True).sum() if 'Current RC B2B Meets Rule 2' in df_b2b_for_styling else 0
                        current_rule_3_met_b2b = (df_b2b_for_styling['Current RC B2B Meets Rule 3'] == True).sum() if 'Current RC B2B Meets Rule 3' in df_b2b_for_styling else 0
                        current_overall_compliant_b2b = (df_b2b_for_styling['Current RC B2B Overall Compliant'] == True).sum() if 'Current RC B2B Overall Compliant' in df_b2b_for_styling else 0
                        # --- End Current B2B KPIs ---

                        # --- Calculate Simulated B2B KPIs (Averages) ---
                        avg_sim_gross_margin_b2b = pd.to_numeric(df_b2b_for_styling['New RC B2B Margin (%)'], errors='coerce').mean() # Renamed from avg_margin_b2b

                        avg_sim_markup_cost_b2b = pd.NA # Renamed from avg_markup_cost_b2b
                        if 'New RC B2B Price' in df_b2b_for_styling and 'Unit Cost' in df_b2b_for_styling:
                            new_price_b2b = pd.to_numeric(df_b2b_for_styling['New RC B2B Price'], errors='coerce')
                            unit_cost_b2b_sim = pd.to_numeric(df_b2b_for_styling['Unit Cost'], errors='coerce')
                            valid_markup_cost_mask_sim_b2b = unit_cost_b2b_sim.notna() & (unit_cost_b2b_sim != 0) & new_price_b2b.notna()
                            if valid_markup_cost_mask_sim_b2b.any():
                                markup_cost_sim_b2b = ((new_price_b2b[valid_markup_cost_mask_sim_b2b] - unit_cost_b2b_sim[valid_markup_cost_mask_sim_b2b]) / unit_cost_b2b_sim[valid_markup_cost_mask_sim_b2b]) * 100
                                avg_sim_markup_cost_b2b = markup_cost_sim_b2b.mean()

                        avg_sim_markup_map_b2b = pd.NA # Renamed from avg_markup_map_b2b
                        if 'New RC B2B Price' in df_b2b_for_styling and 'MAP' in df_b2b_for_styling:
                            new_price_b2b_sim = pd.to_numeric(df_b2b_for_styling['New RC B2B Price'], errors='coerce')
                            map_price_b2b_sim = pd.to_numeric(df_b2b_for_styling['MAP'], errors='coerce')
                            valid_markup_map_mask_sim_b2b = map_price_b2b_sim.notna() & (map_price_b2b_sim != 0) & new_price_b2b_sim.notna()
                            if valid_markup_map_mask_sim_b2b.any():
                                markup_map_sim_b2b = ((new_price_b2b_sim[valid_markup_map_mask_sim_b2b] - map_price_b2b_sim[valid_markup_map_mask_sim_b2b]) / map_price_b2b_sim[valid_markup_map_mask_sim_b2b]) * 100
                                avg_sim_markup_map_b2b = markup_map_sim_b2b.mean()
                        
                        sim_rule_1_met_b2b = (df_b2b_for_styling['RC B2B Meets Rule 1'] == True).sum() if 'RC B2B Meets Rule 1' in df_b2b_for_styling else 0
                        sim_rule_2_met_b2b = (df_b2b_for_styling['RC B2B Meets Rule 2'] == True).sum() if 'RC B2B Meets Rule 2' in df_b2b_for_styling else 0
                        sim_rule_3_met_b2b = (df_b2b_for_styling['RC B2B Meets Rule 3'] == True).sum() if 'RC B2B Meets Rule 3' in df_b2b_for_styling else 0
                        sim_overall_compliant_b2b = (df_b2b_for_styling['RC B2B Overall Compliant'] == True).sum() if 'RC B2B Overall Compliant' in df_b2b_for_styling else 0
                        # --- End Simulated B2B KPIs ---

                        # --- UI Structure for B2B KPIs (Consolidated) ---
                        with st.expander("📊 B2B Simulation: KPIs & Impact vs. Current", expanded=True):
                            kpi_sim_row1_b2b = st.columns(3)
                            # Calculate deltas for simulated B2B KPIs
                            # delta_gross_margin_b2b = avg_sim_gross_margin_b2b - avg_current_gross_margin_b2b if pd.notna(avg_sim_gross_margin_b2b) and pd.notna(avg_current_gross_margin_b2b) else None
                            # delta_markup_cost_b2b = avg_sim_markup_cost_b2b - avg_current_markup_cost_b2b if pd.notna(avg_sim_markup_cost_b2b) and pd.notna(avg_current_markup_cost_b2b) else None
                            # delta_markup_map_b2b = avg_sim_markup_map_b2b - avg_current_markup_map_b2b if pd.notna(avg_sim_markup_map_b2b) and pd.notna(avg_current_markup_map_b2b) else None

                            kpi_sim_row1_b2b[0].metric("Avg. Gross Margin B2B", 
                                                     f"{avg_sim_gross_margin_b2b:.2f}%" if pd.notna(avg_sim_gross_margin_b2b) else "N/A",
                                                     delta=get_delta_string_for_metric(avg_sim_gross_margin_b2b, avg_current_gross_margin_b2b, kind="percentage"),
                                                     help="Simulated Average: ((New RC B2B Price - Unit Cost) / New RC B2B Price) * 100")
                            kpi_sim_row1_b2b[1].metric("Avg. Markup on Cost B2B", 
                                                     f"{avg_sim_markup_cost_b2b:.2f}%" if pd.notna(avg_sim_markup_cost_b2b) else "N/A",
                                                     delta=get_delta_string_for_metric(avg_sim_markup_cost_b2b, avg_current_markup_cost_b2b, kind="percentage"),
                                                     help="Simulated Average: ((New RC B2B Price - Unit Cost) / Unit Cost) * 100")
                            kpi_sim_row1_b2b[2].metric("Avg. Markup on MAP B2B", 
                                                     f"{avg_sim_markup_map_b2b:.2f}%" if pd.notna(avg_sim_markup_map_b2b) else "N/A",
                                                     delta=get_delta_string_for_metric(avg_sim_markup_map_b2b, avg_current_markup_map_b2b, kind="percentage"),
                                                     help="Simulated Average: ((New RC B2B Price - MAP) / MAP) * 100")

                            kpi_sim_row2_b2b = st.columns(4)
                            # Calculate deltas for B2B rule compliance counts
                            # delta_rule1_met_b2b = sim_rule_1_met_b2b - current_rule_1_met_b2b if pd.notna(sim_rule_1_met_b2b) and pd.notna(current_rule_1_met_b2b) else None
                            # delta_rule2_met_b2b = sim_rule_2_met_b2b - current_rule_2_met_b2b if pd.notna(sim_rule_2_met_b2b) and pd.notna(current_rule_2_met_b2b) else None
                            # delta_rule3_met_b2b = sim_rule_3_met_b2b - current_rule_3_met_b2b if pd.notna(sim_rule_3_met_b2b) and pd.notna(current_rule_3_met_b2b) else None
                            # delta_overall_compliant_b2b = sim_overall_compliant_b2b - current_overall_compliant_b2b if pd.notna(sim_overall_compliant_b2b) and pd.notna(current_overall_compliant_b2b) else None

                            kpi_sim_row2_b2b[0].metric(f"{b2b_display_rule_names[0]} Met", f"{sim_rule_1_met_b2b:,}",
                                                     delta=get_delta_string_for_metric(sim_rule_1_met_b2b, current_rule_1_met_b2b, kind="count"))
                            kpi_sim_row2_b2b[1].metric(f"{b2b_display_rule_names[1]} Met", f"{sim_rule_2_met_b2b:,}",
                                                     delta=get_delta_string_for_metric(sim_rule_2_met_b2b, current_rule_2_met_b2b, kind="count"))
                            kpi_sim_row2_b2b[2].metric(f"{b2b_display_rule_names[2]} Met", f"{sim_rule_3_met_b2b:,}",
                                                     delta=get_delta_string_for_metric(sim_rule_3_met_b2b, current_rule_3_met_b2b, kind="count"))
                            kpi_sim_row2_b2b[3].metric(f"{b2b_display_rule_names[3]} Met", f"{sim_overall_compliant_b2b:,}",
                                                     delta=get_delta_string_for_metric(sim_overall_compliant_b2b, current_overall_compliant_b2b, kind="count"))
                        st.markdown("--- ") # Visual separator after KPIs

                        active_b2b_original_rule_cols = [col for col in b2b_original_rule_cols if col in df_b2b_for_styling.columns]

                        # Prepare DataFrame for display by removing current rule columns
                        cols_for_b2b_display = [col for col in df_b2b_for_styling.columns if col not in b2b_current_rule_metric_cols]
                        df_b2b_for_display_table = df_b2b_for_styling[cols_for_b2b_display].copy()
                        
                        b2b_styler = format_and_style_rules_df(df_b2b_for_display_table, active_b2b_original_rule_cols)
                        
                        column_config_b2b = {original: st.column_config.TextColumn(new_display_name) 
                                             for original, new_display_name in zip(b2b_original_rule_cols, b2b_display_rule_names) 
                                             if original in df_b2b_for_display_table.columns}
                        # Add renaming for New RC B2B Price
                        if 'New RC B2B Price' in df_b2b_for_display_table.columns:
                            column_config_b2b['New RC B2B Price'] = st.column_config.NumberColumn("New Price (after discount)", format="%.2f")
                        # Configure other numeric columns for B2B if they exist
                        if RC_B2B_PRICE_COLUMN in df_b2b_for_display_table.columns:
                            column_config_b2b[RC_B2B_PRICE_COLUMN] = st.column_config.NumberColumn(format="%.2f")
                        if 'Current RC B2B Margin (%)' in df_b2b_for_display_table.columns:
                            column_config_b2b['Current RC B2B Margin (%)'] = st.column_config.NumberColumn(format="%.2f%%")
                        if 'New RC B2B Margin (%)' in df_b2b_for_display_table.columns:
                            column_config_b2b['New RC B2B Margin (%)'] = st.column_config.NumberColumn(format="%.2f%%")
                        if 'Min B2B Competitor Price' in df_b2b_for_display_table.columns:
                            column_config_b2b['Min B2B Competitor Price'] = st.column_config.NumberColumn(format="%.2f")
                        if 'MAP' in df_b2b_for_display_table.columns:
                            column_config_b2b['MAP'] = st.column_config.NumberColumn(format="%.2f")
                        if 'Unit Cost' in df_b2b_for_display_table.columns:
                            column_config_b2b['Unit Cost'] = st.column_config.NumberColumn(format="%.2f")
                        
                        st.dataframe(b2b_styler, use_container_width=True, hide_index=True, column_config=column_config_b2b)

                        # Button to add B2B results to scenario
                        if not df_b2b_for_display_table.empty:
                            if st.button("➕ Add B2B Preview to Scenario", key="add_b2b_to_scenario"):
                                items_in_current_b2b_view = df_b2b_for_display_table['ITEM NAME'].unique()
                                df_to_stage_b2b = df_results[df_results['ITEM NAME'].isin(items_in_current_b2b_view)].copy()
                                df_to_stage_b2b['Applied Discount (%)'] = discount_rc_b2b # Make sure this is the B2B discount
                                df_to_stage_b2b['Scenario Type'] = 'B2B'
                                
                                # Define columns to keep for B2B staging
                                cols_to_keep_for_staging_b2b = ([col for col in cols_for_b2b_display if col in df_to_stage_b2b.columns] 
                                    + ['Applied Discount (%)', 'Scenario Type'] + [col for col in b2b_original_rule_cols if col in df_to_stage_b2b.columns])
                                seen_b2b = set()
                                cols_to_keep_for_staging_b2b_unique = [x for x in cols_to_keep_for_staging_b2b if not (x in seen_b2b or seen_b2b.add(x))]
                                st.session_state.staged_scenario_items.append(df_to_stage_b2b[cols_to_keep_for_staging_b2b_unique])                                
                                st.success(f"{len(df_to_stage_b2b)} B2B items added to scenario with {discount_rc_b2b}% discount")                                                    
                        else:
                            st.info("No B2B simulation data to display based on current selections and applied discounts")
                    else:
                        st.info("No B2B simulation data to display based on current selections and applied discounts")
            else:
                st.info("No products match the current sidebar filters. Please adjust sidebar filters to populate products for simulation.")
                            
        # --- Consolidated Scenario Analysis Section ---
        st.markdown("--- ") # Visual Separator
        st.header("📊 Análisis de Escenario Consolidado")
        st.caption("Aquí se muestran los productos que has añadido al escenario con sus respectivos descuentos aplicados. Puedes eliminar lotes individualmente o limpiar todo el escenario.")

        # --- KPIs Consolidados del Escenario ---
        if st.session_state.staged_scenario_items: # Only show if there are items
            with st.expander("📊 KPIs Agregados del Escenario Completo", expanded=True):
                df_scenario_all_items = pd.concat(st.session_state.staged_scenario_items, ignore_index=True).copy()

                cost_col = 'Unit Cost'
                
                # Initialize Effective Columns
                df_scenario_all_items['Effective Current Price'] = pd.NA
                df_scenario_all_items['Effective New Price'] = pd.NA
                df_scenario_all_items['Effective Current Margin (%)'] = pd.NA
                df_scenario_all_items['Effective New Margin (%)'] = pd.NA

                b2c_mask_scenario = df_scenario_all_items['Scenario Type'] == 'B2C'
                b2b_mask_scenario = df_scenario_all_items['Scenario Type'] == 'B2B'

                # --- Effective Current Price ---
                if RC_PRICE_COLUMN in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2c_mask_scenario, 'Effective Current Price'] = df_scenario_all_items.loc[b2c_mask_scenario, RC_PRICE_COLUMN]
                if RC_B2B_PRICE_COLUMN in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2b_mask_scenario, 'Effective Current Price'] = df_scenario_all_items.loc[b2b_mask_scenario, RC_B2B_PRICE_COLUMN]

                # --- Effective New Price ---
                if 'New RC Price' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2c_mask_scenario, 'Effective New Price'] = df_scenario_all_items.loc[b2c_mask_scenario, 'New RC Price']
                if 'New RC B2B Price' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2b_mask_scenario, 'Effective New Price'] = df_scenario_all_items.loc[b2b_mask_scenario, 'New RC B2B Price']

                # --- Effective Margins ---
                if cost_col in df_scenario_all_items.columns:
                    numeric_cost_scenario = pd.to_numeric(df_scenario_all_items[cost_col], errors='coerce')
                    cost_is_valid_mask_scenario = numeric_cost_scenario.notna()

                    # Effective Current Margin
                    current_price_eff = pd.to_numeric(df_scenario_all_items['Effective Current Price'], errors='coerce')
                    current_price_valid_eff = current_price_eff.notna() & (current_price_eff != 0)
                    final_current_margin_mask_eff = cost_is_valid_mask_scenario & current_price_valid_eff
                    if final_current_margin_mask_eff.any():
                        df_scenario_all_items.loc[final_current_margin_mask_eff, 'Effective Current Margin (%)'] = \
                            ((current_price_eff[final_current_margin_mask_eff] - numeric_cost_scenario[final_current_margin_mask_eff]) / current_price_eff[final_current_margin_mask_eff]) * 100

                    # Effective New Margin
                    new_price_eff = pd.to_numeric(df_scenario_all_items['Effective New Price'], errors='coerce')
                    new_price_valid_eff = new_price_eff.notna() & (new_price_eff != 0)
                    final_new_margin_mask_eff = cost_is_valid_mask_scenario & new_price_valid_eff
                    if final_new_margin_mask_eff.any():
                        df_scenario_all_items.loc[final_new_margin_mask_eff, 'Effective New Margin (%)'] = \
                            ((new_price_eff[final_new_margin_mask_eff] - numeric_cost_scenario[final_new_margin_mask_eff]) / new_price_eff[final_new_margin_mask_eff]) * 100
                
                # --- Effective Markups ---
                map_col = 'MAP' # Ensure map_col is defined
                df_scenario_all_items['Effective Current Markup on Cost (%)'] = pd.NA
                df_scenario_all_items['Effective New Markup on Cost (%)'] = pd.NA
                df_scenario_all_items['Effective Current Markup on MAP (%)'] = pd.NA
                df_scenario_all_items['Effective New Markup on MAP (%)'] = pd.NA

                if cost_col in df_scenario_all_items.columns:
                    numeric_cost_scenario = pd.to_numeric(df_scenario_all_items[cost_col], errors='coerce')
                    cost_is_valid_and_nonzero_scenario = numeric_cost_scenario.notna() & (numeric_cost_scenario != 0)

                    # Effective Current Markup on Cost
                    current_price_eff = pd.to_numeric(df_scenario_all_items['Effective Current Price'], errors='coerce')
                    current_price_valid_eff_markup_cost = current_price_eff.notna() & cost_is_valid_and_nonzero_scenario
                    if current_price_valid_eff_markup_cost.any():
                        df_scenario_all_items.loc[current_price_valid_eff_markup_cost, 'Effective Current Markup on Cost (%)'] = \
                            ((current_price_eff[current_price_valid_eff_markup_cost] - numeric_cost_scenario[current_price_valid_eff_markup_cost]) / numeric_cost_scenario[current_price_valid_eff_markup_cost]) * 100

                    # Effective New Markup on Cost
                    new_price_eff = pd.to_numeric(df_scenario_all_items['Effective New Price'], errors='coerce')
                    new_price_valid_eff_markup_cost = new_price_eff.notna() & cost_is_valid_and_nonzero_scenario
                    if new_price_valid_eff_markup_cost.any():
                        df_scenario_all_items.loc[new_price_valid_eff_markup_cost, 'Effective New Markup on Cost (%)'] = \
                            ((new_price_eff[new_price_valid_eff_markup_cost] - numeric_cost_scenario[new_price_valid_eff_markup_cost]) / numeric_cost_scenario[new_price_valid_eff_markup_cost]) * 100

                if map_col in df_scenario_all_items.columns:
                    numeric_map_scenario = pd.to_numeric(df_scenario_all_items[map_col], errors='coerce')
                    map_is_valid_and_nonzero_scenario = numeric_map_scenario.notna() & (numeric_map_scenario != 0)

                    # Effective Current Markup on MAP
                    current_price_eff_map = pd.to_numeric(df_scenario_all_items['Effective Current Price'], errors='coerce') # Re-fetch to avoid altering current_price_eff used for cost markup
                    current_price_valid_eff_markup_map = current_price_eff_map.notna() & map_is_valid_and_nonzero_scenario
                    if current_price_valid_eff_markup_map.any():
                        df_scenario_all_items.loc[current_price_valid_eff_markup_map, 'Effective Current Markup on MAP (%)'] = \
                            ((current_price_eff_map[current_price_valid_eff_markup_map] - numeric_map_scenario[current_price_valid_eff_markup_map]) / numeric_map_scenario[current_price_valid_eff_markup_map]) * 100

                    # Effective New Markup on MAP
                    new_price_eff_map = pd.to_numeric(df_scenario_all_items['Effective New Price'], errors='coerce') # Re-fetch
                    new_price_valid_eff_markup_map = new_price_eff_map.notna() & map_is_valid_and_nonzero_scenario
                    if new_price_valid_eff_markup_map.any():
                        df_scenario_all_items.loc[new_price_valid_eff_markup_map, 'Effective New Markup on MAP (%)'] = \
                            ((new_price_eff_map[new_price_valid_eff_markup_map] - numeric_map_scenario[new_price_valid_eff_markup_map]) / numeric_map_scenario[new_price_valid_eff_markup_map]) * 100

                # --- Calculate Aggregated Average KPIs for Markups ---
                avg_scenario_current_markup_cost = pd.to_numeric(df_scenario_all_items['Effective Current Markup on Cost (%)'], errors='coerce').mean()
                avg_scenario_new_markup_cost = pd.to_numeric(df_scenario_all_items['Effective New Markup on Cost (%)'], errors='coerce').mean()
                avg_scenario_current_markup_map = pd.to_numeric(df_scenario_all_items['Effective Current Markup on MAP (%)'], errors='coerce').mean()
                avg_scenario_new_markup_map = pd.to_numeric(df_scenario_all_items['Effective New Markup on MAP (%)'], errors='coerce').mean()

                # --- Calculate Aggregated Average KPIs for Margins --- 
                avg_scenario_current_gross_margin = pd.to_numeric(df_scenario_all_items['Effective Current Margin (%)'], errors='coerce').mean()
                avg_scenario_new_gross_margin = pd.to_numeric(df_scenario_all_items['Effective New Margin (%)'], errors='coerce').mean()

                # --- Effective Rule Compliance --- 
                df_scenario_all_items['Effective Current Rule 1 Met'] = pd.NA
                df_scenario_all_items['Effective Current Rule 2 Met'] = pd.NA
                df_scenario_all_items['Effective Current Rule 3 Met'] = pd.NA
                df_scenario_all_items['Effective Current Overall Compliant Met'] = pd.NA
                df_scenario_all_items['Effective New Rule 1 Met'] = pd.NA
                df_scenario_all_items['Effective New Rule 2 Met'] = pd.NA
                df_scenario_all_items['Effective New Rule 3 Met'] = pd.NA
                df_scenario_all_items['Effective New Overall Compliant Met'] = pd.NA
                
                # b2c_mask_scenario and b2b_mask_scenario are already defined

                # Rule 1
                if 'Current RC Meets Rule 1' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2c_mask_scenario, 'Effective Current Rule 1 Met'] = df_scenario_all_items.loc[b2c_mask_scenario, 'Current RC Meets Rule 1']
                if 'Current RC B2B Meets Rule 1' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2b_mask_scenario, 'Effective Current Rule 1 Met'] = df_scenario_all_items.loc[b2b_mask_scenario, 'Current RC B2B Meets Rule 1']
                if 'RC Meets Rule 1' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2c_mask_scenario, 'Effective New Rule 1 Met'] = df_scenario_all_items.loc[b2c_mask_scenario, 'RC Meets Rule 1']
                if 'RC B2B Meets Rule 1' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2b_mask_scenario, 'Effective New Rule 1 Met'] = df_scenario_all_items.loc[b2b_mask_scenario, 'RC B2B Meets Rule 1']

                # Rule 2
                if 'Current RC Meets Rule 2' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2c_mask_scenario, 'Effective Current Rule 2 Met'] = df_scenario_all_items.loc[b2c_mask_scenario, 'Current RC Meets Rule 2']
                if 'Current RC B2B Meets Rule 2' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2b_mask_scenario, 'Effective Current Rule 2 Met'] = df_scenario_all_items.loc[b2b_mask_scenario, 'Current RC B2B Meets Rule 2']
                if 'RC Meets Rule 2' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2c_mask_scenario, 'Effective New Rule 2 Met'] = df_scenario_all_items.loc[b2c_mask_scenario, 'RC Meets Rule 2']
                if 'RC B2B Meets Rule 2' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2b_mask_scenario, 'Effective New Rule 2 Met'] = df_scenario_all_items.loc[b2b_mask_scenario, 'RC B2B Meets Rule 2']

                # Rule 3
                if 'Current RC Meets Rule 3' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2c_mask_scenario, 'Effective Current Rule 3 Met'] = df_scenario_all_items.loc[b2c_mask_scenario, 'Current RC Meets Rule 3']
                if 'Current RC B2B Meets Rule 3' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2b_mask_scenario, 'Effective Current Rule 3 Met'] = df_scenario_all_items.loc[b2b_mask_scenario, 'Current RC B2B Meets Rule 3']
                if 'RC Meets Rule 3' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2c_mask_scenario, 'Effective New Rule 3 Met'] = df_scenario_all_items.loc[b2c_mask_scenario, 'RC Meets Rule 3']
                if 'RC B2B Meets Rule 3' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2b_mask_scenario, 'Effective New Rule 3 Met'] = df_scenario_all_items.loc[b2b_mask_scenario, 'RC B2B Meets Rule 3']

                # Overall Compliant
                if 'Current RC Overall Compliant' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2c_mask_scenario, 'Effective Current Overall Compliant Met'] = df_scenario_all_items.loc[b2c_mask_scenario, 'Current RC Overall Compliant']
                if 'Current RC B2B Overall Compliant' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2b_mask_scenario, 'Effective Current Overall Compliant Met'] = df_scenario_all_items.loc[b2b_mask_scenario, 'Current RC B2B Overall Compliant']
                if 'RC Overall Compliant' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2c_mask_scenario, 'Effective New Overall Compliant Met'] = df_scenario_all_items.loc[b2c_mask_scenario, 'RC Overall Compliant']
                if 'RC B2B Overall Compliant' in df_scenario_all_items.columns:
                    df_scenario_all_items.loc[b2b_mask_scenario, 'Effective New Overall Compliant Met'] = df_scenario_all_items.loc[b2b_mask_scenario, 'RC B2B Overall Compliant']

                # --- Calculate Aggregated Counts for Rule Compliance (Number of Trues) ---
                scenario_counts = {}
                rules_to_count = ['Rule 1', 'Rule 2', 'Rule 3', 'Overall Compliant']
                for rule_base_name in rules_to_count: 
                    scenario_counts[f'Current {rule_base_name} Met'] = (df_scenario_all_items[f'Effective Current {rule_base_name} Met'] == True).sum()
                    scenario_counts[f'New {rule_base_name} Met'] = (df_scenario_all_items[f'Effective New {rule_base_name} Met'] == True).sum()

                # --- Display Aggregated Scenario KPIs ---
                st.subheader("KPIs Agregados del Escenario")

                total_items_in_scenario = len(df_scenario_all_items)
                st.metric("Total de Productos en Escenario", f"{total_items_in_scenario:,}")

                # --- Financial KPIs ---
                kpi_cols_finance = st.columns(3)
                
                kpi_cols_finance[0].metric("Avg. Margen Bruto",
                                           f"{avg_scenario_new_gross_margin:.2f}%" if pd.notna(avg_scenario_new_gross_margin) else "N/A",
                                           delta=get_delta_string_for_metric(avg_scenario_new_gross_margin, avg_scenario_current_gross_margin, kind="percentage"),
                                           help="Promedio del Margen Bruto ((Precio Efectivo - Costo) / Precio Efectivo) para todos los items del escenario. Delta vs. estado actual.")

                kpi_cols_finance[1].metric("Avg. Markup s/Costo",
                                           f"{avg_scenario_new_markup_cost:.2f}%" if pd.notna(avg_scenario_new_markup_cost) else "N/A",
                                           delta=get_delta_string_for_metric(avg_scenario_new_markup_cost, avg_scenario_current_markup_cost, kind="percentage"),
                                           help="Promedio del Markup sobre Costo ((Precio Efectivo - Costo) / Costo) para todos los items del escenario. Delta vs. estado actual.")

                kpi_cols_finance[2].metric("Avg. Markup s/MAP",
                                           f"{avg_scenario_new_markup_map:.2f}%" if pd.notna(avg_scenario_new_markup_map) else "N/A",
                                           delta=get_delta_string_for_metric(avg_scenario_new_markup_map, avg_scenario_current_markup_map, kind="percentage"),
                                           help="Promedio del Markup sobre MAP ((Precio Efectivo - MAP) / MAP) para todos los items del escenario. Delta vs. estado actual.")

                st.markdown("---")
                st.markdown("##### Cumplimiento de Reglas (Cantidad de Productos)")

                # --- Rule Compliance KPIs ---
                kpi_cols_rules = st.columns(4)

                rules_display_names = {
                    'Rule 1': 'Regla 1: vs Comp.',
                    'Rule 2': 'Regla 2: vs MAP',
                    'Rule 3': 'Regla 3: Margen',
                    'Overall Compliant': 'Cumplimiento General'
                }
                
                rule_keys_for_display = ['Rule 1', 'Rule 2', 'Rule 3', 'Overall Compliant']

                for i, rule_key in enumerate(rule_keys_for_display):
                    current_met = scenario_counts.get(f'Current {rule_key} Met', 0)
                    new_met = scenario_counts.get(f'New {rule_key} Met', 0)
                    kpi_cols_rules[i].metric(f"{rules_display_names[rule_key]} Met",
                                              f"{new_met:,}",
                                              delta=get_delta_string_for_metric(new_met, current_met, kind="count"))

        # Moved the message for empty scenario here, so it appears if the expander is not shown
        if not st.session_state.staged_scenario_items: 
            st.info("No hay productos en el escenario. Añade productos desde las vistas previas B2C o B2B.")
        # Display each staged lot (moved below the aggregated KPIs expander)
        if st.session_state.staged_scenario_items:           
            # Button to clear the entire scenario
            if st.button("🗑️ Limpiar Todo el Escenario", key="clear_scenario"):
                st.session_state.staged_scenario_items = []
                st.session_state.staged_scenario_kpis = {}
                st.rerun() # Rerun to reflect the cleared state immediately
            
            # Display each staged lot
            scenario_df_list = [] # To collect all DFs for aggregated KPIs
            for i, lot_df in enumerate(st.session_state.staged_scenario_items):
                lot_type = lot_df['Scenario Type'].iloc[0] if not lot_df.empty and 'Scenario Type' in lot_df.columns else 'N/A'
                applied_discount = lot_df['Applied Discount (%)'].iloc[0] if not lot_df.empty and 'Applied Discount (%)' in lot_df.columns else 'N/A'
                
                # Define columns to display based on lot_type
                if lot_type == 'B2C':
                    original_price_col_name = RC_PRICE_COLUMN
                    new_price_col_name = 'New RC Price'
                    current_margin_col_name = 'Current RC Margin (%)'
                    new_margin_col_name = 'New RC Margin (%)'
                    rule_cols_original = b2c_original_rule_cols
                    rule_cols_display_names = b2c_display_rule_names
                    base_cols_scenario = base_cols_b2c # Includes SKU, Item Name, Year/Model
                elif lot_type == 'B2B':
                    original_price_col_name = RC_B2B_PRICE_COLUMN
                    new_price_col_name = 'New RC B2B Price'
                    current_margin_col_name = 'Current RC B2B Margin (%)'
                    new_margin_col_name = 'New RC B2B Margin (%)'
                    rule_cols_original = b2b_original_rule_cols
                    rule_cols_display_names = b2b_display_rule_names
                    base_cols_scenario = base_cols_b2b # Includes SKU, Item Name, Year/Model
                else:
                    # Default to B2C if type is N/A or unexpected
                    original_price_col_name = RC_PRICE_COLUMN
                    new_price_col_name = 'New RC Price'
                    current_margin_col_name = 'Current RC Margin (%)'
                    new_margin_col_name = 'New RC Margin (%)'
                    rule_cols_original = b2c_original_rule_cols
                    rule_cols_display_names = b2c_display_rule_names
                    base_cols_scenario = base_cols_b2c

                columns_for_scenario_lot_display = []
                if all(col in lot_df.columns for col in base_cols_scenario):
                     columns_for_scenario_lot_display.extend(base_cols_scenario)
                
                # Add price and margin columns if they exist
                price_margin_cols = [
                    original_price_col_name, current_margin_col_name, 
                    new_price_col_name, new_margin_col_name
                ]
                for col in price_margin_cols:
                    if col in lot_df.columns:
                        columns_for_scenario_lot_display.append(col)

                # Add rule columns
                for rule_col in rule_cols_original:
                    if rule_col in lot_df.columns:
                        columns_for_scenario_lot_display.append(rule_col)
                        
                # Add reference price columns (common for B2C/B2B context in scenario)
                reference_cols = ['Min B2C Competitor Price', 'Min B2B Competitor Price', 'MAP', 'Unit Cost']                    
                for col in reference_cols:
                    if col in lot_df.columns and col not in columns_for_scenario_lot_display:
                        columns_for_scenario_lot_display.append(col)
                        
                # Add discount and type
                if 'Applied Discount (%)' in lot_df.columns:
                    columns_for_scenario_lot_display.append('Applied Discount (%)')
                if 'Scenario Type' in lot_df.columns:
                     columns_for_scenario_lot_display.append('Scenario Type')
                
                # Ensure no duplicates and preserve order as much as possible
                seen_scenario_lot_cols = set()
                unique_cols_for_scenario_lot_display = [x for x in columns_for_scenario_lot_display if not (x in seen_scenario_lot_cols or seen_scenario_lot_cols.add(x))]
                
                df_display_lot = lot_df[unique_cols_for_scenario_lot_display].copy() # Work with a copy
                
                with st.expander(f"Lote {i+1}: {lot_type} ({len(df_display_lot)} productos) con {applied_discount}% de descuento", expanded=False):
                    styler_lot = format_and_style_rules_df(df_display_lot, [col for col in rule_cols_original if col in df_display_lot.columns])
                    
                    column_config_lot = {original: st.column_config.TextColumn(new_display_name) 
                                       for original, new_display_name in zip(rule_cols_original, rule_cols_display_names) 
                                       if original in df_display_lot.columns}
                    
                    # Add specific formatting for other columns, checking existence
                    if new_price_col_name in df_display_lot.columns:
                        column_config_lot[new_price_col_name] = st.column_config.NumberColumn("New Price (after discount)", format="%.2f")
                    if original_price_col_name in df_display_lot.columns:
                        column_config_lot[original_price_col_name] = st.column_config.NumberColumn("Original Price", format="%.2f")
                    if current_margin_col_name in df_display_lot.columns:
                        column_config_lot[current_margin_col_name] = st.column_config.NumberColumn("Current Margin", format="%.2f%%")
                    if new_margin_col_name in df_display_lot.columns:
                        column_config_lot[new_margin_col_name] = st.column_config.NumberColumn("New Margin", format="%.2f%%")
                    if 'Min B2C Competitor Price' in df_display_lot.columns:
                        column_config_lot['Min B2C Competitor Price'] = st.column_config.NumberColumn(format="%.2f")
                    if 'Min B2B Competitor Price' in df_display_lot.columns:
                         column_config_lot['Min B2B Competitor Price'] = st.column_config.NumberColumn(format="%.2f")
                    if 'MAP' in df_display_lot.columns:
                        column_config_lot['MAP'] = st.column_config.NumberColumn(format="%.2f")
                    if 'Unit Cost' in df_display_lot.columns:
                        column_config_lot['Unit Cost'] = st.column_config.NumberColumn(format="%.2f")
                    if 'Applied Discount (%)' in df_display_lot.columns:
                        column_config_lot['Applied Discount (%)'] = st.column_config.NumberColumn(format="%.1f%%")

                    st.dataframe(styler_lot, use_container_width=True, hide_index=True, column_config=column_config_lot)
                    
                    if st.button(f"Eliminar Lote {i+1}", key=f"delete_lot_{i}"):
                        st.session_state.staged_scenario_items.pop(i)
                        st.rerun() # Rerun to update the display
                scenario_df_list.append(lot_df)

                # --- Calculate and Display Aggregated KPIs for the Scenario ---     

    # --- DATA COMPLETENESS TAB --- 
    with tab_completeness:
        st.header("🔍 Data Completeness Analysis")
        st.caption("Analysis of missing data in the original dataset loaded from Google Sheets.")

        if df_original.empty:
            st.warning("No data loaded from Google Sheets to analyze.")
        else:
            # Overall Summary
            st.subheader("Overall Dataset Summary")
            total_rows = len(df_original)
            total_cols = len(df_original.columns)
            total_cells = total_rows * total_cols
            missing_cells = df_original.isnull().sum().sum()
            overall_completeness_percentage = (1 - (missing_cells / total_cells)) * 100 if total_cells > 0 else 0
            
            summary_cols = st.columns(3)
            summary_cols[0].metric("Total Rows", f"{total_rows:,}")
            summary_cols[1].metric("Total Columns", f"{total_cols:,}")
            summary_cols[2].metric("Overall Data Completeness", f"{overall_completeness_percentage:.2f}%")

            st.divider()

            # Column-by-Column Analysis
            st.subheader("Column-by-Column Completeness")
            
            column_data = []
            for col in df_original.columns:
                missing_count = df_original[col].isnull().sum()
                total_count = len(df_original[col])
                completeness_percentage = (1 - (missing_count / total_count)) * 100 if total_count > 0 else 0
                column_data.append({
                    "Column Name": col,
                    "Data Type": str(df_original[col].dtype),
                    "Missing Values": missing_count,
                    "% Missing": f"{(missing_count / total_count) * 100:.2f}%" if total_count > 0 else "N/A",
                    "Completeness": completeness_percentage / 100 # for st.progress
                })
            
            df_column_analysis = pd.DataFrame(column_data)

            st.dataframe(
                df_column_analysis,
                column_config={
                    "Column Name": st.column_config.TextColumn(width="medium"),
                    "Data Type": st.column_config.TextColumn(width="small"),
                    "Missing Values": st.column_config.NumberColumn(format="%d", width="small"),
                    "% Missing": st.column_config.TextColumn(width="small"),
                    "Completeness": st.column_config.ProgressColumn(
                        format="%.2f%%",
                        min_value=0,
                        max_value=1,
                        width="medium",
                    ),
                },
                use_container_width=True,
                hide_index=True
            )
            
            st.divider()

            # Highlight Critical Columns
            st.subheader("Completeness of Critical Columns")
            critical_columns_for_general_display = [
                'ITEM NAME', 'SKU', 'EAN', 'BRAND', 'SPORT', 'TYPE1', 'Year/Model', 
                # 'Unit Cost', 'MAP', RC_PRICE_COLUMN, # These will get special treatment
            ] + ALL_COMPETITOR_COLUMNS # Competitors are fine with general display
            if RC_B2B_PRICE_COLUMN in df_original.columns and RC_B2B_PRICE_COLUMN not in critical_columns_for_general_display:
                # critical_columns_for_general_display.append(RC_B2B_PRICE_COLUMN) # Also special treatment
                pass # Will be handled by the special list

            # Columns for in-depth analysis
            in_depth_analysis_cols = ['Unit Cost', 'MAP', RC_PRICE_COLUMN]
            if RC_B2B_PRICE_COLUMN in df_original.columns:
                in_depth_analysis_cols.append(RC_B2B_PRICE_COLUMN)
            
            # Ensure only existing columns are processed for general display
            critical_columns_present_general = [col for col in critical_columns_for_general_display if col in df_original.columns]
            
            if not critical_columns_present_general and not any(col in df_original.columns for col in in_depth_analysis_cols):
                st.info("None of the pre-defined critical columns were found in the dataset.")
            
            if critical_columns_present_general:
                st.markdown("##### General Critical Columns")
                critical_column_data_general = []
                for col in critical_columns_present_general:
                    missing_count = df_original[col].isnull().sum()
                    total_count = len(df_original[col])
                    completeness_percentage = (1 - (missing_count / total_count)) * 100 if total_count > 0 else 0
                    critical_column_data_general.append({
                        "Critical Column": col,
                        "Missing Values": missing_count,
                        "% Complete": completeness_percentage
                    })
                df_critical_analysis_general = pd.DataFrame(critical_column_data_general)
                
                num_critical_cols_to_display_general = len(df_critical_analysis_general)
                layout_cols_per_row_general = 3
                num_rows_for_critical_general = (num_critical_cols_to_display_general + layout_cols_per_row_general - 1) // layout_cols_per_row_general

                for i in range(num_rows_for_critical_general):
                    row_layout_cols = st.columns(layout_cols_per_row_general)
                    for j in range(layout_cols_per_row_general):
                        idx = i * layout_cols_per_row_general + j
                        if idx < num_critical_cols_to_display_general:
                            col_name = df_critical_analysis_general.iloc[idx]["Critical Column"]
                            completeness = df_critical_analysis_general.iloc[idx]["% Complete"]
                            missing_vals = df_critical_analysis_general.iloc[idx]["Missing Values"]
                            with row_layout_cols[j]:
                                st.markdown(f"**{col_name}**")
                                st.progress(completeness / 100)
                                st.caption(f"{completeness:.2f}% complete ({missing_vals:,} missing)")
                st.caption("The progress bar shows the percentage of non-missing values for these critical columns.")
                st.divider()

            # In-depth analysis for specified columns
            st.markdown("##### In-Depth Analysis of Key Financial Columns")
            for col_name in in_depth_analysis_cols:
                if col_name in df_original.columns:
                    st.markdown(f"#### {col_name}")
                    
                    # Basic Completeness
                    missing_count = df_original[col_name].isnull().sum()
                    total_count = len(df_original[col_name])
                    completeness_percentage = (1 - (missing_count / total_count)) * 100 if total_count > 0 else 0
                    st.progress(completeness_percentage / 100)
                    st.caption(f"{completeness_percentage:.2f}% complete ({missing_count:,} of {total_count:,} rows missing)")

                    with st.expander(f"Show Detailed Analysis for {col_name}"):
                        # Descriptive Statistics
                        st.markdown("**Descriptive Statistics (for non-missing values):**")
                        # Ensure column is numeric for describe()
                        numeric_col_series = pd.to_numeric(df_original[col_name], errors='coerce')
                        if not numeric_col_series.isnull().all():
                            desc_stats = numeric_col_series.describe().loc[['min', 'max', 'mean', '50%', 'std']]
                            desc_stats.index = ['Minimum', 'Maximum', 'Mean', 'Median', 'Std. Deviation']
                            st.table(desc_stats.to_frame(name="Value").style.format("{:.2f}"))
                            
                            # Data Quality Checks (Zeros and Negatives)
                            if col_name in ['Unit Cost', RC_PRICE_COLUMN, RC_B2B_PRICE_COLUMN]:
                                st.markdown("**Data Quality Checks:**")
                                num_zeros = (numeric_col_series == 0).sum()
                                num_negatives = (numeric_col_series < 0).sum()
                                st.metric(label=f"Number of Zero Values in {col_name}", value=f"{num_zeros:,}")
                                st.metric(label=f"Number of Negative Values in {col_name}", value=f"{num_negatives:,}")
                        else:
                            st.info(f"No valid numeric data in '{col_name}' to calculate descriptive statistics.")

                        # Relational Completeness
                        st.markdown("**Relational Completeness:**")
                        if col_name == RC_PRICE_COLUMN:
                            if 'Unit Cost' in df_original.columns:
                                missing_related = df_original[df_original[RC_PRICE_COLUMN].notna() & df_original['Unit Cost'].isnull()].shape[0]
                                st.metric(f"Products with '{RC_PRICE_COLUMN}' but MISSING 'Unit Cost'", f"{missing_related:,}")
                            if 'MAP' in df_original.columns:
                                missing_related = df_original[df_original[RC_PRICE_COLUMN].notna() & df_original['MAP'].isnull()].shape[0]
                                st.metric(f"Products with '{RC_PRICE_COLUMN}' but MISSING 'MAP'", f"{missing_related:,}")
                            
                            # New Check: RC_PRICE_COLUMN <= RC_B2B_PRICE_COLUMN
                            if RC_B2B_PRICE_COLUMN in df_original.columns:
                                st.markdown("**Price Logic Checks:**")
                                rc_b2c_numeric = pd.to_numeric(df_original[RC_PRICE_COLUMN], errors='coerce')
                                rc_b2b_numeric = pd.to_numeric(df_original[RC_B2B_PRICE_COLUMN], errors='coerce')
                                
                                valid_comparison_mask = rc_b2c_numeric.notna() & rc_b2b_numeric.notna()
                                anomaly_count = (rc_b2c_numeric[valid_comparison_mask] <= rc_b2b_numeric[valid_comparison_mask]).sum()
                                
                                help_text_anomaly = f"Counts products where '{RC_PRICE_COLUMN}' (B2C price) is less than or equal to '{RC_B2B_PRICE_COLUMN}' (B2B price), among products where both prices are present and numeric."
                                st.metric(label=f"Instances: '{RC_PRICE_COLUMN}' <= '{RC_B2B_PRICE_COLUMN}'", value=f"{anomaly_count:,}", help=help_text_anomaly)

                        elif col_name == RC_B2B_PRICE_COLUMN and RC_B2B_PRICE_COLUMN in df_original.columns:
                            if 'Unit Cost' in df_original.columns:
                                missing_related = df_original[df_original[RC_B2B_PRICE_COLUMN].notna() & df_original['Unit Cost'].isnull()].shape[0]
                                st.metric(f"Products with '{RC_B2B_PRICE_COLUMN}' but MISSING 'Unit Cost'", f"{missing_related:,}")
                            if 'MAP' in df_original.columns:
                                missing_related = df_original[df_original[RC_B2B_PRICE_COLUMN].notna() & df_original['MAP'].isnull()].shape[0]
                                st.metric(f"Products with '{RC_B2B_PRICE_COLUMN}' but MISSING 'MAP'", f"{missing_related:,}")

                        elif col_name == 'Unit Cost':
                            if RC_PRICE_COLUMN in df_original.columns:
                                missing_related = df_original[df_original['Unit Cost'].notna() & df_original[RC_PRICE_COLUMN].isnull()].shape[0]
                                st.metric(f"Products with 'Unit Cost' but MISSING '{RC_PRICE_COLUMN}'", f"{missing_related:,}")
                            if RC_B2B_PRICE_COLUMN in df_original.columns:
                                missing_related = df_original[df_original['Unit Cost'].notna() & df_original[RC_B2B_PRICE_COLUMN].isnull()].shape[0]
                                st.metric(f"Products with 'Unit Cost' but MISSING '{RC_B2B_PRICE_COLUMN}'", f"{missing_related:,}")
                        
                        elif col_name == 'MAP':
                            if RC_PRICE_COLUMN in df_original.columns:
                                missing_related = df_original[df_original['MAP'].notna() & df_original[RC_PRICE_COLUMN].isnull()].shape[0]
                                st.metric(f"Products with 'MAP' but MISSING '{RC_PRICE_COLUMN}'", f"{missing_related:,}")
                            if RC_B2B_PRICE_COLUMN in df_original.columns:
                                missing_related = df_original[df_original['MAP'].notna() & df_original[RC_B2B_PRICE_COLUMN].isnull()].shape[0]
                                st.metric(f"Products with 'MAP' but MISSING '{RC_B2B_PRICE_COLUMN}'", f"{missing_related:,}")
                        else:
                            st.caption("No specific relational checks defined for this column beyond basic completeness.")
                    st.divider() # Divider after each in-depth column analysis
                # else: # Column not in df_original, skip silently
                #     pass 

# --- Add source information ---
# This should be outside the 'else' of df_original.empty, always visible.
st.sidebar.caption("ℹ️ **Data Source Note:** Competitor pricing data is obtained weekly (Thursdays, 8 AM US Central Time) by scraping competitor websites and matching products using their EAN codes.")