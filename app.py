import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page Configuration
st.set_page_config(page_title="Pizza Sales Dashboard", page_icon="🍕", layout="wide")

# Title
st.title("🍕 Professional Pizza Sales Dashboard")
st.markdown("Welcome to the comprehensive analysis of our pizza sales data.")

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('pizza_sales.csv')
        
        # Date and Time conversion
        # Handle mixed date formats (e.g. 1/1/2015 and 13-01-2015)
        df['order_date'] = pd.to_datetime(df['order_date'], format='mixed', dayfirst=True)
        
        # Helper for hour extraction
        temp_time = pd.to_datetime(df['order_time'], format='%H:%M:%S')
        df['hour'] = temp_time.dt.hour
        df['order_time'] = temp_time.dt.time
        
        # Extract useful time features
        df['month'] = df['order_date'].dt.month_name()
        df['day_of_week'] = df['order_date'].dt.day_name()
        
        # Ensure numeric columns are correct
        df['total_price'] = pd.to_numeric(df['total_price'], errors='coerce')
        df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is not None:
    # Sidebar Filters
    st.sidebar.header("Filter Data")
    
    # Date Range
    min_date = df['order_date'].min()
    max_date = df['order_date'].max()
    
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Category Filter
    categories = st.sidebar.multiselect(
        "Select Pizza Category",
        options=df['pizza_category'].unique(),
        default=df['pizza_category'].unique()
    )
    
    # Apply Filters
    mask = (df['order_date'] >= pd.to_datetime(start_date)) & \
           (df['order_date'] <= pd.to_datetime(end_date)) & \
           (df['pizza_category'].isin(categories))
    
    filtered_df = df.loc[mask]

    # Main Dashboard Layout
    
    # KPI Section with custom styling
    st.subheader("Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = filtered_df['total_price'].sum()
    total_orders = filtered_df['order_id'].nunique()
    total_pizzas = filtered_df['quantity'].sum()
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Pizzas Sold", f"{total_pizzas:,.0f}")
    col4.metric("Avg Order Value", f"${avg_order_value:,.2f}")
    
    st.markdown("---")

    # Charts Section 1: Trends & Categories
    col_chart1, col_chart2 = st.columns([2, 1]) # Give more space to the trend chart
    
    with col_chart1:
        st.subheader("Sales Trend Over Time")
        daily_sales = filtered_df.groupby('order_date')['total_price'].sum().reset_index()
        
        # Add Moving Average for smoother trend
        daily_sales['7_Day_MA'] = daily_sales['total_price'].rolling(window=7).mean()
        
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=daily_sales['order_date'], 
            y=daily_sales['total_price'],
            mode='lines',
            name='Daily Revenue',
            line=dict(color='#1f77b4', width=1),
            opacity=0.5
        ))
        fig_daily.add_trace(go.Scatter(
            x=daily_sales['order_date'], 
            y=daily_sales['7_Day_MA'],
            mode='lines',
            name='7-Day Average',
            line=dict(color='#ff7f0e', width=3)
        ))
        
        fig_daily.update_layout(
            title="Daily Revenue & 7-Day Moving Average",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            hovermode="x unified",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_daily, use_container_width=True)
        
    with col_chart2:
        st.subheader("Sales Distribution by Category")
        category_sales = filtered_df.groupby('pizza_category')['total_price'].sum().reset_index()
        
        fig_cat = px.pie(
            category_sales, 
            values='total_price', 
            names='pizza_category', 
            title='Revenue Share',
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_cat.update_traces(textposition='inside', textinfo='percent+label')
        fig_cat.update_layout(showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    # Charts Section 2: Detailed Temporal Analysis
    st.subheader("Temporal Analysis")
    col_temp1, col_temp2 = st.columns(2)

    with col_temp1:
        st.subheader("Busiest Days of the Week")
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_orders = filtered_df.groupby('day_of_week')['order_id'].nunique().reset_index()
        
        fig_days = px.bar(
            daily_orders, 
            x='day_of_week', 
            y='order_id', 
            title='Total Orders by Day',
            category_orders={"day_of_week": days_order},
            color='order_id',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_days, use_container_width=True)

    with col_temp2:
        st.subheader("Busiest Hours of the Day")
        hourly_orders = filtered_df.groupby('hour')['order_id'].nunique().reset_index()
        
        fig_hours = px.bar(
            hourly_orders, 
            x='hour', 
            y='order_id', 
            title='Total Orders by Hour',
            color='order_id',
            color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_hours, use_container_width=True)

    # Charts Section 3: Peak Operations Analysis (Heatmap)
    st.subheader("Peak Operations Heatmap")
    
    # Heatmap: Day of Week vs Hour
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = filtered_df.groupby(['day_of_week', 'hour'])['order_id'].nunique().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='order_id').fillna(0)
    
    # Reorder index
    heatmap_pivot = heatmap_pivot.reindex(days_order)
    
    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Orders"),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        color_continuous_scale='RdBu_r',
        aspect="auto",
        title="Heatmap: Order Intensity (Day vs Hour)"
    )
    fig_heatmap.update_xaxes(dtick=1)
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Charts Section 4: Product Performance Details
    st.subheader("Product Performance")
    col_prod1, col_prod2 = st.columns(2)
    
    with col_prod1:
        st.subheader("Top 5 Best Sellers")
        top_pizzas = filtered_df.groupby('pizza_name')['total_price'].sum().sort_values(ascending=False).head(5).reset_index()
        
        fig_top = px.bar(
            top_pizzas, 
            y='pizza_name', 
            x='total_price', 
            orientation='h', 
            title='Top 5 Pizzas by Revenue', 
            text='total_price',
            color='total_price',
            color_continuous_scale='Viridis'
        )
        
        fig_top.update_traces(texttemplate='$%{text:,.0f}', textposition='inside')
        fig_top.update_layout(
            yaxis={'categoryorder':'total ascending'}, 
            xaxis_title="Revenue ($)",
            yaxis_title=None,
            showlegend=False
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col_prod2:
        st.subheader("Bottom 5 Performers")
        bottom_pizzas = filtered_df.groupby('pizza_name')['total_price'].sum().sort_values(ascending=True).head(5).reset_index()
        
        fig_bottom = px.bar(
            bottom_pizzas, 
            y='pizza_name', 
            x='total_price', 
            orientation='h', 
            title='Bottom 5 Pizzas by Revenue', 
            text='total_price',
            color='total_price',
            color_continuous_scale='RdBu'
        )
        
        fig_bottom.update_traces(texttemplate='$%{text:,.0f}', textposition='inside')
        fig_bottom.update_layout(
            yaxis={'categoryorder':'total descending'}, 
            xaxis_title="Revenue ($)",
            yaxis_title=None,
            showlegend=False
        )
        st.plotly_chart(fig_bottom, use_container_width=True)

    # Charts Section 5: Size Analysis
    col_size1, col_size2 = st.columns(2)
    
    with col_size1:
        st.subheader("Sales by Pizza Size")
        size_sales = filtered_df.groupby('pizza_size')['total_price'].sum().reset_index()
        size_order = ['S', 'M', 'L', 'XL', 'XXL']
        
        fig_size = px.bar(
            size_sales, 
            x='pizza_size', 
            y='total_price', 
            title='Total Revenue by Size', 
            category_orders={"pizza_size": size_order}, 
            text_auto='.2s',
            color='total_price',
            color_continuous_scale='Blues'
        )
        fig_size.update_layout(showlegend=False)
        st.plotly_chart(fig_size, use_container_width=True)
        
    with col_size2:
        st.subheader("Size Preference by Category")
        # Stacked bar chart
        size_cat_sales = filtered_df.groupby(['pizza_category', 'pizza_size'])['quantity'].sum().reset_index()
        
        fig_stack = px.bar(
            size_cat_sales, 
            x='pizza_category', 
            y='quantity', 
            color='pizza_size', 
            title='Quantity Sold by Category & Size',
            category_orders={"pizza_size": size_order},
            barmode='stack'
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    # Charts Section 6: Detailed Demand Analysis
    st.subheader("Deep Dive: Demand by Pizza Type")
    st.markdown("Analyze which pizza types are most popular across different timeframes.")
    
    # Professional Forecasting & Demand Insights Section
    st.markdown("---")
    st.subheader("Demand Forecasting & Strategic Insights")
    
    # Helper function to clean pizza names
    def clean_name(name):
        return name.replace("The ", "").replace(" Pizza", "")
    
    # Calculate comprehensive demand statistics
    # FIX: Ensure we include days with 0 sales in the calculation for accurate daily average and volatility
    
    # 1. Group by date and pizza to get daily sales where they occurred
    daily_sales_raw = filtered_df.groupby(['order_date', 'pizza_name'])['quantity'].sum().reset_index()
    
    # 2. Create a full date range for the selected period
    full_date_range = pd.date_range(start=daily_sales_raw['order_date'].min(), end=daily_sales_raw['order_date'].max())
    
    # 3. Pivot to create a matrix of Dates x Pizzas, filling missing days with 0
    daily_sales_full = daily_sales_raw.pivot(index='order_date', columns='pizza_name', values='quantity').reindex(full_date_range).fillna(0)
    
    # 4. Calculate stats across the full timeline (now including 0s)
    volatility_stats = daily_sales_full.agg(['mean', 'std', 'min', 'max']).T.reset_index()
    
    volatility_stats['cv'] = volatility_stats['std'] / volatility_stats['mean']  # Coefficient of Variation
    # Handle division by zero or very low mean if necessary (though unlikely with this dataset)
    volatility_stats['cv'] = volatility_stats['cv'].fillna(0)
    
    volatility_stats['volatility_risk'] = volatility_stats['cv'].apply(lambda x: 'High' if x > 0.8 else 'Medium' if x > 0.4 else 'Low') # Adjusted thresholds for full-calendar data (CV is usually higher when including 0s)
    
    # Weekly pattern analysis
    daily_pizza_qty = filtered_df.groupby(['day_of_week', 'pizza_name'])['quantity'].sum().reset_index()
    weekend_days = ['Saturday', 'Sunday']
    weekday_sales = daily_pizza_qty[~daily_pizza_qty['day_of_week'].isin(weekend_days)].groupby('pizza_name')['quantity'].mean()
    weekend_sales = daily_pizza_qty[daily_pizza_qty['day_of_week'].isin(weekend_days)].groupby('pizza_name')['quantity'].mean()
    weekend_lift = ((weekend_sales - weekday_sales) / weekday_sales * 100).fillna(0)
    
    # Hourly pattern analysis
    hourly_pizza_qty = filtered_df.groupby(['hour', 'pizza_name'])['quantity'].sum().reset_index()
    
    # Monthly trend analysis (growing/declining)
    monthly_pizza_trend = filtered_df.groupby(['pizza_name', 'month'])['quantity'].sum().reset_index()
    monthly_pizza_trend['month_num'] = pd.to_datetime(monthly_pizza_trend['month'], format='%B', errors='coerce').dt.month
    monthly_pizza_trend = monthly_pizza_trend.dropna(subset=['month_num'])
    
    def calculate_trend(group):
        if len(group) > 1:
            return np.polyfit(group['month_num'], group['quantity'], 1)[0]
        return 0
    
    trend_slope = monthly_pizza_trend.groupby('pizza_name').apply(calculate_trend).reset_index()
    trend_slope.columns = ['pizza_name', 'slope']
    
    # Key Metrics Dashboard
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    
    peak_combination = daily_pizza_qty.loc[daily_pizza_qty['quantity'].idxmax()]
    best_pizza = volatility_stats.loc[volatility_stats['mean'].idxmax()]
    most_stable = volatility_stats.loc[volatility_stats['cv'].idxmin()]
    most_volatile = volatility_stats.loc[volatility_stats['cv'].idxmax()]
    
    with col_kpi1:
        st.metric(
            "Peak Single-Day Performance",
            f"{clean_name(peak_combination['pizza_name'])}",
            f"{int(peak_combination['quantity'])} units on {peak_combination['day_of_week']}"
        )
    
    with col_kpi2:
        st.metric(
            "Highest Average Daily Sales",
            f"{clean_name(best_pizza['pizza_name'])}",
            f"{int(best_pizza['mean'])} units/day"
        )
    
    with col_kpi3:
        st.metric(
            "Lowest Demand Volatility",
            f"{clean_name(most_stable['pizza_name'])}",
            f"CV: {most_stable['cv']:.2f}"
        )
    
    with col_kpi4:
        st.metric(
            "Highest Demand Volatility",
            f"{clean_name(most_volatile['pizza_name'])}",
            f"CV: {most_volatile['cv']:.2f}"
        )
    
    # Detailed Analysis Sections
    tab_forecast1, tab_forecast2, tab_forecast3 = st.tabs([
        "Demand Stability Analysis", 
        "Seasonal & Temporal Patterns", 
        "Strategic Recommendations"
    ])
    
    with tab_forecast1:
        st.markdown("### Demand Volatility Assessment")
        st.markdown("Understanding demand consistency is critical for inventory optimization and forecasting accuracy.")
        
        # Create volatility classification table
        volatility_display = volatility_stats[['pizza_name', 'mean', 'std', 'cv', 'volatility_risk', 'min', 'max']].copy()
        volatility_display.columns = ['Pizza', 'Avg Daily Sales', 'Std Deviation', 'Coefficient of Variation', 'Risk Level', 'Min Daily', 'Max Daily']
        volatility_display = volatility_display.sort_values('Coefficient of Variation')
        volatility_display['Pizza'] = volatility_display['Pizza'].apply(clean_name)
        volatility_display['Avg Daily Sales'] = volatility_display['Avg Daily Sales'].round(1)
        volatility_display['Std Deviation'] = volatility_display['Std Deviation'].round(1)
        volatility_display['Coefficient of Variation'] = volatility_display['Coefficient of Variation'].round(3)
        
        st.dataframe(volatility_display, use_container_width=True)
        
        st.markdown("""
        **Interpretation Guide:**
        - **Coefficient of Variation (CV)**: Lower values indicate more predictable demand patterns
        - **CV < 0.3**: Low volatility - highly predictable, minimal safety stock needed
        - **CV 0.3 - 0.5**: Medium volatility - standard safety stock recommended
        - **CV > 0.5**: High volatility - significant safety stock buffer required
        """)
    
    with tab_forecast2:
        st.markdown("### Temporal Demand Patterns")
        
        col_pattern1, col_pattern2 = st.columns(2)
        
        with col_pattern1:
            st.markdown("#### Weekend vs Weekday Performance")
            weekend_analysis = pd.DataFrame({
                'Pizza': weekday_sales.index,
                'Weekday Avg': weekday_sales.values,
                'Weekend Avg': [weekend_sales.get(idx, 0) for idx in weekday_sales.index],
                'Weekend Lift %': [weekend_lift.get(idx, 0) for idx in weekday_sales.index]
            })
            weekend_analysis = weekend_analysis.sort_values('Weekend Lift %', ascending=False).head(10)
            weekend_analysis['Pizza'] = weekend_analysis['Pizza'].apply(clean_name)
            
            st.dataframe(weekend_analysis, use_container_width=True)
        
        with col_pattern2:
            st.markdown("#### Peak Demand Hours")
            peak_hours_by_pizza = hourly_pizza_qty.groupby('pizza_name').apply(
                lambda x: x.loc[x['quantity'].idxmax()]
            ).reset_index(drop=True)
            peak_hours_by_pizza = peak_hours_by_pizza[['pizza_name', 'hour', 'quantity']].sort_values('quantity', ascending=False).head(10)
            peak_hours_by_pizza['pizza_name'] = peak_hours_by_pizza['pizza_name'].apply(clean_name)
            peak_hours_by_pizza.columns = ['Pizza', 'Peak Hour', 'Quantity at Peak']
            
            st.dataframe(peak_hours_by_pizza, use_container_width=True)
        
        st.markdown("#### Monthly Sales Trends")
        trending_up = trend_slope[trend_slope['slope'] > 0].sort_values('slope', ascending=False).head(5)
        trending_down = trend_slope[trend_slope['slope'] < 0].sort_values('slope').head(5)
        
        col_trend1, col_trend2 = st.columns(2)
        
        with col_trend1:
            st.markdown("**Growing Demand (Positive Trend)**")
            if len(trending_up) > 0:
                trending_up_display = trending_up.copy()
                trending_up_display['pizza_name'] = trending_up_display['pizza_name'].apply(clean_name)
                trending_up_display.columns = ['Pizza', 'Monthly Growth Rate']
                trending_up_display['Monthly Growth Rate'] = trending_up_display['Monthly Growth Rate'].round(2)
                st.dataframe(trending_up_display, use_container_width=True)
            else:
                st.info("No clear positive trends identified in the selected period.")
        
        with col_trend2:
            st.markdown("**Declining Demand (Negative Trend)**")
            if len(trending_down) > 0:
                trending_down_display = trending_down.copy()
                trending_down_display['pizza_name'] = trending_down_display['pizza_name'].apply(clean_name)
                trending_down_display.columns = ['Pizza', 'Monthly Decline Rate']
                trending_down_display['Monthly Decline Rate'] = trending_down_display['Monthly Decline Rate'].round(2)
                st.dataframe(trending_down_display, use_container_width=True)
            else:
                st.info("No clear negative trends identified in the selected period.")
                
        # New: Total Sold per Pizza by Month Table
        st.markdown("#### Total Quantity Sold by Pizza and Month")
        # Pivot the monthly data to create a readable matrix
        monthly_sales_matrix = monthly_pizza_trend.pivot(index='pizza_name', columns='month', values='quantity').fillna(0)
        # Sort columns by month order
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        # Filter to only include months present in the data
        available_months = [m for m in months_order if m in monthly_sales_matrix.columns]
        monthly_sales_matrix = monthly_sales_matrix[available_months]
        
        # Clean index names
        monthly_sales_matrix.index = monthly_sales_matrix.index.map(clean_name)
        
        # Display with heat-map like styling (using background_gradient if possible, or just dataframe)
        # Note: background_gradient requires matplotlib. If it fails, we fall back to standard dataframe.
        try:
            st.dataframe(monthly_sales_matrix.style.format("{:.0f}").background_gradient(cmap="Greens", axis=1), use_container_width=True)
        except ImportError:
             # Fallback if matplotlib is missing in environment
             st.dataframe(monthly_sales_matrix.style.format("{:.0f}"), use_container_width=True)
        except Exception as e:
             # General fallback
             st.dataframe(monthly_sales_matrix, use_container_width=True)
    
    with tab_forecast3:
        st.markdown("### Strategic Inventory & Forecasting Recommendations")
        
        # Calculate recommended safety stock levels
        volatility_stats['recommended_buffer'] = volatility_stats.apply(
            lambda row: row['mean'] * 0.10 if row['cv'] < 0.3 
            else row['mean'] * 0.20 if row['cv'] < 0.5 
            else row['mean'] * 0.35, axis=1
        )
        
        volatility_stats['recommended_daily_stock'] = volatility_stats['mean'] + volatility_stats['recommended_buffer']
        
        recommendations = volatility_stats[['pizza_name', 'mean', 'cv', 'volatility_risk', 'recommended_daily_stock']].copy()
        recommendations.columns = ['Pizza', 'Average Daily Sales', 'Volatility (CV)', 'Risk Level', 'Recommended Daily Stock']
        recommendations = recommendations.sort_values('Risk Level', ascending=True)
        recommendations['Pizza'] = recommendations['Pizza'].apply(clean_name)
        recommendations['Average Daily Sales'] = recommendations['Average Daily Sales'].round(1)
        recommendations['Recommended Daily Stock'] = recommendations['Recommended Daily Stock'].round(0).astype(int)
        recommendations['Volatility (CV)'] = recommendations['Volatility (CV)'].round(3)
        
        st.dataframe(recommendations, use_container_width=True)
        
        st.markdown("---")
        
        # Strategic Insights - Written directly from data analysis
        st.markdown("#### Strategic Insights & Recommendations")
        
        # Get actual values from recommendations table for insights
        high_risk_items = recommendations[recommendations['Risk Level'] == 'High'].head(3)
        low_risk_items = recommendations[recommendations['Risk Level'] == 'Low'].head(3)
        
        # High-risk insights
        if len(high_risk_items) > 0:
            st.markdown("**HIGH PRIORITY: Demand Volatility Management**")
            for idx, row in high_risk_items.iterrows():
                avg_sales = row['Average Daily Sales']
                cv = row['Volatility (CV)']
                recommended_stock = row['Recommended Daily Stock']
                buffer = recommended_stock - avg_sales
                buffer_pct = (buffer / avg_sales * 100) if avg_sales > 0 else 0
                
                st.markdown(f"""
                - **{row['Pizza']}**: Exhibits significant demand uncertainty with a coefficient of variation of {cv:.3f}. 
                  Average daily sales of {avg_sales:.1f} units fluctuate substantially, requiring a safety stock buffer 
                  of approximately {buffer:.0f} units ({buffer_pct:.0f}% above average daily sales). 
                  Implement dynamic inventory management with day-of-week adjustments and monitor closely for demand spikes.
                """)
        
        # Weekend optimization insights
        if len(weekend_lift) > 0:
            top_weekend_pizzas = weekend_lift.nlargest(3)
            st.markdown("**WEEKEND DEMAND OPTIMIZATION**")
            for pizza_name in top_weekend_pizzas.index:
                lift = top_weekend_pizzas[pizza_name]
                weekday_avg = weekday_sales.get(pizza_name, 0)
                weekend_avg = weekend_sales.get(pizza_name, 0)
                if weekday_avg > 0:
                    prep_increase = max(lift / 2, 10)  # Minimum 10% increase
                    st.markdown(f"""
                    - **{clean_name(pizza_name)}**: Demonstrates {lift:.1f}% higher demand on weekends 
                      ({weekend_avg:.1f} units vs. {weekday_avg:.1f} units on weekdays). 
                      Recommendation: Increase weekend preparation volume by approximately {prep_increase:.0f}% 
                      to capture additional revenue opportunities and reduce stockout risk during peak periods.
                    """)
        
        # Peak hour insights
        if len(hourly_pizza_qty) > 0:
            peak_hour_data = hourly_pizza_qty.loc[hourly_pizza_qty.groupby('pizza_name')['quantity'].idxmax()]
            top_peak_hours = peak_hour_data.nlargest(3, 'quantity')
            st.markdown("**PEAK HOUR DEMAND PATTERNS**")
            for idx, row in top_peak_hours.iterrows():
                st.markdown(f"""
                - **{clean_name(row['pizza_name'])}**: Reaches peak demand at {int(row['hour']):02d}:00 with {int(row['quantity'])} units sold. 
                  Initiate preparation activities approximately one hour prior ({int(row['hour']-1):02d}:00) 
                  to ensure product availability during this critical sales window.
                """)
        
        # Stable demand insights
        if len(low_risk_items) > 0:
            st.markdown("**INVENTORY OPTIMIZATION OPPORTUNITIES**")
            for idx, row in low_risk_items.iterrows():
                cv = row['Volatility (CV)']
                avg_sales = row['Average Daily Sales']
                st.markdown(f"""
                - **{row['Pizza']}**: Shows highly predictable demand patterns with a coefficient of variation of {cv:.3f} 
                  and consistent daily sales averaging {avg_sales:.1f} units. This predictability enables 
                  reduction in safety stock levels, lowering carrying costs while maintaining service levels. 
                  Consider implementing just-in-time inventory practices for this product.
                """)
        
        # Growth trend insights
        if len(trending_up) > 0:
            top_growing = trending_up.head(3)
            st.markdown("**EMERGING GROWTH OPPORTUNITIES**")
            for idx, row in top_growing.iterrows():
                growth_rate = row['slope']
                st.markdown(f"""
                - **{clean_name(row['pizza_name'])}**: Exhibits a positive sales trend with a monthly growth rate 
                  of {growth_rate:.2f} units. Consider increasing baseline inventory levels and promotional support 
                  to capitalize on this upward trajectory. Monitor closely to confirm sustained demand growth.
                """)
        
        # Declining trend insights
        if len(trending_down) > 0:
            top_declining = trending_down.head(3)
            st.markdown("**ATTENTION REQUIRED: DECLINING TRENDS**")
            for idx, row in top_declining.iterrows():
                decline_rate = abs(row['slope'])
                st.markdown(f"""
                - **{clean_name(row['pizza_name'])}**: Shows declining sales trend with a monthly decrease rate 
                  of {decline_rate:.2f} units. Investigate root causes such as changing customer preferences, 
                  competitive pressures, or pricing sensitivity. Consider promotional strategies, recipe optimization, 
                  or gradual inventory reduction to minimize waste.
                """)
        
        st.markdown("---")
        
        st.markdown("""
        **ANALYTICAL METHODOLOGY:**
        - **Coefficient of Variation (CV)**: Calculated as standard deviation divided by mean daily sales. Lower values indicate more predictable demand.
        - **Safety Stock Recommendations**: Based on CV thresholds - Low risk (CV < 0.3): 10% buffer, Medium risk (CV 0.3-0.5): 20% buffer, High risk (CV > 0.5): 35% buffer.
        - **Weekend Lift Analysis**: Calculated as percentage difference between weekend and weekday average sales volumes.
        - **Trend Analysis**: Linear regression applied to monthly sales data to identify growth or decline patterns.
        - **Recommendations**: Assume standard supplier lead times and should be adjusted based on actual operational constraints and supplier reliability.
        """)
    
    tab1, tab2, tab3 = st.tabs(["Weekly Demand", "Hourly Demand", "Monthly Trends"])
    
    with tab1:
        st.subheader("Pizza Popularity by Day of Week")
        # Analyze ALL pizzas, not just top N
        weekly_demand = filtered_df.groupby(['day_of_week', 'pizza_name'])['quantity'].sum().reset_index()
        
        # Ensure correct day order
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Use a taller height for the heatmap to accommodate all pizzas readable
        fig_weekly_demand = px.imshow(
            weekly_demand.pivot(index='pizza_name', columns='day_of_week', values='quantity').reindex(columns=days_order),
            labels=dict(x="Day of Week", y="Pizza Type", color="Quantity"),
            title="Heatmap: All Pizzas Sold by Day",
            color_continuous_scale='Greens',
            aspect="auto",
            height=800 # Make chart taller to fit all labels
        )
        st.plotly_chart(fig_weekly_demand, use_container_width=True)

    with tab2:
        st.subheader("Pizza Popularity by Hour of Day")
        st.markdown("Identify when specific pizzas are in high demand to optimize preparation.")
        
        # Analyze ALL pizzas here too
        hourly_demand = filtered_df.groupby(['hour', 'pizza_name'])['quantity'].sum().reset_index()
        
        # Create a complete grid to handle missing hours/pizzas
        hourly_pivot = hourly_demand.pivot(index='pizza_name', columns='hour', values='quantity').fillna(0)
        
        fig_hourly_demand = px.imshow(
            hourly_pivot,
            labels=dict(x="Hour of Day", y="Pizza Type", color="Quantity"),
            title="Heatmap: All Pizzas Sold by Hour",
            color_continuous_scale='Magma_r', # Reversed Magma for clear visibility of hot spots
            aspect="auto",
            height=800 # Make chart taller
        )
        fig_hourly_demand.update_xaxes(dtick=1)
        st.plotly_chart(fig_hourly_demand, use_container_width=True)
        
        st.info("Tip: Use this chart to schedule prep work. For example, if 'Thai Chicken Pizza' peaks at 18:00, start prep at 17:00.")

    with tab3:
        st.subheader("Sales Trends by Category Over the Year")
        
        # 1. Ensure we have month numbers for sorting
        filtered_df['month_num'] = filtered_df['order_date'].dt.month
        
        # 2. Group by month_num (for sorting), month name (for display), and category
        monthly_cat_trend = filtered_df.groupby(['month_num', 'month', 'pizza_category'])['quantity'].sum().reset_index()
        
        # 3. CRITICAL: Sort by month_num to ensure Plotly connects Jan -> Feb -> Mar
        monthly_cat_trend = monthly_cat_trend.sort_values('month_num')
        
        fig_monthly_trend = px.line(
            monthly_cat_trend,
            x='month',
            y='quantity',
            color='pizza_category',
            title='Monthly Quantity Sold by Category',
            markers=True
        )
        
        # 4. Force x-axis order so the graph doesn't default to alphabetical (April, August...)
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        fig_monthly_trend.update_xaxes(categoryorder='array', categoryarray=months_order)
        
        st.plotly_chart(fig_monthly_trend, use_container_width=True)

    # Raw Data & Export
    # Note: Using st.dataframe directly with style.format can cause issues in some Streamlit versions on Cloud
    # It's safer to format a copy for display or just show the raw dataframe
    with st.expander("📂 Access Raw Data & Export"):
        # Create a display copy to avoid affecting the original dataframe logic if needed later
        display_df = filtered_df.copy()
        # We format columns here just by rounding/string conversion if needed, or rely on st.dataframe's native handling
        # Streamlit's dataframe handles numbers well, but let's just show the raw data without complex styling to avoid the Styler error
        st.dataframe(display_df)

else:
    st.warning("Please make sure 'pizza_sales.csv' is in the same directory.")
