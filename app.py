import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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
