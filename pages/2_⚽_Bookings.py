import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import calendar

# Page configuration
st.set_page_config(
    page_title="Booking Analytics - Infinity Sports Arena",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pricing constants
WEEKDAY_DAY_RATE = 3500  # 06:00-18:00
WEEKDAY_NIGHT_RATE = 4000  # 18:00-23:59
WEEKEND_RATE = 4000  # All times

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2ecc71;
        text-align: center;
        padding: 1rem 0;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e8f8f5;
        padding: 1rem;
        border-left: 4px solid #2ecc71;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Data loading function
@st.cache_data
def load_booking_data(file_path='kawdana_bookings.xlsx'):
    """Load and parse booking data from Excel file"""
    try:
        xl_file = pd.ExcelFile(file_path)
        sheet_names = xl_file.sheet_names
        
        all_bookings = []
        
        # Month mapping
        month_map = {
            'APR': (4, 'April', 2025),
            'MAY': (5, 'May', 2025),
            'JUNE': (6, 'June', 2025),
            'JUN': (6, 'June', 2025),
            'JULY': (7, 'July', 2025),
            'JUL': (7, 'July', 2025),
            'AUGUST': (8, 'August', 2025),
            'AUG': (8, 'August', 2025),
            'SEP': (9, 'September', 2025),
            'OCT': (10, 'October', 2025),
            'NOV': (11, 'November', 2025),
            'DEC': (12, 'December', 2025)
        }
        
        for sheet_name in sheet_names:
            # Extract month from sheet name
            month_key = sheet_name.split('-')[0].strip().upper()
            if month_key not in month_map:
                continue
            
            month_num, month_name, year = month_map[month_key]
            
            # Read sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            
            # Get days of week from row 3
            days_of_week = df.iloc[3, 1:32].tolist()
            
            # Parse time slots (starting from row 4, every 2 rows)
            for idx in range(4, len(df), 2):
                if pd.isna(df.iloc[idx, 0]):
                    continue
                
                time_slot = str(df.iloc[idx, 0])
                if ':' not in time_slot:
                    continue
                
                # Extract hour
                try:
                    hour = int(time_slot.split(':')[0])
                except:
                    continue
                
                # Parse each day
                for day_idx in range(1, 32):
                    if day_idx >= len(df.columns):
                        break
                    
                    is_booked = df.iloc[idx, day_idx]
                    
                    # Get day of week
                    day_of_week = days_of_week[day_idx - 1] if day_idx - 1 < len(days_of_week) else None
                    
                    # Skip if no day info
                    if pd.isna(day_of_week):
                        continue
                    
                    # Create date
                    try:
                        date = datetime(year, month_num, day_idx)
                    except:
                        continue
                    
                    # Determine pricing
                    is_weekend = str(day_of_week).upper() in ['SAT', 'SUN', 'SATURDAY', 'SUNDAY']
                    is_daytime = 6 <= hour < 18
                    
                    if is_weekend:
                        rate = WEEKEND_RATE
                        rate_type = 'Weekend'
                    elif is_daytime:
                        rate = WEEKDAY_DAY_RATE
                        rate_type = 'Weekday Day'
                    else:
                        rate = WEEKDAY_NIGHT_RATE
                        rate_type = 'Weekday Night'
                    
                    # Check if booked
                    booked = is_booked == True or str(is_booked).upper() == 'TRUE'
                    
                    all_bookings.append({
                        'Date': date,
                        'Month': month_name,
                        'Month_Num': month_num,
                        'Day': day_idx,
                        'Day_of_Week': day_of_week,
                        'Time_Slot': time_slot,
                        'Hour': hour,
                        'Is_Booked': booked,
                        'Rate': rate if booked else 0,
                        'Rate_Type': rate_type,
                        'Is_Weekend': is_weekend
                    })
        
        return pd.DataFrame(all_bookings)
    
    except Exception as e:
        st.error(f"Error loading booking data: {e}")
        return None

# Sidebar
with st.sidebar:
    st.markdown("### ⚽ Booking Analytics")
    st.markdown("**Kawdana Futsal Court**")
    st.markdown("---")
    
    # File upload option
    st.markdown("#### 📁 Data Source")
    uploaded_file = st.file_uploader(
        "Upload Booking Data (Excel)",
        type=['xlsx'],
        help="Upload kawdana_bookings.xlsx or similar format"
    )
    
    st.markdown("---")
    
    # Pricing info
    st.markdown("#### 💰 Pricing Structure")
    st.markdown(f"""
    **Weekday Day** (06:00-18:00)  
    Rs {WEEKDAY_DAY_RATE:,}/hour
    
    **Weekday Night** (18:00-23:59)  
    Rs {WEEKDAY_NIGHT_RATE:,}/hour
    
    **Weekend** (All times)  
    Rs {WEEKEND_RATE:,}/hour
    """)

# Load data
if uploaded_file:
    df = load_booking_data(uploaded_file)
else:
    df = load_booking_data('kawdana_bookings.xlsx')

if df is None or len(df) == 0:
    st.error("No booking data available. Please upload kawdana_bookings.xlsx file.")
    st.stop()

# Date filter
with st.sidebar:
    st.markdown("---")
    st.markdown("#### 📅 Filters")
    
    available_months = sorted(df['Month'].unique(), key=lambda x: df[df['Month']==x]['Month_Num'].iloc[0])
    
    selected_months = st.multiselect(
        "Select Months",
        options=available_months,
        default=available_months
    )
    
    if not selected_months:
        selected_months = available_months
    
    df_filtered = df[df['Month'].isin(selected_months)]
    
    # Time filter
    time_ranges = st.radio(
        "Time Range",
        ['All Times', 'Daytime (06:00-18:00)', 'Evening (18:00-23:59)']
    )
    
    if time_ranges == 'Daytime (06:00-18:00)':
        df_filtered = df_filtered[df_filtered['Hour'] < 18]
    elif time_ranges == 'Evening (18:00-23:59)':
        df_filtered = df_filtered[df_filtered['Hour'] >= 18]

# Main content
st.markdown('<h1 class="main-header">⚽ Booking Analytics Dashboard</h1>', unsafe_allow_html=True)

st.markdown(f"**Analysis Period:** {', '.join(selected_months)} | **Court:** Kawdana Futsal")

# Calculate metrics
total_slots = len(df_filtered)
booked_slots = df_filtered['Is_Booked'].sum()
occupancy_rate = (booked_slots / total_slots * 100) if total_slots > 0 else 0
total_revenue = df_filtered[df_filtered['Is_Booked']]['Rate'].sum()
avg_per_booking = total_revenue / booked_slots if booked_slots > 0 else 0

# KPIs
st.markdown("### 📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total Bookings",
        value=f"{booked_slots:,}",
        delta=f"{len(selected_months)} months"
    )

with col2:
    st.metric(
        label="Occupancy Rate",
        value=f"{occupancy_rate:.1f}%",
        delta="Of total slots"
    )

with col3:
    st.metric(
        label="Est. Revenue",
        value=f"Rs {total_revenue/1000:.1f}K",
        delta=f"Rs {total_revenue:,.0f}"
    )

with col4:
    st.metric(
        label="Avg per Booking",
        value=f"Rs {avg_per_booking:,.0f}",
        delta="Revenue/booking"
    )

with col5:
    best_month = df_filtered[df_filtered['Is_Booked']].groupby('Month')['Rate'].sum().idxmax() if booked_slots > 0 else "N/A"
    st.metric(
        label="Best Month",
        value=best_month,
        delta="Highest revenue"
    )

st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📅 Overview",
    "🔥 Heatmaps",
    "📈 Trends",
    "⏰ Peak Hours",
    "📊 Occupancy",
    "💡 Insights"
])

# Tab 1: Overview
with tab1:
    st.markdown("### Monthly Booking Summary")
    
    # Monthly stats
    monthly_stats = df_filtered[df_filtered['Is_Booked']].groupby('Month').agg({
        'Is_Booked': 'sum',
        'Rate': 'sum'
    }).reset_index()
    monthly_stats.columns = ['Month', 'Bookings', 'Revenue']
    
    # Sort by month number
    month_order = {m: df[df['Month']==m]['Month_Num'].iloc[0] for m in monthly_stats['Month']}
    monthly_stats['Month_Num'] = monthly_stats['Month'].map(month_order)
    monthly_stats = monthly_stats.sort_values('Month_Num')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly bookings
        fig = px.bar(
            monthly_stats,
            x='Month',
            y='Bookings',
            title='Monthly Bookings',
            text='Bookings',
            color='Bookings',
            color_continuous_scale='Greens'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Monthly revenue
        fig = px.bar(
            monthly_stats,
            x='Month',
            y='Revenue',
            title='Monthly Revenue (Estimated)',
            text=monthly_stats['Revenue'].apply(lambda x: f"Rs {x/1000:.0f}K"),
            color='Revenue',
            color_continuous_scale='Blues'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Weekday vs Weekend
    st.markdown("#### Weekday vs Weekend Comparison")
    
    weekend_stats = df_filtered[df_filtered['Is_Booked']].groupby('Is_Weekend').agg({
        'Is_Booked': 'sum',
        'Rate': 'sum'
    }).reset_index()
    weekend_stats['Type'] = weekend_stats['Is_Weekend'].map({True: 'Weekend', False: 'Weekday'})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.pie(
            weekend_stats,
            values='Is_Booked',
            names='Type',
            title='Bookings Distribution',
            color_discrete_sequence=['#3498db', '#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            weekend_stats,
            values='Rate',
            names='Type',
            title='Revenue Distribution',
            color_discrete_sequence=['#3498db', '#e74c3c']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        weekend_stats['Avg_Rate'] = weekend_stats['Rate'] / weekend_stats['Is_Booked']
        fig = px.bar(
            weekend_stats,
            x='Type',
            y='Avg_Rate',
            title='Average Rate per Booking',
            text=weekend_stats['Avg_Rate'].apply(lambda x: f"Rs {x:,.0f}"),
            color='Type',
            color_discrete_sequence=['#3498db', '#e74c3c']
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Heatmaps
with tab2:
    st.markdown("### Booking Heatmaps")
    
    # Select month for heatmap
    heatmap_month = st.selectbox(
        "Select Month for Detailed Heatmap",
        options=selected_months
    )
    
    month_df = df_filtered[df_filtered['Month'] == heatmap_month]
    
    # Create pivot table for heatmap
    heatmap_data = month_df.pivot_table(
        index='Time_Slot',
        columns='Day',
        values='Is_Booked',
        aggfunc='first',
        fill_value=False
    )
    
    # Convert boolean to numeric
    heatmap_data = heatmap_data.astype(int)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[[0, '#ecf0f1'], [1, '#27ae60']],
        text=heatmap_data.values,
        texttemplate='%{text}',
        showscale=False,
        hovertemplate='Day %{x}<br>%{y}<br>Booked: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Booking Heatmap - {heatmap_month}',
        xaxis_title='Day of Month',
        yaxis_title='Time Slot',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Hourly heatmap across all selected months
    st.markdown("#### Hourly Occupancy Pattern (All Selected Months)")
    
    hourly_dow = df_filtered[df_filtered['Is_Booked']].groupby(['Hour', 'Day_of_Week']).size().reset_index(name='Bookings')
    
    # Create pivot
    dow_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hourly_pivot = hourly_dow.pivot_table(
        index='Hour',
        columns='Day_of_Week',
        values='Bookings',
        fill_value=0
    )
    
    # Reorder columns
    hourly_pivot = hourly_pivot[[col for col in dow_order if col in hourly_pivot.columns]]
    
    fig = go.Figure(data=go.Heatmap(
        z=hourly_pivot.values,
        x=hourly_pivot.columns,
        y=hourly_pivot.index,
        colorscale='Greens',
        text=hourly_pivot.values,
        texttemplate='%{text}',
        hovertemplate='%{x}<br>Hour: %{y}:00<br>Bookings: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Bookings by Hour and Day of Week',
        xaxis_title='Day of Week',
        yaxis_title='Hour',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Trends
with tab3:
    st.markdown("### Booking Trends Over Time")
    
    # Daily bookings over time
    daily_bookings = df_filtered[df_filtered['Is_Booked']].groupby('Date').size().reset_index(name='Bookings')
    daily_bookings['7-Day Avg'] = daily_bookings['Bookings'].rolling(window=7, min_periods=1).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_bookings['Date'],
        y=daily_bookings['Bookings'],
        mode='markers',
        name='Daily Bookings',
        marker=dict(size=5, color='lightblue'),
        opacity=0.6
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_bookings['Date'],
        y=daily_bookings['7-Day Avg'],
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color='#27ae60', width=3)
    ))
    
    fig.update_layout(
        title='Daily Bookings Trend',
        xaxis_title='Date',
        yaxis_title='Number of Bookings',
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Month-over-month growth
        monthly_bookings = df_filtered[df_filtered['Is_Booked']].groupby('Month').size().reset_index(name='Bookings')
        monthly_bookings['Month_Num'] = monthly_bookings['Month'].map(
            {m: df[df['Month']==m]['Month_Num'].iloc[0] for m in monthly_bookings['Month']}
        )
        monthly_bookings = monthly_bookings.sort_values('Month_Num')
        monthly_bookings['MoM_Growth'] = monthly_bookings['Bookings'].pct_change() * 100
        
        fig = px.line(
            monthly_bookings,
            x='Month',
            y='Bookings',
            title='Monthly Booking Trend',
            markers=True,
            text='Bookings'
        )
        fig.update_traces(textposition='top center', line_color='#2ecc71')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # MoM growth
        growth_data = monthly_bookings[monthly_bookings['MoM_Growth'].notna()]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=growth_data['Month'],
            y=growth_data['MoM_Growth'],
            marker_color=growth_data['MoM_Growth'].apply(
                lambda x: '#27ae60' if x > 0 else '#e74c3c'
            ),
            text=growth_data['MoM_Growth'].apply(lambda x: f"{x:+.1f}%"),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Month-over-Month Growth',
            xaxis_title='Month',
            yaxis_title='Growth (%)',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Peak Hours
with tab4:
    st.markdown("### Peak Hours Analysis")
    
    # Bookings by hour
    hourly_bookings = df_filtered[df_filtered['Is_Booked']].groupby('Hour').agg({
        'Is_Booked': 'sum',
        'Rate': 'sum'
    }).reset_index()
    hourly_bookings.columns = ['Hour', 'Bookings', 'Revenue']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            hourly_bookings,
            x='Hour',
            y='Bookings',
            title='Bookings by Hour of Day',
            text='Bookings',
            color='Bookings',
            color_continuous_scale='Greens'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        fig.update_xaxes(tickmode='linear', tick0=6, dtick=1)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            hourly_bookings,
            x='Hour',
            y='Revenue',
            title='Revenue by Hour of Day',
            text=hourly_bookings['Revenue'].apply(lambda x: f"Rs {x/1000:.0f}K"),
            color='Revenue',
            color_continuous_scale='Blues'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        fig.update_xaxes(tickmode='linear', tick0=6, dtick=1)
        st.plotly_chart(fig, use_container_width=True)
    
    # Peak hours table
    st.markdown("#### Top 5 Peak Hours")
    
    top_hours = hourly_bookings.nlargest(5, 'Bookings')[['Hour', 'Bookings', 'Revenue']]
    top_hours['Time Slot'] = top_hours['Hour'].apply(lambda x: f"{x:02d}:00-{x+1:02d}:00")
    top_hours['Revenue'] = top_hours['Revenue'].apply(lambda x: f"Rs {x:,.0f}")
    top_hours = top_hours[['Time Slot', 'Bookings', 'Revenue']]
    
    st.dataframe(top_hours, use_container_width=True, hide_index=True)
    
    # Empty slots analysis
    st.markdown("#### Empty Slot Opportunities")
    
    empty_slots = df_filtered[~df_filtered['Is_Booked']].groupby('Hour').size().reset_index(name='Empty_Slots')
    empty_slots = empty_slots.nlargest(5, 'Empty_Slots')
    empty_slots['Time Slot'] = empty_slots['Hour'].apply(lambda x: f"{x:02d}:00-{x+1:02d}:00")
    empty_slots['Potential Revenue'] = empty_slots['Empty_Slots'] * WEEKDAY_DAY_RATE
    empty_slots['Potential Revenue'] = empty_slots['Potential Revenue'].apply(lambda x: f"Rs {x:,.0f}")
    empty_slots = empty_slots[['Time Slot', 'Empty_Slots', 'Potential Revenue']]
    
    st.dataframe(empty_slots, use_container_width=True, hide_index=True)

# Tab 5: Occupancy
with tab5:
    st.markdown("### Occupancy Analysis")
    
    # Overall occupancy
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_slots_period = len(df_filtered)
        booked_slots_period = df_filtered['Is_Booked'].sum()
        occupancy = (booked_slots_period / total_slots_period * 100) if total_slots_period > 0 else 0
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=occupancy,
            title={'text': "Overall Occupancy Rate"},
            delta={'reference': 50, 'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#27ae60"},
                'steps': [
                    {'range': [0, 33], 'color': "#ecf0f1"},
                    {'range': [33, 66], 'color': "#95a5a6"},
                    {'range': [66, 100], 'color': "#34495e"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Weekday occupancy
        weekday_data = df_filtered[~df_filtered['Is_Weekend']]
        weekday_occ = (weekday_data['Is_Booked'].sum() / len(weekday_data) * 100) if len(weekday_data) > 0 else 0
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=weekday_occ,
            title={'text': "Weekday Occupancy"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3498db"}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Weekend occupancy
        weekend_data = df_filtered[df_filtered['Is_Weekend']]
        weekend_occ = (weekend_data['Is_Booked'].sum() / len(weekend_data) * 100) if len(weekend_data) > 0 else 0
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=weekend_occ,
            title={'text': "Weekend Occupancy"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#e74c3c"}
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly occupancy trend
    monthly_occ = df_filtered.groupby('Month').apply(
        lambda x: (x['Is_Booked'].sum() / len(x) * 100) if len(x) > 0 else 0
    ).reset_index(name='Occupancy')
    
    # Sort by month number
    month_order = {m: df[df['Month']==m]['Month_Num'].iloc[0] for m in monthly_occ['Month']}
    monthly_occ['Month_Num'] = monthly_occ['Month'].map(month_order)
    monthly_occ = monthly_occ.sort_values('Month_Num')
    
    fig = px.line(
        monthly_occ,
        x='Month',
        y='Occupancy',
        title='Monthly Occupancy Trend',
        markers=True,
        text=monthly_occ['Occupancy'].apply(lambda x: f"{x:.1f}%")
    )
    fig.update_traces(textposition='top center', line_color='#2ecc71', line_width=3)
    fig.update_layout(yaxis_title='Occupancy Rate (%)', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of week occupancy
    dow_occ = df_filtered.groupby('Day_of_Week').apply(
        lambda x: (x['Is_Booked'].sum() / len(x) * 100) if len(x) > 0 else 0
    ).reset_index(name='Occupancy')
    
    # Reorder days
    dow_order_map = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}
    dow_occ['Order'] = dow_occ['Day_of_Week'].map(dow_order_map)
    dow_occ = dow_occ.sort_values('Order')
    
    fig = px.bar(
        dow_occ,
        x='Day_of_Week',
        y='Occupancy',
        title='Occupancy by Day of Week',
        text=dow_occ['Occupancy'].apply(lambda x: f"{x:.1f}%"),
        color='Occupancy',
        color_continuous_scale='Greens'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, yaxis_title='Occupancy Rate (%)', height=400)
    st.plotly_chart(fig, use_container_width=True)

# Tab 6: Insights
with tab6:
    st.markdown("### 🎯 Automated Insights & Recommendations")
    
    insights = []
    
    # 1. Overall occupancy insight
    if occupancy_rate < 40:
        insights.append({
            "title": "📉 Low Overall Occupancy",
            "desc": f"Current occupancy is **{occupancy_rate:.1f}%**. Consider promotional campaigns during off-peak hours to boost bookings.",
            "priority": "High"
        })
    elif occupancy_rate > 70:
        insights.append({
            "title": "📈 High Occupancy!",
            "desc": f"Excellent occupancy at **{occupancy_rate:.1f}%**! Consider dynamic pricing during peak hours to maximize revenue.",
            "priority": "Info"
        })
    
    # 2. Peak hours
    top_hour = hourly_bookings.loc[hourly_bookings['Bookings'].idxmax()]
    insights.append({
        "title": f"🔥 Peak Hour: {top_hour['Hour']:02d}:00-{top_hour['Hour']+1:02d}:00",
        "desc": f"This time slot has **{int(top_hour['Bookings'])} bookings** generating **Rs {top_hour['Revenue']:,.0f}**. Ensure optimal staffing during this period.",
        "priority": "Info"
    })
    
    # 3. Empty slots
    empty_count = len(df_filtered[~df_filtered['Is_Booked']])
    potential_revenue = empty_count * WEEKDAY_DAY_RATE
    insights.append({
        "title": "💡 Revenue Opportunity",
        "desc": f"**{empty_count:,} empty slots** represent potential revenue of **Rs {potential_revenue:,.0f}**. Target promotions for low-demand hours.",
        "priority": "High"
    })
    
    # 4. Weekday vs Weekend
    if len(weekend_stats) == 2:
        weekend_bookings = weekend_stats[weekend_stats['Is_Weekend'] == True]['Is_Booked'].iloc[0]
        weekday_bookings = weekend_stats[weekend_stats['Is_Weekend'] == False]['Is_Booked'].iloc[0]
        
        if weekend_bookings > weekday_bookings * 1.5:
            insights.append({
                "title": "🎉 Weekend Dominance",
                "desc": f"Weekend bookings (**{int(weekend_bookings)}**) are significantly higher than weekdays (**{int(weekday_bookings)}**). Focus weekday promotions to balance demand.",
                "priority": "Medium"
            })
    
    # 5. Monthly performance
    best_month_name = monthly_stats.loc[monthly_stats['Revenue'].idxmax(), 'Month']
    best_month_bookings = monthly_stats.loc[monthly_stats['Revenue'].idxmax(), 'Bookings']
    best_month_revenue = monthly_stats.loc[monthly_stats['Revenue'].idxmax(), 'Revenue']
    
    insights.append({
        "title": f"🏆 Best Month: {best_month_name}",
        "desc": f"Generated **{int(best_month_bookings)} bookings** and **Rs {best_month_revenue:,.0f}** in revenue. Analyze what made this month successful to replicate results.",
        "priority": "Info"
    })
    
    # 6. Time slot recommendations
    if not empty_slots.empty:
        worst_hour = empty_slots.iloc[0]['Time Slot']
        insights.append({
            "title": f"⏰ Low Demand: {worst_hour}",
            "desc": f"This time slot consistently has low bookings. Consider special rates or targeted marketing for this period.",
            "priority": "Medium"
        })
    
    # Display insights
    for insight in insights:
        color = {"High": "#e74c3c", "Medium": "#f39c12", "Info": "#2ecc71"}[insight["priority"]]
        
        st.markdown(f"""
        <div style="background-color: {color}20; padding: 1rem; border-left: 4px solid {color}; border-radius: 0.25rem; margin: 1rem 0;">
            <h4 style="margin: 0; color: {color};">{insight['title']}</h4>
            <p style="margin: 0.5rem 0 0 0;">{insight['desc']}</p>
            <span style="font-size: 0.85rem; color: {color};"><strong>Priority: {insight['priority']}</strong></span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("### 💡 Strategic Recommendations")
    
    recommendations = []
    
    if occupancy_rate < 50:
        recommendations.append("**Increase Occupancy**: Launch promotional campaigns targeting off-peak hours with discounted rates.")
    
    if weekday_occ < 40:
        recommendations.append("**Weekday Specials**: Offer corporate packages or group discounts (Mon-Fri) to boost weekday bookings.")
    
    if weekend_occ > 70:
        recommendations.append("**Weekend Premium**: Consider higher rates for peak weekend hours to maximize revenue.")
    
    recommendations.append("**Early Bird Discount**: Promote morning slots (06:00-09:00) with special pricing to utilize empty capacity.")
    
    recommendations.append("**Evening Rush Strategy**: Optimize 18:00-21:00 slots (your peak hours) with premium pricing and priority booking.")
    
    if len(monthly_stats) > 1:
        lowest_month = monthly_stats.loc[monthly_stats['Bookings'].idxmin(), 'Month']
        recommendations.append(f"**{lowest_month} Boost**: This was your slowest month. Plan targeted campaigns or events during similar periods.")
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"📅 **Analysis Period:** {', '.join(selected_months)}")
with col2:
    st.markdown(f"📊 **Total Months:** {len(selected_months)}")
with col3:
    st.markdown(f"🔒 **Confidential** | Updated: {datetime.now().strftime('%Y-%m-%d')}")
