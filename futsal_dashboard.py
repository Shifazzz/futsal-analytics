import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

# Page configuration
st.set_page_config(
    page_title="Futsal Court Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Password protection (optional - uncomment if needed)
# def check_password():
#     """Returns `True` if the user had the correct password."""
#     def password_entered():
#         """Checks whether a password entered by the user is correct."""
#         if st.session_state["password"] == "your_secure_password_here":
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]
#         else:
#             st.session_state["password_correct"] = False
#     
#     if "password_correct" not in st.session_state:
#         st.text_input("Password", type="password", on_change=password_entered, key="password")
#         return False
#     elif not st.session_state["password_correct"]:
#         st.text_input("Password", type="password", on_change=password_entered, key="password")
#         st.error("😕 Password incorrect")
#         return False
#     else:
#         return True
# 
# if not check_password():
#     st.stop()

# Generate mock data function
@st.cache_data
def generate_mock_data():
    """Generate mock futsal court data for demonstration"""
    np.random.seed(42)
    
    start_date = datetime.now() - timedelta(days=180)
    dates = pd.date_range(start=start_date, periods=180, freq='D')
    time_slots = [f"{h:02d}:00" for h in range(8, 23)]
    
    data = []
    
    for date in dates:
        day_of_week = date.dayofweek
        is_weekend = day_of_week >= 5
        
        for court in ['Court 1', 'Court 2']:
            for time_slot in time_slots:
                hour = int(time_slot.split(':')[0])
                is_peak = 18 <= hour <= 22
                
                base_prob = 0.45 if is_weekend else 0.35
                if is_peak:
                    base_prob += 0.30
                if court == 'Court 2':
                    base_prob *= 0.85
                
                is_booked = np.random.random() < base_prob
                
                if is_booked:
                    base_price = 50 if is_peak else 40
                    revenue = base_price + np.random.normal(0, 5)
                    
                    customer_type = np.random.choice(
                        ['Regular', 'One-time', 'Corporate'],
                        p=[0.55, 0.35, 0.10]
                    )
                    
                    cancellation_prob = 0.08 if customer_type == 'One-time' else 0.03
                    was_cancelled = np.random.random() < cancellation_prob
                    
                    no_show_prob = 0.05 if customer_type == 'One-time' else 0.02
                    was_no_show = np.random.random() < no_show_prob if not was_cancelled else False
                    
                    if customer_type == 'Regular':
                        advance_days = np.random.choice([1, 2, 3, 7], p=[0.2, 0.3, 0.3, 0.2])
                    elif customer_type == 'Corporate':
                        advance_days = np.random.choice([7, 14, 21], p=[0.5, 0.3, 0.2])
                    else:
                        advance_days = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
                    
                    data.append({
                        'date': date,
                        'court': court,
                        'time_slot': time_slot,
                        'hour': hour,
                        'day_of_week': date.strftime('%A'),
                        'is_weekend': is_weekend,
                        'is_peak_hour': is_peak,
                        'is_booked': True,
                        'revenue': revenue,
                        'customer_type': customer_type,
                        'was_cancelled': was_cancelled,
                        'was_no_show': was_no_show,
                        'advance_booking_days': advance_days,
                        'actual_revenue': 0 if (was_cancelled or was_no_show) else revenue
                    })
                else:
                    data.append({
                        'date': date,
                        'court': court,
                        'time_slot': time_slot,
                        'hour': hour,
                        'day_of_week': date.strftime('%A'),
                        'is_weekend': is_weekend,
                        'is_peak_hour': is_peak,
                        'is_booked': False,
                        'revenue': 0,
                        'customer_type': None,
                        'was_cancelled': False,
                        'was_no_show': False,
                        'advance_booking_days': None,
                        'actual_revenue': 0
                    })
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['year_month'] = df['date'].dt.to_period('M')
    return df

# Load data
@st.cache_data
def load_data(uploaded_file=None):
    """Load data from file or generate mock data"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df['date'] = pd.to_datetime(df['date'])
            df['year_month'] = df['date'].dt.to_period('M')
            return df, False
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return generate_mock_data(), True
    else:
        return generate_mock_data(), True

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=Futsal+Analytics", use_container_width=True)
    st.markdown("### ⚽ Futsal Court Analytics")
    st.markdown("---")
    
    # File upload
    st.markdown("#### 📁 Data Source")
    uploaded_file = st.file_uploader(
        "Upload your data (CSV)",
        type=['csv'],
        help="Upload a CSV file with your futsal court booking data"
    )
    
    if uploaded_file is None:
        st.info("📊 Using mock data for demonstration")
    
    st.markdown("---")
    
    # Date filter
    st.markdown("#### 📅 Filters")
    
# Load data
df, is_mock = load_data(uploaded_file)
booked_df = df[df['is_booked'] == True].copy()

# Date range filter
date_min = df['date'].min().date()
date_max = df['date'].max().date()

with st.sidebar:
    date_range = st.date_input(
        "Date Range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max
    )
    
    if len(date_range) == 2:
        df_filtered = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])]
        booked_df_filtered = df_filtered[df_filtered['is_booked'] == True].copy()
    else:
        df_filtered = df
        booked_df_filtered = booked_df
    
    # Court filter
    courts = ['All'] + list(df['court'].unique())
    selected_court = st.selectbox("Court", courts)
    
    if selected_court != 'All':
        df_filtered = df_filtered[df_filtered['court'] == selected_court]
        booked_df_filtered = booked_df_filtered[booked_df_filtered['court'] == selected_court]
    
    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.markdown("""
    This dashboard provides comprehensive analytics for futsal court operations.
    
    **Confidential Data** 🔒
    """)

# Main content
st.markdown('<h1 class="main-header">⚽ Futsal Court Analytics Dashboard</h1>', unsafe_allow_html=True)

if is_mock:
    st.warning("⚠️ Currently displaying MOCK DATA for demonstration. Upload your CSV file to see your actual data.")

# Calculate metrics
total_slots = len(df_filtered)
booked_slots = df_filtered['is_booked'].sum()
utilization_rate = (booked_slots / total_slots) * 100 if total_slots > 0 else 0

total_potential_revenue = booked_df_filtered['revenue'].sum()
total_actual_revenue = booked_df_filtered['actual_revenue'].sum()
revenue_loss = total_potential_revenue - total_actual_revenue

cancellation_rate = (booked_df_filtered['was_cancelled'].sum() / len(booked_df_filtered)) * 100 if len(booked_df_filtered) > 0 else 0
no_show_rate = (booked_df_filtered['was_no_show'].sum() / len(booked_df_filtered)) * 100 if len(booked_df_filtered) > 0 else 0

# Key Metrics
st.markdown("### 📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Overall Utilization",
        value=f"{utilization_rate:.1f}%",
        delta=f"{booked_slots:,} / {total_slots:,} slots"
    )

with col2:
    st.metric(
        label="Total Revenue",
        value=f"${total_actual_revenue:,.0f}",
        delta=f"-${revenue_loss:.0f} lost" if revenue_loss > 0 else "No losses"
    )

with col3:
    st.metric(
        label="Cancellation Rate",
        value=f"{cancellation_rate:.2f}%",
        delta=f"{int(booked_df_filtered['was_cancelled'].sum())} bookings"
    )

with col4:
    st.metric(
        label="No-Show Rate",
        value=f"{no_show_rate:.2f}%",
        delta=f"{int(booked_df_filtered['was_no_show'].sum())} bookings"
    )

st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Utilization",
    "💰 Revenue",
    "👥 Customers",
    "⏰ Time Patterns",
    "📉 Operational Issues",
    "🎯 Insights"
])

# Tab 1: Utilization
with tab1:
    st.markdown("### Court Utilization Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly utilization by court
        monthly_util = df_filtered.groupby(['year_month', 'court']).agg({
            'is_booked': lambda x: (x.sum() / len(x)) * 100
        }).reset_index()
        monthly_util.columns = ['year_month', 'court', 'utilization']
        monthly_util['year_month'] = monthly_util['year_month'].astype(str)
        
        fig = px.line(
            monthly_util,
            x='year_month',
            y='utilization',
            color='court',
            title='Monthly Utilization Rate by Court',
            labels={'utilization': 'Utilization (%)', 'year_month': 'Month'},
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Court comparison
        court_util = df_filtered.groupby('court').agg({
            'is_booked': ['count', 'sum', lambda x: (x.sum() / len(x)) * 100]
        }).round(2)
        court_util.columns = ['Total Slots', 'Booked Slots', 'Utilization %']
        
        fig = go.Figure(data=[
            go.Bar(
                x=court_util.index,
                y=court_util['Utilization %'],
                text=court_util['Utilization %'].apply(lambda x: f'{x:.1f}%'),
                textposition='auto',
                marker_color=['#2ecc71', '#3498db']
            )
        ])
        fig.update_layout(
            title='Court Utilization Comparison',
            yaxis_title='Utilization (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.markdown("#### Utilization Heatmap: Day vs Hour")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = df_filtered.groupby(['day_of_week', 'hour'])['is_booked'].apply(
        lambda x: (x.sum() / len(x)) * 100
    ).reset_index()
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='is_booked')
    heatmap_pivot = heatmap_pivot.reindex(day_order)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='RdYlGn',
        text=heatmap_pivot.values.round(1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Utilization %")
    ))
    fig.update_layout(
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Revenue
with tab2:
    st.markdown("### Revenue Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly revenue trend
        monthly_revenue = booked_df_filtered.groupby('year_month').agg({
            'revenue': 'sum',
            'actual_revenue': 'sum'
        }).reset_index()
        monthly_revenue['year_month'] = monthly_revenue['year_month'].astype(str)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_revenue['year_month'],
            y=monthly_revenue['revenue'],
            name='Potential Revenue',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            x=monthly_revenue['year_month'],
            y=monthly_revenue['actual_revenue'],
            name='Actual Revenue',
            marker_color='darkblue'
        ))
        fig.update_layout(
            title='Monthly Revenue: Potential vs Actual',
            xaxis_title='Month',
            yaxis_title='Revenue ($)',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue by court
        court_revenue = booked_df_filtered.groupby('court').agg({
            'actual_revenue': 'sum',
            'is_booked': 'count'
        }).round(2)
        court_revenue.columns = ['Total Revenue', 'Bookings']
        court_revenue['Avg per Booking'] = (court_revenue['Total Revenue'] / court_revenue['Bookings']).round(2)
        
        fig = go.Figure(data=[
            go.Bar(
                x=court_revenue.index,
                y=court_revenue['Total Revenue'],
                text=court_revenue['Total Revenue'].apply(lambda x: f'${x:,.0f}'),
                textposition='auto',
                marker_color=['#27ae60', '#2980b9']
            )
        ])
        fig.update_layout(
            title='Total Revenue by Court',
            yaxis_title='Revenue ($)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Revenue breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        # Revenue by hour
        hourly_revenue = booked_df_filtered.groupby('hour')['actual_revenue'].sum().reset_index()
        fig = px.bar(
            hourly_revenue,
            x='hour',
            y='actual_revenue',
            title='Revenue Distribution by Hour',
            labels={'hour': 'Hour of Day', 'actual_revenue': 'Revenue ($)'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Revenue by customer type
        customer_revenue = booked_df_filtered.groupby('customer_type')['actual_revenue'].sum().reset_index()
        fig = px.pie(
            customer_revenue,
            values='actual_revenue',
            names='customer_type',
            title='Revenue by Customer Type',
            hole=0.3
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Customers
with tab3:
    st.markdown("### Customer Behavior Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer type distribution
        customer_dist = booked_df_filtered['customer_type'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=customer_dist.index,
            values=customer_dist.values,
            hole=0.3,
            marker=dict(colors=['#2ecc71', '#3498db', '#e74c3c'])
        )])
        fig.update_layout(
            title='Customer Type Distribution',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Advance booking patterns
        advance_booking = booked_df_filtered['advance_booking_days'].value_counts().sort_index()
        fig = go.Figure(data=[
            go.Bar(
                x=advance_booking.index,
                y=advance_booking.values,
                marker_color='teal'
            )
        ])
        fig.update_layout(
            title='Advance Booking Patterns',
            xaxis_title='Days in Advance',
            yaxis_title='Number of Bookings',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer metrics table
    st.markdown("#### Customer Segment Metrics")
    customer_metrics = booked_df_filtered.groupby('customer_type').agg({
        'actual_revenue': ['sum', 'mean', 'count'],
        'was_cancelled': 'sum',
        'was_no_show': 'sum'
    }).round(2)
    customer_metrics.columns = ['Total Revenue', 'Avg Revenue', 'Bookings', 'Cancellations', 'No-Shows']
    customer_metrics['% of Total Revenue'] = (
        (customer_metrics['Total Revenue'] / customer_metrics['Total Revenue'].sum()) * 100
    ).round(2)
    st.dataframe(customer_metrics, use_container_width=True)

# Tab 4: Time Patterns
with tab4:
    st.markdown("### Time-Based Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weekday vs Weekend
        day_type_util = df_filtered.groupby('is_weekend')['is_booked'].apply(
            lambda x: (x.sum() / len(x)) * 100
        ).round(2)
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Weekday', 'Weekend'],
                y=[day_type_util[False], day_type_util[True]],
                text=[f"{day_type_util[False]:.1f}%", f"{day_type_util[True]:.1f}%"],
                textposition='auto',
                marker_color=['#3498db', '#e74c3c']
            )
        ])
        fig.update_layout(
            title='Utilization: Weekday vs Weekend',
            yaxis_title='Utilization Rate (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Peak vs Off-Peak
        peak_util = df_filtered.groupby('is_peak_hour')['is_booked'].apply(
            lambda x: (x.sum() / len(x)) * 100
        ).round(2)
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Off-Peak', 'Peak Hours'],
                y=[peak_util[False], peak_util[True]],
                text=[f"{peak_util[False]:.1f}%", f"{peak_util[True]:.1f}%"],
                textposition='auto',
                marker_color=['#95a5a6', '#f39c12']
            )
        ])
        fig.update_layout(
            title='Utilization: Peak vs Off-Peak Hours',
            yaxis_title='Utilization Rate (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Daily trend
    daily_bookings = df_filtered[df_filtered['is_booked'] == True].groupby('date').size().reset_index(name='bookings')
    daily_bookings['ma_7'] = daily_bookings['bookings'].rolling(window=7).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_bookings['date'],
        y=daily_bookings['bookings'],
        mode='lines',
        name='Daily Bookings',
        line=dict(color='lightblue', width=1),
        opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=daily_bookings['date'],
        y=daily_bookings['ma_7'],
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color='darkblue', width=3)
    ))
    fig.update_layout(
        title='Daily Bookings Trend',
        xaxis_title='Date',
        yaxis_title='Bookings',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 5: Operational Issues
with tab5:
    st.markdown("### Operational Issues Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Cancellation Rate",
            f"{cancellation_rate:.2f}%",
            delta=f"{int(booked_df_filtered['was_cancelled'].sum())} bookings"
        )
    
    with col2:
        st.metric(
            "No-Show Rate",
            f"{no_show_rate:.2f}%",
            delta=f"{int(booked_df_filtered['was_no_show'].sum())} bookings"
        )
    
    with col3:
        st.metric(
            "Total Revenue Loss",
            f"${revenue_loss:,.0f}",
            delta=f"{(revenue_loss/total_potential_revenue*100):.1f}% of potential"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Issues by customer type
        issues_by_customer = booked_df_filtered.groupby('customer_type').agg({
            'was_cancelled': ['sum', lambda x: (x.sum() / len(x)) * 100],
            'was_no_show': ['sum', lambda x: (x.sum() / len(x)) * 100]
        }).round(2)
        issues_by_customer.columns = ['Cancellations', 'Cancel Rate %', 'No-Shows', 'No-Show Rate %']
        
        st.markdown("#### Issues by Customer Type")
        st.dataframe(issues_by_customer, use_container_width=True)
    
    with col2:
        # Revenue loss by customer type
        revenue_loss_by_type = booked_df_filtered.groupby('customer_type').apply(
            lambda x: x['revenue'].sum() - x['actual_revenue'].sum()
        ).reset_index(name='revenue_loss')
        
        fig = go.Figure(data=[
            go.Bar(
                x=revenue_loss_by_type['customer_type'],
                y=revenue_loss_by_type['revenue_loss'],
                marker_color='crimson',
                text=revenue_loss_by_type['revenue_loss'].apply(lambda x: f'${x:.0f}'),
                textposition='auto'
            )
        ])
        fig.update_layout(
            title='Revenue Loss by Customer Type',
            xaxis_title='Customer Type',
            yaxis_title='Revenue Loss ($)',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 6: Insights
with tab6:
    st.markdown("### 🎯 Key Insights & Recommendations")
    
    # Generate insights
    insights = []
    recommendations = []
    
    # Best performing court
    court_util_tab = df_filtered.groupby('court').agg({
        'is_booked': lambda x: (x.sum() / len(x)) * 100
    }).round(2)
    best_court = court_util_tab.idxmax()[0]
    insights.append(f"🏆 **{best_court}** has the highest utilization at **{court_util_tab.loc[best_court].values[0]:.1f}%**")
    
    # Peak performance
    hourly_util_tab = df_filtered.groupby('hour').agg({
        'is_booked': lambda x: (x.sum() / len(x)) * 100
    }).round(2)
    top_3_hours = hourly_util_tab.nlargest(3, 'is_booked')
    peak_hours_str = ', '.join([f'{int(h)}:00' for h in top_3_hours.index])
    insights.append(f"⏰ **Peak hours** are {peak_hours_str} with average **{top_3_hours['is_booked'].mean():.1f}%** utilization")
    
    # Customer insights
    if len(booked_df_filtered) > 0:
        top_customer = booked_df_filtered.groupby('customer_type')['actual_revenue'].sum().idxmax()
        top_customer_revenue = booked_df_filtered.groupby('customer_type')['actual_revenue'].sum().max()
        insights.append(f"💎 **{top_customer}** customers generate the most revenue: **${top_customer_revenue:,.0f}**")
    
    # Display insights
    st.markdown("#### 📊 Current Performance")
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Generate recommendations
    if utilization_rate < 60:
        recommendations.append({
            "title": "📈 Increase Utilization",
            "desc": f"Current utilization is {utilization_rate:.1f}%. Launch targeted campaigns for off-peak slots.",
            "priority": "High"
        })
    
    if cancellation_rate + no_show_rate > 5:
        recommendations.append({
            "title": "⚠️ Reduce No-Shows",
            "desc": f"Combined cancellation/no-show rate is {(cancellation_rate + no_show_rate):.1f}%. Implement SMS reminders or deposits.",
            "priority": "High"
        })
    
    if len(df_filtered['court'].unique()) > 1:
        court_diff = court_util_tab.max()[0] - court_util_tab.min()[0]
        if court_diff > 10:
            recommendations.append({
                "title": "🏟️ Balance Court Usage",
                "desc": f"{court_diff:.1f}% gap between courts. Investigate and improve underperforming court.",
                "priority": "Medium"
            })
    
    regular_pct = (booked_df_filtered['customer_type'] == 'Regular').sum() / len(booked_df_filtered) * 100 if len(booked_df_filtered) > 0 else 0
    if regular_pct < 60:
        recommendations.append({
            "title": "🎯 Build Loyalty",
            "desc": f"Only {regular_pct:.1f}% are regular customers. Implement loyalty programs.",
            "priority": "Medium"
        })
    
    # Display recommendations
    if recommendations:
        st.markdown("#### 💡 Actionable Recommendations")
        for rec in recommendations:
            color = "#e74c3c" if rec["priority"] == "High" else "#f39c12"
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 1rem; border-left: 4px solid {color}; border-radius: 0.25rem; margin: 1rem 0;">
                <h4 style="margin: 0; color: {color};">{rec['title']}</h4>
                <p style="margin: 0.5rem 0 0 0;">{rec['desc']}</p>
                <span style="font-size: 0.85rem; color: {color};"><strong>Priority: {rec['priority']}</strong></span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("✅ Operations are performing well! Continue monitoring metrics.")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"📅 **Data Range:** {df_filtered['date'].min().date()} to {df_filtered['date'].max().date()}")
with col2:
    st.markdown(f"📊 **Total Records:** {len(df_filtered):,}")
with col3:
    st.markdown(f"🔒 **Confidential** | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")