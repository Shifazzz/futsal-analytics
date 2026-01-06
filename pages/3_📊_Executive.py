import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Executive Summary - Infinity Sports Arena",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #8e44ad;
        text-align: center;
        padding: 1rem 0;
        font-weight: bold;
    }
    .insight-box {
        background-color: #f4ecf7;
        padding: 1rem;
        border-left: 4px solid #8e44ad;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #fff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load financial data
@st.cache_data
def load_financial_data():
    """Load financial data"""
    try:
        df = pd.read_csv('financial_data.csv')
        df.columns = df.columns.str.strip()
        
        numeric_cols = ['Sports_Income_Total', 'FB_Income_Total', 'Total_Income',
                       'Sports_Exp_Total', 'FB_Exp_Total', 'Total_Expenses',
                       'Sports_Surplus', 'FB_Surplus', 'Total_Surplus']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: float(str(x).replace(',', '').replace('"', '')) if pd.notna(x) and x != '' else 0)
        
        df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
        df = df[df['Total_Income'] > 0].reset_index(drop=True)
        
        return df
    except:
        return None

# Load booking data
@st.cache_data
def load_booking_data():
    """Load booking data summary"""
    try:
        xl_file = pd.ExcelFile('kawdana_bookings.xlsx')
        sheet_names = xl_file.sheet_names
        
        month_map = {
            'APR': ('April', 4), 'MAY': ('May', 5), 'JUNE': ('June', 6), 'JUN': ('June', 6),
            'JULY': ('July', 7), 'JUL': ('July', 7), 'AUGUST': ('August', 8), 'AUG': ('August', 8),
            'SEP': ('September', 9), 'OCT': ('October', 10), 'NOV': ('November', 11), 'DEC': ('December', 12)
        }
        
        booking_summary = []
        
        for sheet_name in sheet_names:
            month_key = sheet_name.split('-')[0].strip().upper()
            if month_key not in month_map:
                continue
            
            month_name, month_num = month_map[month_key]
            df = pd.read_excel('kawdana_bookings.xlsx', sheet_name=sheet_name, header=None)
            
            # Count bookings
            bookings = 0
            revenue = 0
            
            days_of_week = df.iloc[3, 1:32].tolist()
            
            for idx in range(4, len(df), 2):
                if pd.isna(df.iloc[idx, 0]):
                    continue
                
                time_slot = str(df.iloc[idx, 0])
                if ':' not in time_slot:
                    continue
                
                hour = int(time_slot.split(':')[0])
                
                for day_idx in range(1, 32):
                    if day_idx >= len(df.columns):
                        break
                    
                    is_booked = df.iloc[idx, day_idx]
                    if is_booked == True or str(is_booked).upper() == 'TRUE':
                        bookings += 1
                        
                        day_of_week = days_of_week[day_idx - 1] if day_idx - 1 < len(days_of_week) else None
                        is_weekend = str(day_of_week).upper() in ['SAT', 'SUN', 'SATURDAY', 'SUNDAY']
                        is_daytime = 6 <= hour < 18
                        
                        if is_weekend:
                            revenue += 4000
                        elif is_daytime:
                            revenue += 3500
                        else:
                            revenue += 4000
            
            booking_summary.append({
                'Month': month_name,
                'Month_Num': month_num,
                'Bookings': bookings,
                'Est_Revenue': revenue
            })
        
        return pd.DataFrame(booking_summary)
    except:
        return None

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Executive Summary")
    st.markdown("**Combined Business View**")
    st.markdown("---")
    
    st.markdown("""
    This dashboard combines:
    - 💰 Financial performance
    - ⚽ Booking operations
    - 🎯 Strategic insights
    """)

# Load data
financial_df = load_financial_data()
booking_df = load_booking_data()

if financial_df is None or booking_df is None:
    st.error("Unable to load data. Please ensure both financial_data.csv and kawdana_bookings.xlsx are available.")
    st.stop()

# Main content
st.markdown('<h1 class="main-header">📊 Executive Summary Dashboard</h1>', unsafe_allow_html=True)

st.markdown("**Unified Business Intelligence - Kawdana Futsal Court**")

# Merge datasets
merged_df = pd.merge(
    financial_df[['Month_Name', 'Sports_Income_Total', 'Total_Income', 'Sports_Surplus', 'Total_Surplus']],
    booking_df[['Month', 'Bookings', 'Est_Revenue']],
    left_on='Month_Name',
    right_on='Month',
    how='outer'
)

# Overall KPIs
st.markdown("### 🎯 Key Business Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

total_financial_revenue = financial_df['Sports_Income_Total'].sum()
total_bookings = booking_df['Bookings'].sum()
total_est_revenue = booking_df['Est_Revenue'].sum()
total_profit = financial_df['Total_Surplus'].sum()
avg_margin = financial_df['Overall_Margin_%'].mean()

with col1:
    st.metric(
        label="Total Sports Revenue",
        value=f"Rs {total_financial_revenue/1000000:.2f}M",
        delta="From financial data"
    )

with col2:
    st.metric(
        label="Total Bookings",
        value=f"{total_bookings:,}",
        delta="9 months"
    )

with col3:
    st.metric(
        label="Est. from Bookings",
        value=f"Rs {total_est_revenue/1000000:.2f}M",
        delta="Booking data"
    )

with col4:
    st.metric(
        label="Total Profit",
        value=f"Rs {total_profit/1000000:.2f}M",
        delta=f"{avg_margin:.1f}% margin"
    )

with col5:
    revenue_per_booking = total_financial_revenue / total_bookings if total_bookings > 0 else 0
    st.metric(
        label="Revenue/Booking",
        value=f"Rs {revenue_per_booking:,.0f}",
        delta="Actual average"
    )

st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Unified View",
    "🔗 Correlation Analysis",
    "💡 Gap Analysis",
    "🎯 Strategic Insights"
])

# Tab 1: Unified View
with tab1:
    st.markdown("### Combined Performance Overview")
    
    # Revenue comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Actual vs Estimated Revenue
        comparison_df = merged_df.dropna(subset=['Sports_Income_Total', 'Est_Revenue'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=comparison_df['Month_Name'],
            y=comparison_df['Sports_Income_Total'],
            name='Actual Sports Revenue',
            marker_color='#2ecc71'
        ))
        
        fig.add_trace(go.Bar(
            x=comparison_df['Month_Name'],
            y=comparison_df['Est_Revenue'],
            name='Est. from Bookings',
            marker_color='#3498db'
        ))
        
        fig.update_layout(
            title='Actual vs Estimated Revenue Comparison',
            xaxis_title='Month',
            yaxis_title='Revenue (Rs)',
            barmode='group',
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bookings vs Revenue scatter
        if not comparison_df.empty:
            fig = px.scatter(
                comparison_df,
                x='Bookings',
                y='Sports_Income_Total',
                size='Bookings',
                color='Sports_Income_Total',
                hover_name='Month_Name',
                title='Bookings vs Actual Revenue',
                labels={'Sports_Income_Total': 'Actual Revenue (Rs)', 'Bookings': 'Number of Bookings'},
                color_continuous_scale='Greens',
                trendline='ols'
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
    
    # Monthly performance table
    st.markdown("#### Monthly Performance Summary")
    
    summary_table = merged_df[['Month_Name', 'Bookings', 'Est_Revenue', 'Sports_Income_Total', 'Sports_Surplus']].copy()
    summary_table.columns = ['Month', 'Bookings', 'Est. Revenue (Rs)', 'Actual Revenue (Rs)', 'Profit (Rs)']
    
    summary_table['Revenue/Booking'] = summary_table['Actual Revenue (Rs)'] / summary_table['Bookings']
    
    # Format numbers
    for col in ['Est. Revenue (Rs)', 'Actual Revenue (Rs)', 'Profit (Rs)', 'Revenue/Booking']:
        summary_table[col] = summary_table[col].apply(lambda x: f"Rs {x:,.0f}" if pd.notna(x) else "N/A")
    
    summary_table['Bookings'] = summary_table['Bookings'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
    
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

# Tab 2: Correlation Analysis
with tab2:
    st.markdown("### 🔗 Relationship Between Bookings and Revenue")
    
    comparison_df = merged_df.dropna(subset=['Sports_Income_Total', 'Est_Revenue', 'Bookings'])
    
    if not comparison_df.empty:
        # Calculate correlation
        correlation = comparison_df['Bookings'].corr(comparison_df['Sports_Income_Total'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Correlation Coefficient",
                value=f"{correlation:.3f}",
                delta="Bookings vs Revenue"
            )
        
        with col2:
            avg_diff = (comparison_df['Sports_Income_Total'] - comparison_df['Est_Revenue']).mean()
            st.metric(
                label="Avg Revenue Difference",
                value=f"Rs {avg_diff:,.0f}",
                delta="Actual - Estimated"
            )
        
        with col3:
            pct_diff = (avg_diff / comparison_df['Est_Revenue'].mean() * 100)
            st.metric(
                label="Avg % Difference",
                value=f"{pct_diff:+.1f}%",
                delta="Pricing variance"
            )
        
        # Detailed correlation plot
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter with trendline
            fig = px.scatter(
                comparison_df,
                x='Bookings',
                y='Sports_Income_Total',
                trendline='ols',
                title=f'Correlation: {correlation:.3f}',
                labels={'Sports_Income_Total': 'Actual Revenue (Rs)', 'Bookings': 'Bookings'},
                hover_data=['Month_Name']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"""
            **Interpretation:** 
            - Correlation of {correlation:.3f} indicates a {"strong" if abs(correlation) > 0.7 else "moderate" if abs(correlation) > 0.4 else "weak"} positive relationship
            - Higher bookings generally lead to higher revenue
            - Other factors (pricing, customer type) also influence revenue
            """)
        
        with col2:
            # Revenue per booking trend
            comparison_df['Revenue_Per_Booking'] = comparison_df['Sports_Income_Total'] / comparison_df['Bookings']
            
            fig = px.line(
                comparison_df,
                x='Month_Name',
                y='Revenue_Per_Booking',
                title='Revenue per Booking Trend',
                markers=True,
                text=comparison_df['Revenue_Per_Booking'].apply(lambda x: f"Rs {x:,.0f}")
            )
            fig.update_traces(textposition='top center', line_color='#8e44ad', line_width=3)
            fig.update_layout(yaxis_title='Revenue per Booking (Rs)', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            avg_rpb = comparison_df['Revenue_Per_Booking'].mean()
            st.info(f"""
            **Average Revenue per Booking: Rs {avg_rpb:,.0f}**
            - Estimated rate: Rs 3,915
            - Actual average: Rs {avg_rpb:,.0f}
            - Difference: Rs {avg_rpb - 3915:+,.0f} ({(avg_rpb - 3915)/3915*100:+.1f}%)
            """)

# Tab 3: Gap Analysis
with tab3:
    st.markdown("### 💡 Revenue Gap Analysis")
    
    comparison_df = merged_df.dropna(subset=['Sports_Income_Total', 'Est_Revenue'])
    comparison_df['Gap'] = comparison_df['Sports_Income_Total'] - comparison_df['Est_Revenue']
    comparison_df['Gap_%'] = (comparison_df['Gap'] / comparison_df['Est_Revenue']) * 100
    
    # Overall gap
    total_gap = comparison_df['Gap'].sum()
    total_est = comparison_df['Est_Revenue'].sum()
    total_actual = comparison_df['Sports_Income_Total'].sum()
    gap_pct = (total_gap / total_est * 100)
    
    st.markdown("#### Overall Gap Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Estimated",
            value=f"Rs {total_est/1000000:.2f}M",
            delta="From bookings"
        )
    
    with col2:
        st.metric(
            label="Total Actual",
            value=f"Rs {total_actual/1000000:.2f}M",
            delta="Financial data"
        )
    
    with col3:
        st.metric(
            label="Revenue Gap",
            value=f"Rs {total_gap/1000:.0f}K",
            delta=f"{gap_pct:+.1f}%"
        )
    
    with col4:
        gap_per_booking = total_gap / comparison_df['Bookings'].sum() if comparison_df['Bookings'].sum() > 0 else 0
        st.metric(
            label="Gap per Booking",
            value=f"Rs {gap_per_booking:,.0f}",
            delta="Extra per slot"
        )
    
    # Gap breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly gap chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=comparison_df['Month_Name'],
            y=comparison_df['Gap'],
            marker_color=comparison_df['Gap'].apply(lambda x: '#27ae60' if x > 0 else '#e74c3c'),
            text=comparison_df['Gap'].apply(lambda x: f"Rs {x/1000:.0f}K"),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Monthly Revenue Gap (Actual - Estimated)',
            xaxis_title='Month',
            yaxis_title='Gap (Rs)',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gap percentage
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=comparison_df['Month_Name'],
            y=comparison_df['Gap_%'],
            marker_color=comparison_df['Gap_%'].apply(lambda x: '#27ae60' if x > 0 else '#e74c3c'),
            text=comparison_df['Gap_%'].apply(lambda x: f"{x:+.1f}%"),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Revenue Gap as % of Estimated',
            xaxis_title='Month',
            yaxis_title='Gap (%)',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Reasons for gap
    st.markdown("#### Possible Reasons for Revenue Gap")
    
    reasons = [
        ("✅ **Dynamic Pricing**", "Actual rates may be higher than standard during peak hours or special events"),
        ("🎁 **Package Deals**", "Multi-hour bookings, tournaments, or group packages increase per-booking revenue"),
        ("🏋️ **Additional Services**", "Equipment rental, training sessions, or other value-added services"),
        ("⚽ **Premium Slots**", "Some time slots may command premium pricing beyond standard rates"),
        ("📅 **Advance Bookings**", "Early booking premiums or last-minute surcharges"),
        ("🎉 **Special Events**", "Tournament hosting or private events with higher pricing")
    ]
    
    for title, desc in reasons:
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;">
            <strong>{title}</strong><br>
            {desc}
        </div>
        """, unsafe_allow_html=True)

# Tab 4: Strategic Insights
with tab4:
    st.markdown("### 🎯 Strategic Insights & Recommendations")
    
    # Generate insights
    insights = []
    
    # 1. Revenue efficiency
    comparison_df = merged_df.dropna(subset=['Sports_Income_Total', 'Bookings'])
    if not comparison_df.empty:
        avg_rpb = (comparison_df['Sports_Income_Total'] / comparison_df['Bookings']).mean()
        
        if avg_rpb > 4500:
            insights.append({
                "title": "💰 High Revenue per Booking",
                "desc": f"Average Rs {avg_rpb:,.0f} per booking exceeds estimated Rs 3,915. Excellent pricing strategy or successful upselling.",
                "priority": "High"
            })
        elif avg_rpb < 3500:
            insights.append({
                "title": "📉 Revenue Optimization Needed",
                "desc": f"Average Rs {avg_rpb:,.0f} per booking is below estimated rates. Review pricing or reduce discounts.",
                "priority": "High"
            })
    
    # 2. Booking volume
    total_bookings = booking_df['Bookings'].sum()
    months_count = len(booking_df)
    avg_monthly_bookings = total_bookings / months_count
    
    if avg_monthly_bookings > 200:
        insights.append({
            "title": "📈 Strong Booking Volume",
            "desc": f"Average {avg_monthly_bookings:.0f} bookings per month shows healthy demand. Focus on retention and upselling.",
            "priority": "Info"
        })
    
    # 3. Financial health
    if not financial_df.empty:
        sports_margin = financial_df['Sports_Margin_%'].mean()
        
        if sports_margin > 50:
            insights.append({
                "title": "✅ Healthy Sports Margins",
                "desc": f"Average sports margin of {sports_margin:.1f}% indicates strong profitability. Excellent operational efficiency.",
                "priority": "Info"
            })
    
    # 4. Growth opportunity
    comparison_df_full = merged_df.dropna(subset=['Sports_Income_Total', 'Est_Revenue'])
    if not comparison_df_full.empty:
        total_gap = (comparison_df_full['Sports_Income_Total'] - comparison_df_full['Est_Revenue']).sum()
        
        if total_gap > 500000:
            insights.append({
                "title": "🎁 Value-Added Revenue",
                "desc": f"Rs {total_gap:,.0f} additional revenue beyond standard bookings suggests successful ancillary services or premium pricing.",
                "priority": "High"
            })
    
    # 5. Combined business view
    if not financial_df.empty:
        sports_contribution = (financial_df['Sports_Income_Total'].sum() / financial_df['Total_Income'].sum()) * 100
        
        insights.append({
            "title": "⚽ Sports is Core Business",
            "desc": f"Sports represents {sports_contribution:.1f}% of total revenue. Continue focusing on court utilization while exploring F&B growth.",
            "priority": "Info"
        })
    
    # Display insights
    for insight in insights:
        color = {"High": "#8e44ad", "Medium": "#f39c12", "Info": "#2ecc71"}[insight["priority"]]
        
        st.markdown(f"""
        <div style="background-color: {color}20; padding: 1rem; border-left: 4px solid {color}; border-radius: 0.25rem; margin: 1rem 0;">
            <h4 style="margin: 0; color: {color};">{insight['title']}</h4>
            <p style="margin: 0.5rem 0 0 0;">{insight['desc']}</p>
            <span style="font-size: 0.85rem; color: {color};"><strong>Priority: {insight['priority']}</strong></span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Strategic recommendations
    st.markdown("### 🚀 Strategic Recommendations")
    
    recommendations = [
        "**📊 Leverage High Revenue Per Booking**: Your actual revenue exceeds booking estimates. Document what drives this (packages, services, premium pricing) and scale it.",
        
        "**⚽ Maximize Court Utilization**: With strong profitability, focus on filling empty slots during off-peak hours with promotional pricing.",
        
        "**💡 Dynamic Pricing Strategy**: Implement time-based pricing tiers to maximize revenue during peak hours while filling off-peak capacity.",
        
        "**🎯 Customer Segmentation**: Analyze booking patterns to create targeted packages for corporates, regulars, and occasional players.",
        
        "**🍔 F&B Cross-Selling**: Use high court traffic to boost F&B revenue through combo deals or post-game promotions.",
        
        "**📱 Digital Booking System**: If not already in place, implement online booking with automated pricing and promotions.",
        
        "**📈 Track Customer Lifetime Value**: Link booking data to customer profiles to identify and retain high-value clients.",
        
        "**🎉 Event Hosting**: Leverage court during low-demand periods for tournaments, training camps, or corporate events at premium rates.",
        
        "**💰 Revenue Optimization**: Monitor the Rs 500/booking premium you're achieving and ensure it's sustainable and replicable.",
        
        "**📊 Integrated Dashboard**: Continue using this combined view to make data-driven decisions linking operations to financial outcomes."
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"📅 **Financial Data:** 7 months (excl. Aug) | **Booking Data:** 9 months")
with col2:
    st.markdown(f"🎯 **Combined Analysis** | **Unified Business View**")
with col3:
    st.markdown(f"🔒 **Confidential** | {datetime.now().strftime('%Y-%m-%d')}")
