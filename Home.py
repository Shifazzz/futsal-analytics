import streamlit as st
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Infinity Sports Arena - Analytics Hub",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 2rem 0;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .icon {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .card-desc {
        text-align: center;
        color: #666;
        font-size: 1rem;
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #888;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏟️ Infinity Sports Arena</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Kawdana Outlet - Business Intelligence Dashboard</p>', unsafe_allow_html=True)

# Welcome message
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.info("""
    📊 **Welcome to your Analytics Hub!**
    
    This dashboard provides comprehensive insights into your sports facility operations and financial performance.
    Use the sidebar to navigate between different analytical views.
    """)

st.markdown("---")

# Dashboard cards
st.markdown("### 📈 Available Dashboards")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="icon">💰</div>
        <div class="card-title">Financial Dashboard</div>
        <div class="card-desc">
            Track revenue, expenses, and profitability across Sports and F&B operations.
            Analyze 7 months of financial data with detailed breakdowns and trends.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📊 View Financial Dashboard", use_container_width=True):
        st.switch_page("pages/1_💰_Financial.py")

with col2:
    st.markdown("""
    <div class="card">
        <div class="icon">⚽</div>
        <div class="card-title">Booking Analytics</div>
        <div class="card-desc">
            Monitor court utilization, occupancy rates, and booking patterns.
            Analyze 9 months of operational data with peak hour insights.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("📅 View Booking Analytics", use_container_width=True):
        st.switch_page("pages/2_⚽_Bookings.py")

with col3:
    st.markdown("""
    <div class="card">
        <div class="icon">📊</div>
        <div class="card-title">Executive Summary</div>
        <div class="card-desc">
            Get a unified view combining financial performance with operational metrics.
            Strategic insights and recommendations for business growth.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎯 View Executive Summary", use_container_width=True):
        st.switch_page("pages/3_📊_Executive.py")

st.markdown("---")

# Quick stats
st.markdown("### 📊 Quick Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📅 Data Coverage",
        value="9 Months",
        delta="Apr-Dec 2025"
    )

with col2:
    st.metric(
        label="💰 Total Revenue",
        value="Rs 7.97M",
        delta="Financial data"
    )

with col3:
    st.metric(
        label="⚽ Total Bookings",
        value="1,787",
        delta="Booking data"
    )

with col4:
    st.metric(
        label="📈 Avg Margin",
        value="42.4%",
        delta="Profitability"
    )

st.markdown("---")

# Features list
col1, col2 = st.columns(2)

with col1:
    st.markdown("### ✨ Key Features")
    st.markdown("""
    - 📊 **Multi-dimensional Analysis**: Financial + Operational metrics
    - 🔄 **Real-time Updates**: Upload new data anytime
    - 📈 **Interactive Charts**: Explore data with dynamic visualizations
    - 💡 **Auto-generated Insights**: AI-powered recommendations
    - 📱 **Mobile Responsive**: Access from any device
    - 🔒 **Secure**: Your data stays confidential
    """)

with col2:
    st.markdown("### 🎯 Business Lines Covered")
    st.markdown("""
    - ⚽ **Sports (Futsal Court)**: Court bookings and utilization
    - 🍔 **F&B Operations**: Restaurant/cafe revenue and costs
    - 💳 **Payment Methods**: Cash vs Bank analysis
    - 📅 **Time-based Insights**: Peak hours, weekday vs weekend
    - 💰 **Revenue Optimization**: Pricing and occupancy analysis
    - 📊 **Profitability Tracking**: Margins by business line
    """)

st.markdown("---")

# Instructions
with st.expander("📖 How to Use This Dashboard"):
    st.markdown("""
    ### Getting Started
    
    1. **Navigate**: Use the sidebar on the left to switch between dashboards
    2. **Filter Data**: Each dashboard has date range and filter options
    3. **Interact**: Click on charts to zoom, hover for details
    4. **Update Data**: Upload new CSV/Excel files via the file uploaders
    5. **Export**: Download charts and data tables as needed
    
    ### Dashboard Details
    
    **💰 Financial Dashboard**
    - View revenue, expenses, and profit trends
    - Compare Sports vs F&B performance
    - Analyze payment methods (Cash vs Bank)
    - Track profitability margins
    
    **⚽ Booking Analytics**
    - See court occupancy heatmaps
    - Identify peak and off-peak hours
    - Compare weekday vs weekend patterns
    - Find empty slots for promotions
    
    **📊 Executive Summary**
    - Combined view of all metrics
    - Booking vs Revenue correlation
    - Strategic insights and recommendations
    - Gap analysis and opportunities
    
    ### Need Help?
    
    - All values are in Sri Lankan Rupees (Rs)
    - Data covers April-December 2025
    - Financial data: 7 months (excluding August)
    - Booking data: 9 months (including August)
    """)

# Footer
st.markdown("""
    <div class="footer">
        <p>🏟️ <strong>Infinity Sports Arena - Kawdana Outlet</strong></p>
        <p>Business Intelligence Dashboard | Last Updated: {}</p>
        <p>🔒 Confidential - For Internal Use Only</p>
    </div>
""".format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)
