import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Infinity Sports Arena - Financial Analytics",
    page_icon="💰",
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
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
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
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Data loading function
@st.cache_data
def load_data(uploaded_file=None):
    """Load financial data from uploaded file or use sample"""
    
    def clean_numeric(val):
        """Clean numeric values"""
        if pd.isna(val) or val == '':
            return 0
        if isinstance(val, str):
            val = val.replace(',', '').replace('"', '').strip()
            try:
                return float(val)
            except:
                return 0
        return float(val)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    else:
        # Use the provided data
        try:
            df = pd.read_csv('financial_data.csv')
        except:
            st.error("Please upload your financial data CSV file")
            return None
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Numeric columns to clean
    numeric_cols = ['Sports_Income_Cash', 'Sports_Income_Bank', 'Sports_Income_Total',
                    'FB_Income_Cash', 'FB_Income_Bank', 'FB_Income_Total',
                    'Total_Income_Cash', 'Total_Income_Bank', 'Total_Income',
                    'Sports_Exp_Maintenance', 'Sports_Exp_Salary', 'Sports_Exp_Marketing',
                    'Sports_Exp_Utilities', 'Sports_Exp_Stationery', 'Sports_Exp_Statutory',
                    'Sports_Exp_TeaMeals', 'Sports_Exp_Purchases', 'Sports_Exp_Other',
                    'Sports_Exp_Total', 'FB_Exp_Purchases', 'FB_Exp_Stock_Adjustment',
                    'FB_Exp_Other', 'FB_Exp_Total', 'Total_Expenses',
                    'Sports_Surplus', 'FB_Surplus', 'Total_Surplus',
                    'Sports_Margin_%', 'FB_Margin_%', 'Overall_Margin_%']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
    
    # Convert date column
    df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
    
    # Filter out empty rows
    df = df[df['Total_Income'] > 0].reset_index(drop=True)
    
    # Calculate additional metrics (columns AG-AL equivalent)
    df['MoM_Revenue_Growth_%'] = df['Total_Income'].pct_change() * 100
    df['MoM_Profit_Growth_%'] = df['Total_Surplus'].pct_change() * 100
    df['Cash_Percentage'] = (df['Total_Income_Cash'] / df['Total_Income']) * 100
    df['Sports_Revenue_%'] = (df['Sports_Income_Total'] / df['Total_Income']) * 100
    df['FB_Revenue_%'] = (df['FB_Income_Total'] / df['Total_Income']) * 100
    
    return df

# Sidebar
with st.sidebar:
    st.markdown("### 💰 Infinity Sports Arena")
    st.markdown("**Financial Analytics Dashboard**")
    st.markdown("---")
    
    # File upload
    st.markdown("#### 📁 Data Source")
    uploaded_file = st.file_uploader(
        "Upload Financial Data CSV",
        type=['csv'],
        help="Upload your financial data CSV file"
    )
    
    st.markdown("---")
    
    # Date filter
    st.markdown("#### 📅 Filters")

# Load data
df = load_data(uploaded_file)

if df is None or len(df) == 0:
    st.error("No data available. Please upload your financial data CSV file.")
    st.stop()

# Date range filter
date_min = df['Month'].min().date()
date_max = df['Month'].max().date()

with st.sidebar:
    date_range = st.date_input(
        "Date Range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max
    )
    
    if len(date_range) == 2:
        df_filtered = df[(df['Month'].dt.date >= date_range[0]) & (df['Month'].dt.date <= date_range[1])]
    else:
        df_filtered = df
    
    # Business line filter
    business_filter = st.selectbox(
        "Business Line",
        ['Combined', 'Sports Only', 'F&B Only']
    )
    
    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.markdown("""
    Financial analytics for Infinity Sports Arena covering:
    - ⚽ Sports (Court Bookings)
    - 🍔 F&B (Food & Beverage)
    
    **Confidential Data** 🔒
    """)

# Main content
st.markdown('<h1 class="main-header">💰 Infinity Sports Arena - Financial Analytics</h1>', unsafe_allow_html=True)

st.markdown(f"**Period:** {df_filtered['Month_Name'].iloc[0]} to {df_filtered['Month_Name'].iloc[-1]} ({len(df_filtered)} months)")

# Key Metrics
st.markdown("### 📊 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

total_revenue = df_filtered['Total_Income'].sum()
total_expenses = df_filtered['Total_Expenses'].sum()
total_profit = df_filtered['Total_Surplus'].sum()
avg_margin = df_filtered['Overall_Margin_%'].mean()
best_month = df_filtered.loc[df_filtered['Total_Income'].idxmax(), 'Month_Name']

with col1:
    st.metric(
        label="Total Revenue",
        value=f"Rs {total_revenue:,.0f}",
        delta=f"Avg Rs {total_revenue/len(df_filtered):,.0f}/mo"
    )

with col2:
    st.metric(
        label="Total Expenses",
        value=f"Rs {total_expenses:,.0f}",
        delta=f"{(total_expenses/total_revenue*100):.1f}% of revenue"
    )

with col3:
    st.metric(
        label="Total Profit",
        value=f"Rs {total_profit:,.0f}",
        delta=f"Avg Rs {total_profit/len(df_filtered):,.0f}/mo"
    )

with col4:
    st.metric(
        label="Profit Margin",
        value=f"{avg_margin:.1f}%",
        delta="Average across period"
    )

with col5:
    st.metric(
        label="Best Month",
        value=best_month,
        delta=f"Rs {df_filtered['Total_Income'].max():,.0f}"
    )

st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Overview",
    "⚽ Sports Analysis",
    "🍔 F&B Analysis",
    "💰 Revenue Breakdown",
    "💸 Expense Analysis",
    "📊 Profitability",
    "🎯 Insights"
])

# Tab 1: Overview
with tab1:
    st.markdown("### Business Performance Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly revenue trend
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df_filtered['Month_Name'],
            y=df_filtered['Sports_Income_Total'],
            name='Sports Revenue',
            marker_color='#2ecc71'
        ))
        
        fig.add_trace(go.Bar(
            x=df_filtered['Month_Name'],
            y=df_filtered['FB_Income_Total'],
            name='F&B Revenue',
            marker_color='#e74c3c'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_filtered['Month_Name'],
            y=df_filtered['Total_Income'],
            name='Total Revenue',
            mode='lines+markers',
            line=dict(color='#3498db', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Monthly Revenue Breakdown',
            xaxis_title='Month',
            yaxis_title='Revenue (Rs )',
            yaxis2=dict(
                title='Total Revenue (Rs )',
                overlaying='y',
                side='right'
            ),
            barmode='stack',
            hovermode='x unified',
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Profit trend
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df_filtered['Month_Name'],
            y=df_filtered['Total_Surplus'],
            name='Total Profit',
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#27ae60', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df_filtered['Month_Name'],
            y=df_filtered['Overall_Margin_%'],
            name='Profit Margin %',
            mode='lines+markers',
            line=dict(color='#f39c12', width=2, dash='dash'),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Monthly Profit & Margin Trend',
            xaxis_title='Month',
            yaxis_title='Profit (Rs )',
            yaxis2=dict(
                title='Margin (%)',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Business mix
    st.markdown("#### Business Line Contribution")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Revenue split
        fig = go.Figure(data=[go.Pie(
            labels=['Sports', 'F&B'],
            values=[df_filtered['Sports_Income_Total'].sum(), df_filtered['FB_Income_Total'].sum()],
            hole=0.4,
            marker=dict(colors=['#2ecc71', '#e74c3c']),
            textinfo='label+percent',
            textfont_size=14
        )])
        fig.update_layout(
            title='Revenue Split',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Profit split
        sports_profit = df_filtered['Sports_Surplus'].sum()
        fb_profit = df_filtered['FB_Surplus'].sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Sports Profit', 'F&B Profit'],
            values=[sports_profit, fb_profit],
            hole=0.4,
            marker=dict(colors=['#27ae60', '#c0392b']),
            textinfo='label+percent',
            textfont_size=14
        )])
        fig.update_layout(
            title='Profit Contribution',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Payment methods
        fig = go.Figure(data=[go.Pie(
            labels=['Cash', 'Bank'],
            values=[df_filtered['Total_Income_Cash'].sum(), df_filtered['Total_Income_Bank'].sum()],
            hole=0.4,
            marker=dict(colors=['#16a085', '#2980b9']),
            textinfo='label+percent',
            textfont_size=14
        )])
        fig.update_layout(
            title='Payment Methods',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Sports Analysis
with tab2:
    st.markdown("### ⚽ Sports Business Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    sports_revenue = df_filtered['Sports_Income_Total'].sum()
    sports_expenses = df_filtered['Sports_Exp_Total'].sum()
    sports_profit = df_filtered['Sports_Surplus'].sum()
    
    with col1:
        st.metric("Sports Revenue", f"Rs {sports_revenue:,.0f}", 
                 delta=f"{(sports_revenue/total_revenue*100):.1f}% of total")
    with col2:
        st.metric("Sports Expenses", f"Rs {sports_expenses:,.0f}",
                 delta=f"{(sports_expenses/sports_revenue*100):.1f}% of revenue")
    with col3:
        st.metric("Sports Profit", f"Rs {sports_profit:,.0f}",
                 delta=f"{df_filtered['Sports_Margin_%'].mean():.1f}% margin")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sports revenue trend
        fig = px.line(
            df_filtered,
            x='Month_Name',
            y='Sports_Income_Total',
            title='Sports Revenue Trend',
            markers=True
        )
        fig.update_traces(line_color='#2ecc71', line_width=3)
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Revenue (Rs )',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sports margin trend
        fig = px.line(
            df_filtered,
            x='Month_Name',
            y='Sports_Margin_%',
            title='Sports Profit Margin Trend',
            markers=True
        )
        fig.update_traces(line_color='#27ae60', line_width=3)
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Margin (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sports expense breakdown
    st.markdown("#### Sports Expense Breakdown")
    
    expense_categories = {
        'Maintenance': df_filtered['Sports_Exp_Maintenance'].sum(),
        'Salary': df_filtered['Sports_Exp_Salary'].sum(),
        'Marketing': df_filtered['Sports_Exp_Marketing'].sum(),
        'Utilities': df_filtered['Sports_Exp_Utilities'].sum(),
        'Stationery': df_filtered['Sports_Exp_Stationery'].sum(),
        'Statutory': df_filtered['Sports_Exp_Statutory'].sum(),
        'Tea & Meals': df_filtered['Sports_Exp_TeaMeals'].sum(),
        'Purchases': df_filtered['Sports_Exp_Purchases'].sum(),
        'Other': df_filtered['Sports_Exp_Other'].sum()
    }
    
    # Remove zero categories
    expense_categories = {k: v for k, v in expense_categories.items() if v > 0}
    
    fig = go.Figure(data=[go.Bar(
        x=list(expense_categories.keys()),
        y=list(expense_categories.values()),
        marker_color='#3498db',
        text=[f"Rs {v:,.0f}" for v in expense_categories.values()],
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Sports Expense Categories',
        xaxis_title='Category',
        yaxis_title='Amount (Rs )',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: F&B Analysis
with tab3:
    st.markdown("### 🍔 F&B Business Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    fb_revenue = df_filtered['FB_Income_Total'].sum()
    fb_expenses = df_filtered['FB_Exp_Total'].sum()
    fb_profit = df_filtered['FB_Surplus'].sum()
    
    with col1:
        st.metric("F&B Revenue", f"Rs {fb_revenue:,.0f}",
                 delta=f"{(fb_revenue/total_revenue*100):.1f}% of total")
    with col2:
        st.metric("F&B Expenses", f"Rs {fb_expenses:,.0f}",
                 delta=f"{(fb_expenses/fb_revenue*100):.1f}% of revenue")
    with col3:
        avg_fb_margin = df_filtered[df_filtered['FB_Income_Total'] > 0]['FB_Margin_%'].mean()
        st.metric("F&B Profit", f"Rs {fb_profit:,.0f}",
                 delta=f"{avg_fb_margin:.1f}% margin")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # F&B revenue trend
        fig = px.line(
            df_filtered[df_filtered['FB_Income_Total'] > 0],
            x='Month_Name',
            y='FB_Income_Total',
            title='F&B Revenue Trend',
            markers=True
        )
        fig.update_traces(line_color='#e74c3c', line_width=3)
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Revenue (Rs )',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # F&B margin trend
        fig = px.line(
            df_filtered[df_filtered['FB_Income_Total'] > 0],
            x='Month_Name',
            y='FB_Margin_%',
            title='F&B Profit Margin Trend',
            markers=True
        )
        fig.update_traces(line_color='#c0392b', line_width=3)
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Margin (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # F&B COGS analysis
    st.markdown("#### F&B Cost Structure")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fb_cogs = df_filtered['FB_Exp_Purchases'].sum()
        fb_stock_adj = df_filtered['FB_Exp_Stock_Adjustment'].sum()
        fb_other = df_filtered['FB_Exp_Other'].sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Purchases (COGS)', 'Stock Adjustment', 'Other'],
            values=[fb_cogs, abs(fb_stock_adj), fb_other],
            marker=dict(colors=['#e74c3c', '#f39c12', '#95a5a6'])
        )])
        fig.update_layout(
            title='F&B Expense Breakdown',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # F&B profitability by month
        fb_data = df_filtered[df_filtered['FB_Income_Total'] > 0].copy()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fb_data['Month_Name'],
            y=fb_data['FB_Income_Total'],
            name='Revenue',
            marker_color='lightcoral'
        ))
        fig.add_trace(go.Bar(
            x=fb_data['Month_Name'],
            y=fb_data['FB_Exp_Total'],
            name='Expenses',
            marker_color='indianred'
        ))
        
        fig.update_layout(
            title='F&B Revenue vs Expenses',
            xaxis_title='Month',
            yaxis_title='Amount (Rs )',
            barmode='group',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 4: Revenue Breakdown
with tab4:
    st.markdown("### 💰 Revenue Analysis")
    
    # Monthly comparison table
    st.markdown("#### Monthly Revenue Breakdown")
    
    revenue_summary = df_filtered[['Month_Name', 'Sports_Income_Total', 'FB_Income_Total', 
                                   'Total_Income', 'Sports_Revenue_%', 'FB_Revenue_%']].copy()
    revenue_summary.columns = ['Month', 'Sports (Rs )', 'F&B (Rs )', 'Total (Rs )', 'Sports %', 'F&B %']
    
    # Format numbers
    for col in ['Sports (Rs )', 'F&B (Rs )', 'Total (Rs )']:
        revenue_summary[col] = revenue_summary[col].apply(lambda x: f"Rs {x:,.0f}")
    for col in ['Sports %', 'F&B %']:
        revenue_summary[col] = revenue_summary[col].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(revenue_summary, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment method trends
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered['Month_Name'],
            y=df_filtered['Total_Income_Cash'],
            name='Cash',
            mode='lines+markers',
            stackgroup='one',
            fillcolor='rgba(22, 160, 133, 0.5)'
        ))
        fig.add_trace(go.Scatter(
            x=df_filtered['Month_Name'],
            y=df_filtered['Total_Income_Bank'],
            name='Bank',
            mode='lines+markers',
            stackgroup='one',
            fillcolor='rgba(41, 128, 185, 0.5)'
        ))
        
        fig.update_layout(
            title='Revenue by Payment Method',
            xaxis_title='Month',
            yaxis_title='Revenue (Rs )',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cash percentage trend
        fig = px.line(
            df_filtered,
            x='Month_Name',
            y='Cash_Percentage',
            title='Cash as % of Total Revenue',
            markers=True
        )
        fig.update_traces(line_color='#16a085', line_width=3)
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Cash %',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Growth analysis
    st.markdown("#### Growth Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # MoM revenue growth
        growth_data = df_filtered[df_filtered['MoM_Revenue_Growth_%'].notna()].copy()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=growth_data['Month_Name'],
            y=growth_data['MoM_Revenue_Growth_%'],
            marker_color=growth_data['MoM_Revenue_Growth_%'].apply(
                lambda x: '#27ae60' if x > 0 else '#e74c3c'
            ),
            text=growth_data['MoM_Revenue_Growth_%'].apply(lambda x: f"{x:+.1f}%"),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Month-over-Month Revenue Growth',
            xaxis_title='Month',
            yaxis_title='Growth (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # MoM profit growth
        profit_growth = df_filtered[df_filtered['MoM_Profit_Growth_%'].notna()].copy()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=profit_growth['Month_Name'],
            y=profit_growth['MoM_Profit_Growth_%'],
            marker_color=profit_growth['MoM_Profit_Growth_%'].apply(
                lambda x: '#27ae60' if x > 0 else '#e74c3c'
            ),
            text=profit_growth['MoM_Profit_Growth_%'].apply(lambda x: f"{x:+.1f}%"),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Month-over-Month Profit Growth',
            xaxis_title='Month',
            yaxis_title='Growth (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 5: Expense Analysis
with tab5:
    st.markdown("### 💸 Expense Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Expense trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_filtered['Month_Name'],
            y=df_filtered['Sports_Exp_Total'],
            name='Sports Expenses',
            mode='lines+markers',
            line=dict(color='#e67e22', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_filtered['Month_Name'],
            y=df_filtered['FB_Exp_Total'],
            name='F&B Expenses',
            mode='lines+markers',
            line=dict(color='#e74c3c', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df_filtered['Month_Name'],
            y=df_filtered['Total_Expenses'],
            name='Total Expenses',
            mode='lines+markers',
            line=dict(color='#c0392b', width=3)
        ))
        
        fig.update_layout(
            title='Monthly Expense Trends',
            xaxis_title='Month',
            yaxis_title='Expenses (Rs )',
            hovermode='x unified',
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Expense as % of revenue
        df_filtered_copy = df_filtered.copy()
        df_filtered_copy['Expense_Ratio'] = (df_filtered_copy['Total_Expenses'] / df_filtered_copy['Total_Income']) * 100
        
        fig = px.line(
            df_filtered_copy,
            x='Month_Name',
            y='Expense_Ratio',
            title='Expenses as % of Revenue',
            markers=True
        )
        fig.update_traces(line_color='#e74c3c', line_width=3)
        fig.add_hline(y=50, line_dash="dash", line_color="red", 
                     annotation_text="50% threshold")
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Expense Ratio (%)',
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top expense categories
    st.markdown("#### Top Expense Categories (Combined)")
    
    all_expenses = {
        'Sports Salary': df_filtered['Sports_Exp_Salary'].sum(),
        'Sports Utilities': df_filtered['Sports_Exp_Utilities'].sum(),
        'Sports Purchases': df_filtered['Sports_Exp_Purchases'].sum(),
        'Sports Maintenance': df_filtered['Sports_Exp_Maintenance'].sum(),
        'F&B Purchases': df_filtered['FB_Exp_Purchases'].sum(),
        'Sports Marketing': df_filtered['Sports_Exp_Marketing'].sum(),
        'Sports Tea & Meals': df_filtered['Sports_Exp_TeaMeals'].sum(),
        'Sports Statutory': df_filtered['Sports_Exp_Statutory'].sum(),
    }
    
    # Sort and get top 10
    all_expenses = dict(sorted(all_expenses.items(), key=lambda x: x[1], reverse=True)[:10])
    
    fig = go.Figure(data=[go.Bar(
        y=list(all_expenses.keys()),
        x=list(all_expenses.values()),
        orientation='h',
        marker_color='#e67e22',
        text=[f"Rs {v:,.0f}" for v in all_expenses.values()],
        textposition='auto'
    )])
    
    fig.update_layout(
        title='Top 10 Expense Categories',
        xaxis_title='Amount (Rs )',
        yaxis_title='Category',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 6: Profitability
with tab6:
    st.markdown("### 📊 Profitability Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Waterfall chart for total period
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Total Revenue", "Total Expenses", "Net Profit"],
            y=[total_revenue, -total_expenses, total_profit],
            text=[f"Rs {total_revenue:,.0f}", f"-Rs {total_expenses:,.0f}", f"Rs {total_profit:,.0f}"],
            textposition="auto",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title="Profit Waterfall (Total Period)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Margin comparison
        avg_sports_margin = df_filtered['Sports_Margin_%'].mean()
        avg_fb_margin = df_filtered[df_filtered['FB_Income_Total'] > 0]['FB_Margin_%'].mean()
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Sports', 'F&B', 'Overall'],
                y=[avg_sports_margin, avg_fb_margin, avg_margin],
                marker_color=['#2ecc71', '#e74c3c', '#3498db'],
                text=[f"{avg_sports_margin:.1f}%", f"{avg_fb_margin:.1f}%", f"{avg_margin:.1f}%"],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Average Profit Margins',
            yaxis_title='Margin (%)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Best vs worst month
        best_month_idx = df_filtered['Total_Surplus'].idxmax()
        worst_month_idx = df_filtered['Total_Surplus'].idxmin()
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Best Month', 'Worst Month', 'Average'],
                y=[
                    df_filtered.loc[best_month_idx, 'Total_Surplus'],
                    df_filtered.loc[worst_month_idx, 'Total_Surplus'],
                    df_filtered['Total_Surplus'].mean()
                ],
                marker_color=['#27ae60', '#e74c3c', '#f39c12'],
                text=[
                    f"Rs {df_filtered.loc[best_month_idx, 'Total_Surplus']:,.0f}",
                    f"Rs {df_filtered.loc[worst_month_idx, 'Total_Surplus']:,.0f}",
                    f"Rs {df_filtered['Total_Surplus'].mean():,.0f}"
                ],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Profit Comparison',
            yaxis_title='Profit (Rs )',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Profitability heatmap
    st.markdown("#### Monthly Profitability Matrix")
    
    profitability_data = df_filtered[['Month_Name', 'Sports_Surplus', 'FB_Surplus', 'Total_Surplus']].copy()
    profitability_data.columns = ['Month', 'Sports Profit', 'F&B Profit', 'Total Profit']
    
    for col in ['Sports Profit', 'F&B Profit', 'Total Profit']:
        profitability_data[col] = profitability_data[col].apply(lambda x: f"Rs {x:,.0f}")
    
    st.dataframe(profitability_data, use_container_width=True, hide_index=True)

# Tab 7: Insights
with tab7:
    st.markdown("### 🎯 Automated Insights & Recommendations")
    
    # Generate insights
    insights = []
    
    # 1. Revenue mix
    sports_pct = (df_filtered['Sports_Income_Total'].sum() / total_revenue) * 100
    fb_pct = (df_filtered['FB_Income_Total'].sum() / total_revenue) * 100
    insights.append({
        "title": "💰 Revenue Mix",
        "desc": f"Sports contributes **{sports_pct:.1f}%** (Rs {df_filtered['Sports_Income_Total'].sum():,.0f}) and F&B contributes **{fb_pct:.1f}%** (Rs {df_filtered['FB_Income_Total'].sum():,.0f}) of total revenue.",
        "priority": "Info"
    })
    
    # 2. Best performing business
    if avg_sports_margin > avg_fb_margin:
        insights.append({
            "title": "⚽ Sports is More Profitable",
            "desc": f"Sports has **{avg_sports_margin:.1f}%** margin vs F&B's **{avg_fb_margin:.1f}%** margin. Focus on maximizing court bookings.",
            "priority": "High"
        })
    
    # 3. Payment preference
    cash_pct = df_filtered['Cash_Percentage'].mean()
    if cash_pct > 65:
        insights.append({
            "title": "💵 High Cash Dependency",
            "desc": f"**{cash_pct:.1f}%** of revenue is cash. Consider promoting digital payments for better tracking and convenience.",
            "priority": "Medium"
        })
    
    # 4. Growth trend
    recent_growth = df_filtered['MoM_Revenue_Growth_%'].iloc[-1]
    if not pd.isna(recent_growth):
        if recent_growth < -10:
            insights.append({
                "title": "📉 Revenue Declining",
                "desc": f"Latest month shows **{recent_growth:.1f}%** decline. Investigate seasonality or competitive factors.",
                "priority": "High"
            })
        elif recent_growth > 20:
            insights.append({
                "title": "📈 Strong Growth",
                "desc": f"Latest month shows **{recent_growth:+.1f}%** growth! Identify success factors to replicate.",
                "priority": "High"
            })
    
    # 5. F&B performance
    if avg_fb_margin < 20 and fb_revenue > 0:
        insights.append({
            "title": "🍔 F&B Margins Low",
            "desc": f"F&B margin is **{avg_fb_margin:.1f}%**. Review pricing, reduce waste, or negotiate better supplier rates.",
            "priority": "High"
        })
    
    # 6. Best month analysis
    best_month_name = df_filtered.loc[df_filtered['Total_Income'].idxmax(), 'Month_Name']
    best_month_revenue = df_filtered['Total_Income'].max()
    insights.append({
        "title": f"🏆 Best Month: {best_month_name}",
        "desc": f"Generated **Rs {best_month_revenue:,.0f}** in revenue. Analyze what made this month successful.",
        "priority": "Info"
    })
    
    # 7. Expense ratio
    avg_expense_ratio = (df_filtered['Total_Expenses'].sum() / df_filtered['Total_Income'].sum()) * 100
    if avg_expense_ratio > 60:
        insights.append({
            "title": "💸 High Expense Ratio",
            "desc": f"Expenses are **{avg_expense_ratio:.1f}%** of revenue. Look for cost optimization opportunities.",
            "priority": "Medium"
        })
    
    # Display insights
    for insight in insights:
        color = {"High": "#e74c3c", "Medium": "#f39c12", "Info": "#3498db"}[insight["priority"]]
        
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
    
    # Sports optimization
    if sports_pct > 80:
        recommendations.append("**Maximize Sports Revenue**: Since sports is 85%+ of business, focus on increasing court utilization during off-peak hours with promotions.")
    
    # F&B improvement
    if avg_fb_margin < 30:
        recommendations.append("**Improve F&B Margins**: Current F&B margins are low. Consider menu engineering, waste reduction, or price optimization.")
    
    # Digital payments
    if cash_pct > 60:
        recommendations.append("**Promote Digital Payments**: Offer small discounts for digital payments to improve tracking and reduce cash handling risks.")
    
    # Seasonal planning
    recommendations.append("**Seasonal Strategy**: Identify peak months (like July) and prepare inventory/staffing accordingly. Offer promotions during slower months.")
    
    # Cost control
    if avg_expense_ratio > 55:
        recommendations.append("**Cost Optimization**: Review top expense categories (especially salaries and utilities) for potential savings without compromising quality.")
    
    # F&B expansion
    if fb_pct < 20 and avg_fb_margin > 25:
        recommendations.append("**Consider F&B Expansion**: F&B is profitable but underutilized. Consider extended hours or enhanced menu to capture more revenue.")
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"📅 **Period:** {df_filtered['Month_Name'].iloc[0]} to {df_filtered['Month_Name'].iloc[-1]}")
with col2:
    st.markdown(f"📊 **Total Months:** {len(df_filtered)}")
with col3:
    st.markdown(f"🔒 **Confidential** | Last Updated: {datetime.now().strftime('%Y-%m-%d')}")