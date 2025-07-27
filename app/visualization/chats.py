import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def get_log_level_distribution(df: pd.DataFrame):
    level_counts = df['level'].value_counts().reset_index()
    level_counts.columns = ['Level', 'Count']
    fig = px.bar(level_counts, x='Level', y='Count', color='Level',
                 title='Log Level Distribution', text='Count')
    fig.update_layout(xaxis_title='Log Level', yaxis_title='Count', height=400)
    return fig

def get_error_type_distribution(df: pd.DataFrame):
    error_counts = df['error_type'].value_counts().reset_index()
    error_counts.columns = ['Error Type', 'Count']
    fig = px.pie(error_counts, names='Error Type', values='Count',
                 title='Error Type Distribution', hole=0.3)
    return fig

def get_log_trend(df: pd.DataFrame):
    df_copy = df.copy()
    if 'timestamp' in df_copy.columns:
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')
        df_copy = df_copy.dropna(subset=['timestamp'])
        df_copy['minute'] = df_copy['timestamp'].dt.floor('min')
        trend = df_copy.groupby(['minute', 'level']).size().reset_index(name='count')
        fig = px.line(trend, x='minute', y='count', color='level',
                      title='Log Volume Over Time')
        fig.update_layout(xaxis_title='Timestamp (per minute)', yaxis_title='Log Count', height=400)
        return fig
    return go.Figure()

def get_component_distribution(df: pd.DataFrame):
    comp_counts = df['component'].value_counts().head(10).reset_index()
    comp_counts.columns = ['Component', 'Count']
    fig = px.bar(comp_counts, x='Component', y='Count', color='Component',
                 title='Top Components by Log Count', text='Count')
    fig.update_layout(xaxis_title='Component', yaxis_title='Count', height=400)
    return fig

def create_visualizations(df: pd.DataFrame):
  
    st.subheader("📊 Visual Analytics")
    st.plotly_chart(get_log_level_distribution(df), use_container_width=True)
    st.plotly_chart(get_error_type_distribution(df), use_container_width=True)
    st.plotly_chart(get_log_trend(df), use_container_width=True)
    st.plotly_chart(get_component_distribution(df), use_container_width=True)
