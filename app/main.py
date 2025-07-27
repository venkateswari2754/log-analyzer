import streamlit as st
import sys
import os

from core.log_analyzer import LogAnalyzer
from app.ai.groq_chain import setup_ai_chain, query_logs, get_similar_logs
from app.visualization.chats import create_visualizations
from app.ai.embeddings import get_embeddings_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.utils import validate_file

from dotenv import load_dotenv
load_dotenv()

# Streamlit page config
st.set_page_config(
    page_title="Log Analyzer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Log Analyzer")
st.markdown("### Intelligent Log Analysis")

# Initialize analyzer in session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = LogAnalyzer()

# Sidebar: Configuration
with st.sidebar:
    st.header("Configuration")

    # Load API key from .env or input
    groq_key_from_env = os.getenv("GROQ_KEY")
    if groq_key_from_env:
        st.session_state.groq_key = groq_key_from_env
    else:
        groq_key = st.text_input("Groq API Key", type="password")
        if groq_key:
            st.session_state.groq_key = groq_key
        else:
            st.warning("⚠️ Add GROQ_KEY to .env or enter manually")

    # Groq model selection
    model_options = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
    selected_model = st.selectbox("Select Groq Model", model_options)
    st.session_state.selected_model = selected_model

    st.markdown("---")
    st.markdown("#### About")
    st.markdown("- 📁 Upload log file\n- 🤖 AI analysis\n- 📊 Visualizations\n- 🔍 Semantic search")

# Layout: Log Upload and Summary
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("📁 Upload Log File")
    uploaded_file = st.file_uploader("Choose a log file", type=["log"])

    if uploaded_file:
        is_valid, message = validate_file(uploaded_file)
        if is_valid:
            st.success(message)
            with st.spinner("Processing log file..."):
                if st.session_state.analyzer.process_log_file(uploaded_file):
                    st.success("✅ Log file processed!")

                    if hasattr(st.session_state, 'groq_key'):
                        with st.spinner("Setting up Groq AI..."):
                            setup_success = setup_ai_chain(
                                analyzer=st.session_state.analyzer,
                                groq_api_key=st.session_state.groq_key,
                                model_name=st.session_state.selected_model
                            )
                            if setup_success:
                                st.success("🤖 Groq AI initialized!")
        else:
            st.error(message)

# Right side: Top Issues and Visualizations
with col2:
    analyzer = st.session_state.analyzer
    if analyzer.log_data is not None:
        st.subheader("🔥 Top Recurring Issues")
        top_issues = analyzer.get_top_issues()
        for i, issue in enumerate(top_issues, 1):
            st.markdown(f"**{i}. {issue['emoji']} {issue['issue']}** - {issue['count']} times ({issue['severity']})")

        st.subheader("📊 Visual Analytics")
        
        # Safe initialization
        fig_levels = fig_timeline = fig_errors = None
        
        result = create_visualizations(analyzer.log_data)
        if result is not None:
            fig_levels, fig_timeline, fig_errors = result

        col_v1, col_v2 = st.columns(2)
        with col_v1:
            if fig_levels:
                st.plotly_chart(fig_levels, use_container_width=True)
        with col_v2:
            if fig_errors:
                st.plotly_chart(fig_errors, use_container_width=True)
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True)

# AI Log Analysis Tabs
if uploaded_file and analyzer.log_data is not None:
    st.markdown("---")
    st.subheader("🤖 Log Analysis")
    tab1, tab2 = st.tabs(["🔍 Ask Questions", "🔎 Semantic Search"])

    # Tab 1: Ask Questions
    with tab1:
        sample_questions = [
            "List top 3 issues which are reoccurring",
            "What are the most common error patterns?",
            "What database errors occurred?",
            "Are there memory-related issues?",
        ]
        selected = st.selectbox("Prompt:", ["Custom Query"] + sample_questions)
        query = st.text_input("Your Question:", value="" if selected == "Custom Query" else selected)

        if st.button("🔍 Analyze"):
            with st.spinner("Analyzing logs..."):
                if hasattr(st.session_state, 'groq_key'):
                    response = query_logs(analyzer, query)
                    st.markdown("#### AI Response:")
                    st.markdown(response)

    # Tab 2: Semantic Search
    with tab2:
        search_query = st.text_input("Search similar logs")
        result_count = st.slider("Results", 1, 10, 5)
        if st.button("🔎 Search"):
            with st.spinner("Searching..."):
                results = get_similar_logs(analyzer, search_query, result_count)
                for i, log in enumerate(results, 1):
                    with st.expander(f"📝 Log {i}"):
                        st.write("**Content:**", log['content'])
                        st.write("**Timestamp:**", log['metadata'].get('timestamp'))
                        st.write("**Component:**", log['metadata'].get('component'))
                        st.write("**Error Type:**", log['metadata'].get('error_type'))
