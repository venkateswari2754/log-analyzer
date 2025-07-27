import streamlit as st
import pandas as pd
import re
import json
import os
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.schema import Document
import tempfile
import hashlib
from dotenv import load_dotenv
import pickle
from pathlib import Path
import threading
import warnings
warnings.filterwarnings('ignore')
import logging
# Suppress warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger().setLevel(logging.ERROR)

# Load environment variables from .env file
load_dotenv()

# Cache embeddings model globally to avoid reloading
@st.cache_resource
def get_embeddings_model():
    """Cache the embeddings model to avoid reloading"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

class LogAnalyzer:
    """
    Optimized Industry-level Log Analyzer with Groq AI capabilities
    """
    
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.log_data = None
        self.parsed_logs = []
        self.embeddings = None
        self.vector_store_path = "vector_store"
        self.log_hash = None
        self.error_patterns = {
            'ERROR': r'ERROR|error|Error',
            'EXCEPTION': r'Exception|exception|EXCEPTION',
            'FATAL': r'FATAL|fatal|Fatal',
            'WARNING': r'WARNING|warning|Warning|WARN',
            'TIMEOUT': r'timeout|Timeout|TIMEOUT|timed out',
            'CONNECTION': r'connection|Connection|CONNECTION.*failed|refused',
            'MEMORY': r'OutOfMemoryError|Memory|memory.*leak|heap',
            'DATABASE': r'SQLException|database|Database.*error|DB.*error',
            'NETWORK': r'network|Network.*error|socket.*error|connection.*reset',
            'AUTHENTICATION': r'authentication|Authentication.*failed|login.*failed'
        }
    
    def get_file_hash(self, file_content: str) -> str:
        """Generate hash of file content to check if processing is needed"""
        return hashlib.md5(file_content.encode()).hexdigest()
    
    def load_cached_data(self, file_hash: str) -> bool:
        """Load cached processed data if available"""
        try:
            cache_file = f"cache_{file_hash}.pkl"
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.parsed_logs = cached_data['parsed_logs']
                    self.log_data = cached_data['log_data']
                    self.log_hash = file_hash
                st.success("📂 Loaded cached processed data!")
                return True
        except Exception:
            # Silently handle cache loading errors
            pass
        return False
    
    def save_cached_data(self, file_hash: str):
        """Save processed data to cache"""
        try:
            cache_file = f"cache_{file_hash}.pkl"
            cached_data = {
                'parsed_logs': self.parsed_logs,
                'log_data': self.log_data
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception:
            # Silently handle cache saving errors
            pass
    
    def load_vector_store(self, file_hash: str) -> bool:
        """Load existing vector store if available"""
        try:
            vector_path = f"{self.vector_store_path}_{file_hash}"
            if os.path.exists(vector_path):
                self.embeddings = get_embeddings_model()
                # Add allow_dangerous_deserialization=True to suppress the warning
                self.vector_store = FAISS.load_local(
                    vector_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                st.success("⚡ Loaded existing vector store!")
                return True
        except Exception:
            # Silently handle vector store loading errors
            pass
        return False
    
    def save_vector_store(self, file_hash: str):
        """Save vector store for future use"""
        try:
            vector_path = f"{self.vector_store_path}_{file_hash}"
            self.vector_store.save_local(vector_path)
        except Exception:
            # Silently handle vector store saving errors
            pass
    
    def validate_file(self, uploaded_file) -> Tuple[bool, str]:
        """
        Validate uploaded file format and size
        """
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file extension
        if not uploaded_file.name.endswith('.log'):
            return False, "Upload only valid file format (.log files only)"
        
        # Check file size (100MB limit)
        if uploaded_file.size > 100 * 1024 * 1024:  # 100MB in bytes
            return False, "File size exceeds 100MB limit"
        
        return True, "File validation successful"
    
    def parse_log_line(self, line: str) -> Dict[str, Any]:
        """
        Parse individual log line to extract structured information
        """
        # Common log patterns
        timestamp_pattern = r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})'
        level_pattern = r'(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|TRACE)'
        
        parsed = {
            'raw_line': line.strip(),
            'timestamp': None,
            'level': 'UNKNOWN',
            'message': line.strip(),
            'component': None,
            'error_type': None
        }
        
        # Extract timestamp
        timestamp_match = re.search(timestamp_pattern, line)
        if timestamp_match:
            parsed['timestamp'] = timestamp_match.group(1)
        
        # Extract log level
        level_match = re.search(level_pattern, line, re.IGNORECASE)
        if level_match:
            parsed['level'] = level_match.group(1).upper()
        
        # Extract component/class name
        component_pattern = r'\[([^\]]+)\]|\b([A-Z][a-zA-Z]*(?:\.[A-Z][a-zA-Z]*)*)\b'
        component_match = re.search(component_pattern, line)
        if component_match:
            parsed['component'] = component_match.group(1) or component_match.group(2)
        
        # Identify error types
        for error_type, pattern in self.error_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                parsed['error_type'] = error_type
                break
        
        return parsed
    
    def process_log_file(self, uploaded_file) -> bool:
        """
        Optimized process and analyze the uploaded log file
        """
        try:
            # Read file content
            content = uploaded_file.read().decode('utf-8')
            file_hash = self.get_file_hash(content)
            
            # Check if we have cached data for this file
            if self.load_cached_data(file_hash):
                return True
            
            # Show progress bar for processing
            lines = content.split('\n')
            total_lines = len([line for line in lines if line.strip()])
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Parse each line with progress tracking
            self.parsed_logs = []
            processed = 0
            
            for i, line in enumerate(lines):
                if line.strip():  # Skip empty lines
                    parsed = self.parse_log_line(line)
                    self.parsed_logs.append(parsed)
                    processed += 1
                    
                    # Update progress every 100 lines
                    if processed % 100 == 0 or processed == total_lines:
                        progress = processed / total_lines
                        progress_bar.progress(progress)
                        status_text.text(f'Processing logs: {processed}/{total_lines}')
            
            # Create DataFrame for analysis
            self.log_data = pd.DataFrame(self.parsed_logs)
            self.log_hash = file_hash
            
            # Save to cache
            self.save_cached_data(file_hash)
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return True
            
        except Exception as e:
            st.error(f"Error processing log file: {str(e)}")
            return False
    
    def get_top_issues(self, top_n: int = 3) -> List[Dict]:
        """
        Identify top recurring issues with emoji indicators
        """
        if self.log_data is None:
            return []
        
        # Filter error-level logs
        error_logs = self.log_data[
            self.log_data['level'].isin(['ERROR', 'FATAL', 'WARNING', 'WARN'])
        ]
        
        # Count issues by error type and message patterns
        issue_counter = Counter()
        
        for _, log in error_logs.iterrows():
            if log['error_type']:
                issue_counter[log['error_type']] += 1
            else:
                # Extract key terms from message for generic grouping
                message_words = re.findall(r'\b\w{4,}\b', log['message'].lower())
                if message_words:
                    key_term = max(message_words, key=len)  # Use longest word as key
                    issue_counter[f"GENERIC_{key_term}"] += 1
        
        # Get top issues
        top_issues = []
        for issue, count in issue_counter.most_common(top_n):
            # Assign emoji based on frequency
            if count > 5:
                emoji = "🔥"  # High severity
            elif count > 3:
                emoji = "⚠️"   # Medium severity
            else:
                emoji = "⚡"   # Low severity
            
            top_issues.append({
                'issue': issue.replace('GENERIC_', ''),
                'count': count,
                'emoji': emoji,
                'severity': 'High' if count > 5 else 'Medium' if count > 3 else 'Low'
            })
        
        return top_issues
    
    def create_visualizations(self):
        """
        Create various visualizations for log analysis
        """
        if self.log_data is None:
            return None, None, None
        
        # Log level distribution
        level_counts = self.log_data['level'].value_counts()
        fig_levels = px.pie(
            values=level_counts.values,
            names=level_counts.index,
            title="Log Level Distribution"
        )
        
        # Timeline analysis (if timestamps available)
        timeline_fig = None
        if self.log_data['timestamp'].notna().any():
            hourly_counts = defaultdict(int)
            for timestamp in self.log_data['timestamp'].dropna():
                try:
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    hour_key = dt.strftime('%H:00')
                    hourly_counts[hour_key] += 1
                except:
                    continue
            
            if hourly_counts:
                timeline_fig = px.bar(
                    x=list(hourly_counts.keys()),
                    y=list(hourly_counts.values()),
                    title="Log Activity by Hour"
                )
        
        # Error type distribution
        error_types = self.log_data['error_type'].value_counts()
        fig_errors = px.bar(
            x=error_types.index,
            y=error_types.values,
            title="Error Types Distribution"
        )
        
        return fig_levels, timeline_fig, fig_errors
    
    def setup_ai_chain(self, groq_api_key: str, model_name: str = "llama3-8b-8192"):
        """
        Optimized setup for AI-powered log analysis using Groq
        """
        try:
            # Check if vector store already exists for this file
            if self.log_hash and self.load_vector_store(self.log_hash):
                # Create Groq LLM
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name=model_name,
                    temperature=0
                )
                
                # Create QA chain with existing vector store
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
                return True
            
            # If no existing vector store, create new one
            st.info("🔄 Creating vector embeddings (this may take a moment)...")
            
            # Initialize embeddings model (cached)
            self.embeddings = get_embeddings_model()
            
            # Create documents from parsed logs with progress tracking
            documents = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Sample logs if too many (for performance)
            logs_to_process = self.parsed_logs
            if len(self.parsed_logs) > 1000:
                st.warning(f"⚡ Large dataset detected ({len(self.parsed_logs)} logs). Sampling 1000 logs for faster processing.")
                # Sample error logs first, then others
                error_logs = [log for log in self.parsed_logs if log['level'] in ['ERROR', 'FATAL', 'WARNING']]
                other_logs = [log for log in self.parsed_logs if log['level'] not in ['ERROR', 'FATAL', 'WARNING']]
                
                # Take all error logs up to 700, then sample other logs
                error_sample = error_logs[:700]
                other_sample = other_logs[:300] if len(other_logs) > 300 else other_logs
                logs_to_process = error_sample + other_sample
            
            for i, log in enumerate(logs_to_process):
                content = f"Level: {log['level']}, Message: {log['message']}"
                if log['timestamp']:
                    content = f"Timestamp: {log['timestamp']}, " + content
                if log['component']:
                    content += f", Component: {log['component']}"
                if log['error_type']:
                    content += f", Error Type: {log['error_type']}"
                
                doc = Document(page_content=content, metadata=log)
                documents.append(doc)
                
                # Update progress
                progress = (i + 1) / len(logs_to_process)
                progress_bar.progress(progress)
                status_text.text(f'Creating documents: {i + 1}/{len(logs_to_process)}')
            
            # Split documents with smaller chunks for better performance
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Reduced chunk size
                chunk_overlap=50  # Reduced overlap
            )
            splits = text_splitter.split_documents(documents)
            
            status_text.text('Generating embeddings...')
            
            # Create embeddings and vector store
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            
            # Save vector store for future use
            if self.log_hash:
                self.save_vector_store(self.log_hash)
            
            # Create Groq LLM
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_name,
                temperature=0
            )
            
            # Create QA chain with fewer retrieved documents for speed
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return True
            
        except Exception as e:
            st.error(f"Error setting up AI chain: {str(e)}")
            return False
    
    def query_logs(self, question: str) -> str:
        """
        Query logs using Groq AI
        """
        if self.qa_chain is None:
            return "AI chain not initialized. Please provide Groq API key."
        
        try:
            result = self.qa_chain({"query": question})
            return result["result"]
        except Exception as e:
            return f"Error querying logs: {str(e)}"
    
    def get_similar_logs(self, query: str, k: int = 5) -> List[Dict]:
        """
        Get similar logs using vector search
        """
        if self.vector_store is None:
            return []
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            similar_logs = []
            for doc in docs:
                similar_logs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })
            return similar_logs
        except Exception:
            # Silently handle similarity search errors
            return []

def main():
    st.set_page_config(
        page_title="Log Analyzer",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Log Analyzer")
    st.markdown("### Intelligent Log Analysis")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = LogAnalyzer()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Try to get Groq API key from environment variable
        groq_key_from_env = os.getenv("GROQ_API_KEY")
        
        if groq_key_from_env:           
            groq_key = groq_key_from_env
            st.session_state.groq_key = groq_key
        else:
            # Fallback to manual input if not found in .env
            groq_key = st.text_input(
                "Groq API Key",
                type="password",
                help="Enter your Groq API key to enable AI-powered analysis (or add GROQ_KEY to .env file)",
                placeholder="Add GROQ_KEY to .env file or enter manually"
            )
            
            if groq_key:
                st.session_state.groq_key = groq_key
            else:
                st.warning("⚠️ No Groq API key found in .env file or entered manually")
        
        # Model selection
        model_options = [
            "llama3-8b-8192",
            "llama3-70b-8192", 
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ]
        
        selected_model = st.selectbox(
            "Select Groq Model",
            model_options,
            help="Choose the Groq model for analysis"
        )
        
        st.session_state.selected_model = selected_model        
        
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This Log Analyzer provides:
        - 📁 Log file upload (.log format)
        - 🔍 Intelligent issue detection
        - 📊 Visual analytics        
        - 📋 Detailed reporting
        - 🔍 Semantic log search
        """)
        
        st.markdown("---")
        st.markdown("### Groq Models")
        st.markdown("""
        - **llama3-8b-8192**: Fast, efficient
        - **llama3-70b-8192**: More capable
        - **mixtral-8x7b-32768**: Large context
        - **gemma-7b-it**: Google's model
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Upload Log File")
        
        uploaded_file = st.file_uploader(
            "Choose a log file",
            type=['log'],
            help="Upload a .log file (max 100MB)"
        )
        
        if uploaded_file:
            # Validate file
            is_valid, message = st.session_state.analyzer.validate_file(uploaded_file)
            
            if is_valid:
                st.success(message)
                
                # Process file
                with st.spinner("Processing log file..."):
                    if st.session_state.analyzer.process_log_file(uploaded_file):
                        st.success("✅ Log file processed successfully!")
                        
                        # Setup AI if API key provided
                        if hasattr(st.session_state, 'groq_key'):
                            # Check if AI is already setup for this file
                            if (hasattr(st.session_state.analyzer, 'qa_chain') and 
                                st.session_state.analyzer.qa_chain is not None and
                                st.session_state.analyzer.log_hash == st.session_state.analyzer.get_file_hash(uploaded_file.getvalue().decode('utf-8'))):
                                st.success("🤖 AI already initialized for this file!")
                            else:
                                with st.spinner("Setting up Groq AI analysis..."):
                                    success = st.session_state.analyzer.setup_ai_chain(
                                        st.session_state.groq_key,
                                        st.session_state.selected_model
                                    )
                                    if success:
                                        st.success("🤖 Groq AI initialized successfully!")
                        
                        # Display basic stats
                        st.markdown("#### 📊 Basic Statistics")
                        log_data = st.session_state.analyzer.log_data
                        
                        metrics_col1, metrics_col2 = st.columns(2)
                        with metrics_col1:
                            st.metric("Total Logs", len(log_data))
                            st.metric("Error Logs", len(log_data[log_data['level'].isin(['ERROR', 'FATAL'])]))
                        
                        with metrics_col2:
                            st.metric("Warning Logs", len(log_data[log_data['level'] == 'WARNING']))
                            st.metric("Unique Components", log_data['component'].nunique())
            
            else:
                st.error(message)
    
    with col2:
        if uploaded_file and hasattr(st.session_state.analyzer, 'log_data') and st.session_state.analyzer.log_data is not None:
            
            # Top Issues Analysis
            st.subheader("🔥 Top Recurring Issues")
            top_issues = st.session_state.analyzer.get_top_issues()
            
            if top_issues:
                for i, issue in enumerate(top_issues, 1):
                    st.markdown(f"""
                    **{i}. {issue['emoji']} {issue['issue']}**
                    - Occurrences: **{issue['count']}**
                    - Severity: **{issue['severity']}**
                    """)
            else:
                st.info("No significant issues detected in the logs.")
            
            # Visualizations
            st.subheader("📈 Visual Analytics")
            
            fig_levels, fig_timeline, fig_errors = st.session_state.analyzer.create_visualizations()
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                if fig_levels:
                    st.plotly_chart(fig_levels, use_container_width=True)
            
            with viz_col2:
                if fig_errors:
                    st.plotly_chart(fig_errors, use_container_width=True)
            
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True)
    
    # AI Query Section
    if (uploaded_file and 
        hasattr(st.session_state.analyzer, 'log_data') and 
        st.session_state.analyzer.log_data is not None):
        
        st.markdown("---")
        st.subheader("🤖 Log Analysis")
        
        # Create tabs for different AI features
        ai_tab1, ai_tab2 = st.tabs(["🔍 Ask Questions", "🔎 Semantic Search"])
        
        with ai_tab1:
            st.markdown("#### Sample Prompts:")
            sample_prompts = [
                "List top 3 issues which are reoccurring",
                "What are the most common error patterns?",
                "Analyze the timeline of errors",
                "What components are failing most frequently?",
                "Summarize the overall system health",
                "What database errors occurred?",
                "Are there any memory-related issues?",
                "Show authentication failures"
            ]
            
            selected_prompt = st.selectbox("Choose a sample prompt:", ["Custom Query"] + sample_prompts)
            
            if selected_prompt == "Custom Query":
                user_query = st.text_input("Enter your question about the logs:")
            else:
                user_query = selected_prompt
            
            if st.button("🔍 Analyze", key="ai_analyze") and user_query:
                with st.spinner("Analyzing logs with Groq AI..."):
                    if hasattr(st.session_state, 'groq_key'):
                        response = st.session_state.analyzer.query_logs(user_query)
                        st.markdown("#### AI Response:")
                        st.markdown(response)
                    else:
                        # Fallback analysis without AI
                        if "top 3 issues" in user_query.lower():
                            top_issues = st.session_state.analyzer.get_top_issues()
                            response = "Top 3 recurring issues:\n"
                            for i, issue in enumerate(top_issues, 1):
                                response += f"{i}. {issue['emoji']} {issue['issue']} - {issue['count']} occurrences\n"
                            st.markdown(response)
                        else:
                            st.info("Please provide a Groq API key for advanced AI analysis, or use the 'List top 3 issues' query.")
        
        with ai_tab2:
            st.markdown("#### Semantic Log Search")
            st.markdown("Find logs by meaning, not just keywords!")
            
            search_query = st.text_input(
                "Search for similar logs:",
                placeholder="e.g., 'database connection failed', 'memory issues', 'user authentication'"
            )
            
            search_count = st.slider("Number of results", 1, 10, 5)
            
            if st.button("🔎 Search Similar Logs", key="semantic_search") and search_query:
                if hasattr(st.session_state, 'groq_key'):
                    with st.spinner("Searching for similar logs..."):
                        similar_logs = st.session_state.analyzer.get_similar_logs(search_query, search_count)
                        
                        if similar_logs:
                            st.markdown("#### Similar Logs Found:")
                            for i, log in enumerate(similar_logs, 1):
                                with st.expander(f"📝 Log {i} - {log['metadata'].get('level', 'UNKNOWN')}"):
                                    st.write("**Content:**", log['content'])
                                    st.write("**Timestamp:**", log['metadata'].get('timestamp', 'N/A'))
                                    st.write("**Component:**", log['metadata'].get('component', 'N/A'))
                                    st.write("**Error Type:**", log['metadata'].get('error_type', 'N/A'))
                        else:
                            st.info("No similar logs found.")
                else:
                    st.info("Please provide a Groq API key to enable semantic search.")

if __name__ == "__main__":
    main()