import pandas as pd
import re
import hashlib
import pickle
import os
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime

from app.ai.embeddings import get_embeddings_model
from app.cache.vector_cache import save_cached_data, load_cached_data
from app.cache.vector_cache import save_vector_store, load_vector_store

class LogAnalyzer:
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

    def get_file_hash(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()

    def parse_log_line(self, line: str) -> Dict[str, Any]:
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

        ts = re.search(timestamp_pattern, line)
        if ts:
            parsed['timestamp'] = ts.group(1)

        lv = re.search(level_pattern, line, re.IGNORECASE)
        if lv:
            parsed['level'] = lv.group(1).upper()

        comp = re.search(r'\[([^\]]+)\]|\b([A-Z][a-zA-Z]*(?:\.[A-Z][a-zA-Z]*)*)\b', line)
        if comp:
            parsed['component'] = comp.group(1) or comp.group(2)

        for etype, pattern in self.error_patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                parsed['error_type'] = etype
                break

        return parsed

    def process_log_file(self, uploaded_file) -> bool:
        try:
            content = uploaded_file.read().decode('utf-8')
            file_hash = self.get_file_hash(content)

            if load_cached_data(file_hash):
                return True

            lines = content.splitlines()
            self.parsed_logs = [self.parse_log_line(line) for line in lines if line.strip()]
            self.log_data = pd.DataFrame(self.parsed_logs)
            self.log_hash = file_hash

            save_cached_data(file_hash, self)
            return True
        except Exception as e:
            print(f"Error processing log file: {str(e)}")
            return False

    def get_top_issues(self, top_n: int = 3) -> List[Dict]:
        if self.log_data is None:
            return []

        error_logs = self.log_data[
            self.log_data['level'].isin(['ERROR', 'FATAL', 'WARNING', 'WARN'])
        ]

        issue_counter = Counter()

        for _, log in error_logs.iterrows():
            if log['error_type']:
                issue_counter[log['error_type']] += 1
            else:
                words = re.findall(r'\b\w{4,}\b', log['message'].lower())
                if words:
                    key = max(words, key=len)
                    issue_counter[f"GENERIC_{key}"] += 1

        top_issues = []
        for issue, count in issue_counter.most_common(top_n):
            emoji = "🔥" if count > 5 else "⚠️" if count > 3 else "⚡"
            top_issues.append({
                'issue': issue.replace('GENERIC_', ''),
                'count': count,
                'emoji': emoji,
                'severity': 'High' if count > 5 else 'Medium' if count > 3 else 'Low'
            })

        return top_issues
