import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import MagicMock, patch
from app import LogAnalyzer

class TestLogAnalyzer:
    """
    Comprehensive unit tests for LogAnalyzer class
    """
    
    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = LogAnalyzer()
        self.sample_log_content = """
2024-01-15 10:30:25 INFO [WebApp] Application started successfully
2024-01-15 10:30:30 ERROR [Database] Connection failed to mysql://localhost:3306
2024-01-15 10:30:35 WARN [Security] Invalid login attempt from 192.168.1.100
2024-01-15 10:30:40 ERROR [Database] Connection failed to mysql://localhost:3306
2024-01-15 10:30:45 FATAL [Memory] OutOfMemoryError: Java heap space
2024-01-15 10:30:50 ERROR [Network] Connection timeout after 30 seconds
2024-01-15 10:30:55 ERROR [Database] SQLException: Table 'users' doesn't exist
2024-01-15 10:31:00 ERROR [Database] Connection failed to mysql://localhost:3306
2024-01-15 10:31:05 INFO [WebApp] User login successful for admin@company.com
        """.strip()
    
    def create_mock_file(self, content, filename="test.log", size=1024):
        """Create a mock uploaded file"""
        mock_file = MagicMock()
        mock_file.name = filename
        mock_file.size = size
        mock_file.read.return_value = content.encode('utf-8')
        return mock_file
    
    def test_file_validation_success(self):
        """Test successful file validation"""
        mock_file = self.create_mock_file(self.sample_log_content, "test.log", 1024)
        is_valid, message = self.analyzer.validate_file(mock_file)
        
        assert is_valid == True
        assert message == "File validation successful"
    
    def test_file_validation_wrong_format(self):
        """Test file validation with wrong format"""
        mock_file = self.create_mock_file(self.sample_log_content, "test.txt", 1024)
        is_valid, message = self.analyzer.validate_file(mock_file)
        
        assert is_valid == False
        assert message == "Upload only valid file format (.log files only)"
    
    def test_file_validation_size_exceeded(self):
        """Test file validation with size exceeded"""
        large_size = 101 * 1024 * 1024  # 101MB
        mock_file = self.create_mock_file(self.sample_log_content, "large.log", large_size)
        is_valid, message = self.analyzer.validate_file(mock_file)
        
        assert is_valid == False
        assert message == "File size exceeds 100MB limit"
