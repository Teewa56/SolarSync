import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import psycopg2
import logging
from typing import Optional

load_dotenv()
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Secure database configuration management"""
    
    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = os.getenv("DB_PORT", "5432")
        self.database = os.getenv("DB_NAME", "solarsync")
        self.user = os.getenv("DB_USER")
        self.password = os.getenv("DB_PASSWORD")
        
        # Connection pool settings
        self.pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
        self.max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))
        
        self._validate_config()
    
    def _validate_config(self):
        """Validate required environment variables"""
        if not self.user:
            raise ValueError("DB_USER environment variable is required")
        if not self.password:
            raise ValueError("DB_PASSWORD environment variable is required")
    
    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    def get_psycopg2_connection(self):
        """Get psycopg2 connection"""
        try:
            conn = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port,
                connect_timeout=10
            )
            logger.info("PostgreSQL connection established")
            return conn
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def get_sqlalchemy_engine(self, poolclass=None):
        """Get SQLAlchemy engine with connection pooling"""
        try:
            engine = create_engine(
                self.get_connection_string(),
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                poolclass=poolclass,
                pool_pre_ping=True,  # Verify connections before using
                echo=False
            )
            logger.info("SQLAlchemy engine created")
            return engine
        except Exception as e:
            logger.error(f"Failed to create SQLAlchemy engine: {e}")
            raise


# Global instance
db_config = DatabaseConfig()


# Helper functions for backward compatibility
def get_db_connection():
    """Get psycopg2 connection (backward compatible)"""
    return db_config.get_psycopg2_connection()


def get_engine():
    """Get SQLAlchemy engine"""
    return db_config.get_sqlalchemy_engine()


# Example .env file structure
"""
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=solarsync
DB_USER=solarsync_user
DB_PASSWORD=your_secure_password_here

# Connection Pool Settings
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
"""