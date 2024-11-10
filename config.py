
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # PostgreSQL Configuration
    # First check for complete connection string
    POSTGRES_CONNECTION_STRING = os.getenv('POSTGRES_CONNECTION_STRING')
    
    # If no connection string is provided, build it from components
    if not POSTGRES_CONNECTION_STRING:
        POSTGRES_USER = os.getenv('POSTGRES_USER', 'langchain')
        POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'langchain')
        POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
        POSTGRES_PORT = os.getenv('POSTGRES_PORT', '6024')
        POSTGRES_DB = os.getenv('POSTGRES_DB', 'langchain')
        
        POSTGRES_CONNECTION_STRING = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    
    # MongoDB Configuration
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
    MONGO_DB = os.getenv('MONGO_DB', 'retriever')
    
    # Collection names
    VECTOR_COLLECTION_NAME = os.getenv('VECTOR_COLLECTION_NAME', 'document_store')
    MONGO_COLLECTION_NAME = os.getenv('MONGO_COLLECTION_NAME', 'wp_forms')
    
    @classmethod
    def get_postgres_connection(cls) -> str:
        """Get PostgreSQL connection string with error handling"""
        if not cls.POSTGRES_CONNECTION_STRING:
            raise ValueError("PostgreSQL connection string is not configured")
        return cls.POSTGRES_CONNECTION_STRING
    
    @classmethod
    def get_mongo_uri(cls) -> str:
        """Get MongoDB URI with error handling"""
        if not cls.MONGO_URI:
            raise ValueError("MongoDB URI is not configured")
        return cls.MONGO_URI

    @staticmethod
    def validate_connections():
        """Validate all connection strings are properly formatted"""
        try:
            # Validate PostgreSQL connection string
            postgres_conn = Config.get_postgres_connection()
            if not postgres_conn.startswith('postgresql'):
                raise ValueError("Invalid PostgreSQL connection string format")
            
            # Validate MongoDB URI
            mongo_uri = Config.get_mongo_uri()
            if not mongo_uri.startswith('mongodb'):
                raise ValueError("Invalid MongoDB URI format")
                
            return True
            
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")