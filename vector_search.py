import json
from typing import Dict, List
import psycopg2
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

import warnings
warnings.filterwarnings("ignore")

class VectorSearch:
    def __init__(
        self, 
        connection_string: str = "postgresql+psycopg2://langchain:langchain@localhost:6024/langchain",
        collection_name: str = "document_store"
    ):
        """
        Initialize VectorSearch with connection settings and verify database.
        
        Args:
            connection_string: PostgreSQL connection string
            collection_name: Name for the vector collection
        """
        self.connection_string = connection_string
        self.collection_name = collection_name
        
        # Verify database connection
        if not self._verify_database():
            raise Exception("Database verification failed")
            
        # Initialize vector store
        self.vector_store = self._initialize_store()

    def _verify_database(self) -> bool:
        """
        Parse connection string and verify database connection.
        Returns:
            bool: True if database exists and is accessible
        """
        try:
            # Parse connection string
            conn_str = self.connection_string.split('://')[-1]
            creds, rest = conn_str.split('@')
            username, password = creds.split(':')
            host_port, database = rest.split('/')
            host, port = host_port.split(':')
            
            # Setup connection
            conn = psycopg2.connect(
                user=username,
                password=password,
                host=host,
                port=port,
                database=database
            )
            
            # Check pgvector extension
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            has_vector = cursor.fetchone() is not None
            
            cursor.close()
            conn.close()
            
            if not has_vector:
                print("Error: pgvector extension is not installed in the database")
                return False
                
            return True
            
        except psycopg2.Error as e:
            print(f"Database verification failed: {str(e)}")
            return False

    def _initialize_store(self) -> PGVector:
        """Initialize vector store with embeddings."""
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            return PGVector(
                collection_name=self.collection_name,
                connection_string=self.connection_string,
                embedding_function=embeddings,
            )
        except Exception as e:
            raise Exception(f"Vector store initialization failed: {str(e)}")

    def add_document(self, json_file_path: str) -> bool:
        """
        Process and add a document from JSON file if it doesn't exist.
        
        Args:
            json_file_path: Path to the JSON file
            
        Returns:
            bool: True if document was added successfully
        """
        try:
            # Load JSON file
            with open(json_file_path, 'r') as file:
                json_data = json.load(file)
            
            # Check document ID
            document_id = json_data.get("document_id")
            if document_id is None:
                raise ValueError("JSON document must contain a document_id field")
            
            # Check if document already exists
            try:
                existing = self.vector_store.similarity_search_with_score(
                    "dummy query",
                    k=1,
                    filter={"document_id": document_id}
                )
                if len(existing) > 0:
                    print(f"Document with ID {document_id} already exists")
                    return False
            except Exception:
                pass  
            
            # Prepare document
            text_content = f"""
            Summary: {json_data.get('summary', '')}
            Keywords: {', '.join(json_data.get('key_words', []))}
            """
            
            document = Document(
                page_content=text_content,
                metadata={
                    "document_id": document_id,
                    "keywords": json_data.get("key_words", [])
                }
            )
            
            # Store document
            self.vector_store.add_documents([document])
            print(f"Successfully added document with ID {document_id}")
            return True
            
        except Exception as e:
            print(f"Failed to add document: {str(e)}")
            return False

    def search_docs(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar documents based on a query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of dictionaries containing document content and similarity score
        """
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            return [{
                'content': doc.page_content.strip(),
                'metadata': doc.metadata,
                'similarity_score': float(score)
            } for doc, score in results]
            
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")

# Example usage
def main():
    try:
        # Initialize vector search
        vector_search = VectorSearch(
            connection_string="postgresql+psycopg2://langchain:langchain@localhost:6024/langchain",
            collection_name="document_store"
        )
        
        # # Add a document
        # vector_search.add_document("path/to/your/document.json")
        
        # Search similar documents
        results = vector_search.search_docs(
            query="How to block specific domains in text fields",
            k=5
        )
        
        #print document ids and similarity scores
        for result in results:
            print(f"Document ID: {result['metadata']['document_id']} - Similarity Score: {result['similarity_score']}")
        
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()