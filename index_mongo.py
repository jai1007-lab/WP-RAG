from pymongo import MongoClient
import json
from typing import Dict, List
from config import Config

import warnings
warnings.filterwarnings("ignore")

from vector_search import VectorSearch


class IndexDB:
    def __init__(
        self,
        mongo_uri: str = None,
        database_name: str = None,
        collection_name: str = None
    ):
        """
        Initialize MongoDB connection.
        
        Args:
            mongo_uri: Optional MongoDB URI
            database_name: Optional database name
            collection_name: Optional collection name
        """
        # Validate configuration
        Config.validate_connections()
        
        # Use provided values or defaults from Config
        self.mongo_uri = mongo_uri or Config.get_mongo_uri()
        self.database_name = database_name or Config.MONGO_DB
        self.collection_name = collection_name or Config.MONGO_COLLECTION_NAME
        
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            # Create index on document_id
            self.collection.create_index("document_id")
            print("Successfully connected to MongoDB")
            
        except Exception as e:
            raise Exception(f"MongoDB connection failed: {str(e)}")
    
    def store_document(self, json_file_path: str) -> bool:
        """
        Store a document in MongoDB.
        
        Args:
            json_file_path: Path to JSON file containing the document
            
        Returns:
            bool: True if document was stored successfully
        """
        try:
            # Load and validate document
            with open(json_file_path, 'r') as file:
                document = json.load(file)
            
            if 'document_id' not in document:
                raise ValueError("Document must contain a document_id field")
            
            # Check if document already exists
            if self.collection.find_one({"document_id": document['document_id']}):
                print(f"Document {document['document_id']} already exists in MongoDB")
                return False
            
            # Store document if it doesn't exist
            self.collection.insert_one(document)
            print(f"Stored document {document['document_id']} in MongoDB")
            return True
            
        except Exception as e:
            print(f"Failed to store document: {str(e)}")
            return False
    
    def get_matching_documents(self, vector_search_results: List[Dict]) -> List[Dict]:
        """
        Retrieve documents from MongoDB based on vector search results.
        
        Args:
            vector_search_results: Results from VectorSearch containing document IDs
            
        Returns:
            List of documents with their similarity scores
        """
        try:
            # Extract document IDs and scores from vector search results
            doc_scores = {}
            for result in vector_search_results:
                content = result['content']
                metadata = result['metadata']
                score = result['similarity_score']
                
                if 'document_id' in metadata:
                    doc_scores[metadata['document_id']] = {
                        'vector_content': content,
                        'score': score
                    }
            
            # Fetch documents from MongoDB
            docs = list(self.collection.find(
                {"document_id": {"$in": list(doc_scores.keys())}},
                {"_id": 0}  
            ))
            
            # Combine MongoDB documents with vector search data
            results = []
            for doc in docs:
                doc_id = doc['document_id']
                if doc_id in doc_scores:
                    results.append({
                        'document': doc,
                        'vector_content': doc_scores[doc_id]['vector_content'],
                        'similarity_score': doc_scores[doc_id]['score']
                    })
            
            # Sort by similarity score
            results.sort(key=lambda x: x['similarity_score'])
            return results
            
        except Exception as e:
            raise Exception(f"Failed to retrieve documents: {str(e)}")

# Example usage
def main():
    try:
        
        vector_search = VectorSearch()
        index_db = IndexDB()
        
        # # Add document to both stores
        # json_file = "path/to/your/document.json"
        # vector_search.add_document(json_file)
        # index_db.store_document(json_file)
        
        # Search for similar documents
        query = "How to block specific domains in text fields"
        vector_results = vector_search.search_docs(query, k=5)
        
        # Get full documents with vector search context
        full_results = index_db.get_matching_documents(vector_results)
        
        # Print document ids and similarity scores
        for result in full_results:
            print(f"Document ID: {result['document']['document_id']}")
            print(f"Similarity Score: {result['similarity_score']}")
            print(f"Vector Content: {result['vector_content']}")
            print(f"Document Content: {result['document']['content'][:200]}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()