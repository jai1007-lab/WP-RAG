from vector_search import VectorSearch
from index_mongo import IndexDB

from typing import Dict, List
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")


class QueryLLM:
    def __init__(
        self,
        vector_search,  # VectorSearch instance
        index_db,      # IndexDB instance
        model_name: str = "llama3.1",
        temperature: float = 0.2,
        max_history: int = 5
    ):
        """
        Initialize ConversationalRetriever with conversation history support.
        
        Args:
            vector_search: Instance of VectorSearch class
            index_db: Instance of IndexDB class
            model_name: Name of the Ollama model to use
            temperature: LLM temperature parameter
            max_history: Maximum number of conversation turns to maintain
        """
        self.vector_search = vector_search
        self.index_db = index_db
        self.max_history = max_history
        self.conversation_history = []
        
        # Initialize Ollama LLM
        self.llm = Ollama(
            model=model_name,
            temperature=temperature
        )
        
        # Define conversational prompt template
        self.prompt_template = PromptTemplate(
            template="""You are a helpful assistant engaging in a conversation. Use the following context and conversation history to provide a natural, contextual response.
            
            Context: {context}
            
            Conversation History:
            {history}
            
            Current Question: {question}
            
            Provide a detailed, conversational response that maintains context from the previous discussion:""",
            input_variables=["context", "history", "question"]
        )

    def _format_conversation_history(self) -> str:
        """Format the conversation history for the prompt."""
        formatted_history = []
        for turn in self.conversation_history[-self.max_history:]:
            formatted_history.append(f"Human: {turn['query']}")
            formatted_history.append(f"Assistant: {turn['response']}")
        return "\n".join(formatted_history)

    def _prepare_context(self, vector_results: List[Dict], mongo_results: List[Dict]) -> str:
        """Prepare context from vector and MongoDB results."""
        context_parts = []
        
        for v_result, m_result in zip(vector_results, mongo_results):
            context_parts.append(
                f"""
                Relevance Score: {v_result['similarity_score']}
                
                Vector Search Content:
                {v_result['content']}
                
                Full Document:
                Summary: {m_result['document'].get('summary', '')}
                Keywords: {', '.join(m_result['document'].get('key_words', []))}
                Document ID: {m_result['document'].get('document_id', '')}
                """
            )
        
        return "\n\n".join(context_parts)

    def chat(self, query: str, num_results: int = 5) -> Dict:
        """Process a single chat interaction."""
        try:
            # Get vector search results
            vector_results = self.vector_search.search_docs(query, k=num_results)
            
            # Get MongoDB documents
            mongo_results = self.index_db.get_matching_documents(vector_results)
            
            # Prepare context
            context = self._prepare_context(vector_results, mongo_results)
            
            # Format conversation history
            history = self._format_conversation_history()
            
            # Create prompt
            prompt = self.prompt_template.format(
                context=context,
                history=history,
                question=query
            )
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            
            # Add to conversation history
            self.conversation_history.append({
                'timestamp': datetime.now(),
                'query': query,
                'response': response,
                'context_used': {
                    'vector_results': len(vector_results),
                    'mongo_results': len(mongo_results)
                }
            })
            
            return {
                'query': query,
                'response': response,
                'context': {
                    'vector_results': vector_results,
                    'mongo_results': mongo_results
                }
            }
            
        except Exception as e:
            raise Exception(f"Failed to process chat: {str(e)}")

    def start_interactive_chat(self, num_results: int = 5):
        """
        Start an interactive chat session with the user.
        
        Args:
            num_results: Number of similar documents to retrieve per query
        """
        print("Starting interactive chat session...")
        print('Type "exit" to end the conversation')
        print('Type "reset" to clear conversation history')
        print('Type "summary" to see conversation summary')
        print("-" * 50 + "\n")
        
        while True:
            try:
                query = input("You: ").strip()
                
                if query.lower() == "exit":
                    print("\nEnding chat session...")
                    break
                    
                elif query.lower() == "reset":
                    self.reset_conversation()
                    print("\nConversation history cleared!")
                    continue
                    
                elif query.lower() == "summary":
                    summary = self.get_conversation_summary()
                    print("\nConversation Summary:")
                    print(f"Total turns: {summary['total_turns']}")
                    print(f"Start time: {summary['conversation_start']}")
                    print(f"End time: {summary['conversation_end']}")
                    continue
                
                if not query:
                    continue
                
                result = self.chat(query, num_results=num_results)
                print(f"\nAssistant: {result['response']}")
                print("\n" + "-" * 50 + "\n")
                
            except KeyboardInterrupt:
                print("\nEnding chat session...")
                break
                
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again.")
                continue

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []

    def get_conversation_summary(self) -> Dict:
        """Get a summary of the current conversation."""
        return {
            'total_turns': len(self.conversation_history),
            'conversation_start': self.conversation_history[0]['timestamp'] if self.conversation_history else None,
            'conversation_end': self.conversation_history[-1]['timestamp'] if self.conversation_history else None,
            'history': self.conversation_history
        }


# Example usage
def main():
    try:
        # Initialize components
        vector_search = VectorSearch()
        index_db = IndexDB()
        chatbot = QueryLLM(
            vector_search=vector_search,
            index_db=index_db,
            model_name="llama3.1",
            temperature=0.2
        )
        
        # Start interactive chat session
        chatbot.start_interactive_chat(num_results=3)
        
    except Exception as e:
        print(f"Error during initialization: {str(e)}")

if __name__ == "__main__":
    main()