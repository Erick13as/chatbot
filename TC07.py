import os
from typing import List, Optional, Tuple, Dict
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import logging
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGChatbot:
    def __init__(
        self, 
        model_name: str = "llama3",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        batch_size: int = 50
    ):
        # Initialize embeddings
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                cache_folder="./embedding_cache"
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
        
        self.vector_store = None
        
        # Configure Ollama LLM
        try:
            self.llm = Ollama(
                model=model_name,
                temperature=0.7,
                num_ctx=4096,
                top_k=10,
                top_p=0.9,
                base_url="http://localhost:11434",
            )
        except Exception as e:
            logger.error(f"Error initializing Ollama: {str(e)}")
            raise
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        
        # Prompt template optimized for BBC articles
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable assistant specialized in analyzing BBC news articles.
            Use the provided context to answer questions accurately and comprehensively.
            If the answer cannot be found in the context, clearly state that.
            Maintain the journalistic tone of BBC while responding.
            
            Context: {context}"""),
            ("human", "{question}")
        ])

    def load_bbc_csv(self, csv_path: str) -> Tuple[List[str], Optional[List[str]]]:
        """
        Load BBC dataset from CSV file.
        """
        logger.info(f"Loading BBC dataset from: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            if 'data' not in df.columns:
                raise ValueError(f"Missing required column 'data'. Found: {df.columns.tolist()}")
            
            documents = df['data'].dropna().tolist()
            labels = df['labels'].dropna().tolist() if 'labels' in df.columns else None
            
            logger.info(f"Loaded {len(documents)} documents")
            if labels:
                unique_labels = set(labels)
                logger.info(f"Categories found: {unique_labels}")
                
            return documents, labels
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise

    def process_documents(self, documents: List[str]) -> None:
        """
        Process documents and create vector store.
        """
        logger.info("Starting document processing")
        
        if not documents:
            raise ValueError("No documents provided")
        
        try:
            # Configure text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            
            # Process documents in batches
            all_splits = []
            for i in tqdm(range(0, len(documents), self.batch_size), 
                         desc="Processing documents"):
                batch = documents[i:i + self.batch_size]
                batch_splits = []
                
                for doc in batch:
                    splits = text_splitter.split_text(str(doc))
                    batch_splits.extend(splits)
                
                batch_splits = list(dict.fromkeys(batch_splits))
                all_splits.extend(batch_splits)
            
            # Create vector store with batched processing
            logger.info("Creating vector store")
            embedding_batch_size = min(50, len(all_splits))
            
            if not all_splits:
                raise ValueError("No text splits generated")
            
            # Initialize first batch
            initial_batch = all_splits[:embedding_batch_size]
            self.vector_store = FAISS.from_texts(
                initial_batch,
                self.embeddings,
                metadatas=[{"index": i} for i in range(len(initial_batch))]
            )
            
            # Process remaining batches
            for i in tqdm(range(embedding_batch_size, len(all_splits), embedding_batch_size),
                         desc="Creating vector store"):
                batch = all_splits[i:i + embedding_batch_size]
                if batch:
                    temp_store = FAISS.from_texts(
                        batch,
                        self.embeddings,
                        metadatas=[{"index": j} for j in range(i, i + len(batch))]
                    )
                    self.vector_store.merge_from(temp_store)
            
            logger.info(f"Successfully created vector store with {len(all_splits)} chunks")
            
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def query(self, question: str, k: int = 3) -> Dict:
        """
        Query the chatbot with enhanced response information.
        """
        if not self.vector_store:
            return {
                "error": "Document store not initialized. Please process documents first.",
                "response": None,
                "context": None
            }
            
        try:
            # Get relevant documents
            docs = self.vector_store.similarity_search(question, k=k)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Generate response
            chain = self.prompt | self.llm
            response = chain.invoke({
                "context": context,
                "question": question
            })
            
            return {
                "response": response,
                "context": [doc.page_content for doc in docs],
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            return {
                "error": f"Error generating response: {str(e)}",
                "response": None,
                "context": None
            }

def create_chatbot_with_bbc_data(
    csv_path: str,
    model_name: str = "llama3"
) -> RAGChatbot:
    """
    Create and initialize chatbot with BBC data.
    """
    logger.info("Initializing chatbot")
    
    try:
        chatbot = RAGChatbot(model_name=model_name)
        documents, _ = chatbot.load_bbc_csv(csv_path)
        chatbot.process_documents(documents)
        return chatbot
        
    except Exception as e:
        logger.error(f"Chatbot creation error: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration
    csv_path = "bbc_data.csv"

    print("""
    Before running this script, ensure:
    1. Ollama is installed (https://ollama.ai)
    2. Run these commands in terminal:
       - ollama serve (sirva o no pasar al otro comando)
       - ollama pull llama3
    3. Have bbc_data.csv in the correct directory
    """)

    # Create and initialize chatbot
    chatbot = create_chatbot_with_bbc_data(
        csv_path=csv_path,
        model_name="llama3"
    )
    
    # Interactive query loop
    print("\nChatbot ready! Type 'quit' to exit.")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break
            
        result = chatbot.query(question)
        if result["error"]:
            print(f"Error: {result['error']}")
        else:
            print("\nResponse:")
            print(result["response"])
            #Questions:
            #What are some major sports events discussed in the articles?