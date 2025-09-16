import streamlit as st
import os
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain
from langchain_core.prompts import PromptTemplate

from dependencies import (
    check_password, 
    setup_page_config, 
    show_demo_navigation, 
    display_api_status,
    add_feedback_section
)

# Load environment variables
_ = load_dotenv(find_dotenv())

# Page configuration
setup_page_config(
    page_title="RAG System Quality Evaluator",
    page_icon="📊",
    layout="wide"
)

# Authentication check
if not check_password():
    st.stop()

class RAGEvaluator:
    """Enhanced RAG evaluation system with comprehensive analysis."""
    
    def __init__(self):
        self.api_key = self.get_api_key()
        self.embeddings = None
        self.vector_store = None
        self.qa_chain = None
        self.eval_chain = None
        
    def get_api_key(self) -> str:
        """Get API key from environment or secrets."""
        return (
            os.environ.get("OPENAI_API_KEY") or 
            st.secrets.get("APIKEY", {}).get("OPENAI_API_KEY", "")
        )
    
    def validate_inputs(self, uploaded_file, query_text: str, response_text: str) -> Tuple[bool, str]:
        """Validate all inputs before processing."""
        if not uploaded_file:
            return False, "Please upload a document first."
        
        if not query_text.strip():
            return False, "Please enter a question to evaluate."
        
        if not response_text.strip():
            return False, "Please enter the expected correct answer."
        
        if len(query_text.strip()) < 10:
            return False, "Question should be at least 10 characters long."
        
        if len(response_text.strip()) < 10:
            return False, "Expected answer should be at least 10 characters long."
        
        return True, ""
    
    def process_document(self, uploaded_file, chunk_strategy: str = "character") -> List[Any]:
        """Process and chunk the uploaded document."""
        try:
            # Read and decode the file
            content = uploaded_file.read().decode('utf-8')
            
            # Choose text splitter based on strategy
            if chunk_strategy == "recursive":
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
            else:  # character
                text_splitter = CharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separator="\n"
                )
            
            # Create document chunks
            documents = text_splitter.create_documents([content])
            
            return documents, content
            
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")
    
    def create_vector_store(self, documents: List[Any]) -> FAISS:
        """Create and populate the vector store."""
        if not self.api_key or not self.api_key.startswith("sk-"):
            raise Exception("Valid OpenAI API key required")
        
        try:
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
            
            # Create vector store
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            return self.vector_store
            
        except Exception as e:
            raise Exception(f"Error creating vector store: {str(e)}")
    
    def setup_qa_chain(self, chain_type: str = "stuff") -> RetrievalQA:
        """Set up the QA chain with specified parameters."""
        try:
            # Initialize LLM
            llm = OpenAI(
                openai_api_key=self.api_key,
                temperature=0.1,  # Low temperature for consistent answers
                max_tokens=500
            )
            
            # Create retriever
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type=chain_type,
                retriever=retriever,
                input_key="question",
                return_source_documents=True
            )
            
            return self.qa_chain
            
        except Exception as e:
            raise Exception(f"Error setting up QA chain: {str(e)}")
    
    def setup_eval_chain(self) -> QAEvalChain:
        """Set up the evaluation chain."""
        try:
            # Enhanced evaluation prompt
            eval_prompt = PromptTemplate(
                input_variables=["query", "answer", "result"],
                template="""You are an expert evaluator assessing the quality of AI-generated answers.

Compare the AI's response to the expected correct answer and provide a detailed evaluation.

Question: {query}
Expected Answer: {answer}  
AI Response: {result}

Evaluation Criteria:
1. Accuracy: Does the AI response contain correct information?
2. Completeness: Does it address all aspects of the question?
3. Relevance: Is the response directly related to the question?
4. Clarity: Is the response clear and well-structured?

Provide your evaluation in this format:
GRADE: [CORRECT/PARTIALLY_CORRECT/INCORRECT]
ACCURACY_SCORE: [0-100]
EXPLANATION: [Detailed explanation of the evaluation]
STRENGTHS: [What the AI did well]
IMPROVEMENTS: [What could be better]

Your evaluation:"""
            )
            
            # Create evaluation LLM with different temperature for more detailed analysis
            eval_llm = OpenAI(
                openai_api_key=self.api_key,
                temperature=0.3,
                max_tokens=800
            )
            
            self.eval_chain = QAEvalChain.from_llm(
                llm=eval_llm,
                prompt=eval_prompt
            )
            
            return self.eval_chain
            
        except Exception as e:
            raise Exception(f"Error setting up evaluation chain: {str(e)}")
    
    def evaluate_response(self, query_text: str, expected_answer: str) -> Dict[str, Any]:
        """Perform comprehensive evaluation of the RAG system."""
        try:
            # Prepare QA data
            qa_data = [{
                "question": query_text,
                "answer": expected_answer
            }]
            
            # Get AI predictions
            predictions = self.qa_chain.apply(qa_data)
            
            # Evaluate predictions
            graded_outputs = self.eval_chain.evaluate(
                qa_data,
                predictions,
                question_key="question",
                prediction_key="result",
                answer_key="answer"
            )
            
            # Extract source documents for transparency
            source_docs = predictions[0].get("source_documents", [])
            
            # Calculate additional metrics
            ai_response = predictions[0]["result"]
            response_length = len(ai_response.split())
            expected_length = len(expected_answer.split())
            
            return {
                "predictions": predictions,
                "graded_outputs": graded_outputs,
                "source_documents": source_docs,
                "metrics": {
                    "response_length": response_length,
                    "expected_length": expected_length,
                    "length_ratio": response_length / expected_length if expected_length > 0 else 0,
                    "evaluation_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            raise Exception(f"Error during evaluation: {str(e)}")

def display_header():
    """Display the application header."""
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white; text-align: center;'>
        <h1>📊 RAG System Quality Evaluator</h1>
        <p>Comprehensive testing and validation for Retrieval-Augmented Generation systems</p>
        <p><em>Ensure accuracy and reliability of your AI-powered Q&A systems</em></p>
    </div>
    """, unsafe_allow_html=True)

def display_methodology():
    """Display the evaluation methodology."""
    with st.expander("📋 Evaluation Methodology", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔍 How It Works:**
            1. **Document Processing** - Upload and chunk your document
            2. **Vector Storage** - Create embeddings and searchable index
            3. **Question Testing** - Ask questions with known answers
            4. **AI Response Generation** - Get RAG system's answer
            5. **Evaluation** - Compare against expected correct answer
            6. **Scoring** - Receive detailed accuracy assessment
            """)
        
        with col2:
            st.markdown("""
            **📈 Evaluation Criteria:**
            - **Accuracy** - Factual correctness of the response
            - **Completeness** - Coverage of all question aspects
            - **Relevance** - Direct relation to the question
            - **Clarity** - Structure and readability
            - **Source Attribution** - Which document sections were used
            """)

def display_document_analysis(content: str, documents: List[Any]):
    """Display document analysis information."""
    st.markdown("### 📄 Document Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Characters", f"{len(content):,}")
    
    with col2:
        word_count = len(content.split())
        st.metric("Word Count", f"{word_count:,}")
    
    with col3:
        st.metric("Document Chunks", len(documents))
    
    with col4:
        avg_chunk_size = sum(len(doc.page_content) for doc in documents) // len(documents)
        st.metric("Avg Chunk Size", f"{avg_chunk_size} chars")
    
    # Show first few chunks as preview
    with st.expander("📖 Document Chunks Preview", expanded=False):
        for i, doc in enumerate(documents[:3]):
            st.markdown(f"**Chunk {i+1}:**")
            st.text_area(
                f"Content {i+1}",
                doc.page_content,
                height=100,
                disabled=True,
                label_visibility="collapsed"
            )

def display_evaluation_results(results: Dict[str, Any]):
    """Display comprehensive evaluation results."""
    st.markdown("---")
    st.markdown("## 📊 Evaluation Results")
    
    prediction = results["predictions"][0]
    graded_output = results["graded_outputs"][0]
    metrics = results["metrics"]
    
    # Main results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ❓ Question Tested")
        st.info(prediction["question"])
        
        st.markdown("### ✅ Expected Answer")
        st.success(prediction["answer"])
    
    with col2:
        st.markdown("### 🤖 AI Response")
        st.warning(prediction["result"])
        
        st.markdown("### 📝 Evaluation Grade")
        grade_result = graded_output.get("results", "Not available")
        
        # Color code based on grade
        if "CORRECT" in grade_result.upper():
            st.success(f"✅ {grade_result}")
        elif "PARTIALLY" in grade_result.upper():
            st.warning(f"⚠️ {grade_result}")
        else:
            st.error(f"❌ {grade_result}")
    
    # Detailed metrics
    st.markdown("### 📈 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Response Length", f"{metrics['response_length']} words")
    
    with col2:
        st.metric("Expected Length", f"{metrics['expected_length']} words")
    
    with col3:
        ratio = metrics['length_ratio']
        st.metric("Length Ratio", f"{ratio:.2f}")
    
    with col4:
        source_count = len(results.get("source_documents", []))
        st.metric("Sources Used", source_count)
    
    # Source documents analysis
    if results.get("source_documents"):
        st.markdown("### 📚 Source Documents Used")
        
        for i, doc in enumerate(results["source_documents"]):
            with st.expander(f"📄 Source {i+1}", expanded=False):
                st.text_area(
                    f"Source content {i+1}",
                    doc.page_content,
                    height=120,
                    disabled=True,
                    label_visibility="collapsed"
                )
    
    # Quality analysis
    st.markdown("### 🎯 Quality Analysis")
    
    analysis_cols = st.columns(2)
    
    with analysis_cols[0]:
        st.markdown("""
        **✅ System Strengths:**
        - Retrieved relevant document sections
        - Generated coherent response
        - Maintained context from sources
        - Provided specific information
        """)
    
    with analysis_cols[1]:
        st.markdown("""
        **🔧 Potential Improvements:**
        - Fine-tune chunk size for better context
        - Adjust retrieval parameters (k value)
        - Optimize embedding model selection
        - Enhance prompt engineering
        """)

def main():
    """Main application logic."""
    
    display_header()
    show_demo_navigation()
    display_api_status()
    
    # Introduction
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎯 Ensure RAG System Reliability")
        st.markdown("""
        Test and validate your Retrieval-Augmented Generation system's accuracy by:
        - **Uploading documents** that your RAG system will search through
        - **Asking specific questions** with known correct answers
        - **Comparing AI responses** against expected results
        - **Receiving detailed evaluation** with accuracy scoring
        - **Identifying improvement areas** for system optimization
        """)
    
    with col2:
        st.markdown("### 💼 Business Value:")
        st.markdown("""
        - **Quality Assurance**
        - **Accuracy Validation**  
        - **Performance Monitoring**
        - **System Optimization**
        - **Reliability Testing**
        """)
    
    display_methodology()
    
    # Configuration section
    st.markdown("---")
    st.markdown("## ⚙️ Evaluation Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chunk_strategy = st.selectbox(
            "📄 Document Chunking Strategy:",
            ["character", "recursive"],
            help="Character: Simple character-based splitting. Recursive: Smart splitting on paragraphs, sentences, etc."
        )
        
        chain_type = st.selectbox(
            "🔗 QA Chain Type:",
            ["stuff", "map_reduce", "refine"],
            help="Stuff: Combine all chunks. Map_reduce: Process chunks separately then combine. Refine: Iteratively refine answers."
        )
    
    with col2:
        st.markdown("""
        **📋 Input Requirements:**
        - **Document**: .txt file with content to test against
        - **Question**: Specific question with known answer
        - **Expected Answer**: The correct answer for comparison
        - **File Size**: Maximum 10MB recommended
        """)
    
    # File upload and input section
    st.markdown("---")
    st.markdown("## 📁 Upload Test Document")
    
    uploaded_file = st.file_uploader(
        "Choose a .txt document to test against:",
        type=["txt"],
        help="Upload a text document that contains information to answer your test question"
    )
    
    # Input fields
    col1, col2 = st.columns(2)
    
    with col1:
        query_text = st.text_area(
            "❓ Enter your test question:",
            placeholder="e.g., 'What are the main benefits of artificial intelligence in healthcare?'",
            help="Ask a specific question that can be answered from the uploaded document",
            height=100,
            disabled=not uploaded_file
        )
    
    with col2:
        response_text = st.text_area(
            "✅ Enter the expected correct answer:",
            placeholder="e.g., 'AI in healthcare provides improved diagnosis accuracy, faster analysis...'",
            help="Provide the correct answer that you expect the RAG system to generate",
            height=100,
            disabled=not uploaded_file
        )
    
    # Initialize evaluator
    if uploaded_file:
        evaluator = RAGEvaluator()
        
        # Validate inputs
        is_valid, error_message = evaluator.validate_inputs(uploaded_file, query_text, response_text)
        
        if not is_valid and (query_text or response_text):
            st.error(error_message)
        
        # Process document preview
        try:
            documents, content = evaluator.process_document(uploaded_file, chunk_strategy)
            display_document_analysis(content, documents)
            
            # Reset file pointer for actual processing
            uploaded_file.seek(0)
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            st.stop()
    
    # Evaluation execution
    if uploaded_file and query_text and response_text:
        if evaluator.validate_inputs(uploaded_file, query_text, response_text)[0]:
            
            # Evaluation button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                evaluate_button = st.button(
                    "🚀 Run RAG Evaluation",
                    type="primary",
                    use_container_width=True,
                    help="Start the comprehensive evaluation process"
                )
            
            if evaluate_button:
                try:
                    with st.spinner("🔄 Processing document and setting up RAG system..."):
                        progress_bar = st.progress(0)
                        
                        # Process document
                        progress_bar.progress(20)
                        documents, content = evaluator.process_document(uploaded_file, chunk_strategy)
                        
                        # Create vector store
                        progress_bar.progress(40)
                        vector_store = evaluator.create_vector_store(documents)
                        
                        # Setup QA chain
                        progress_bar.progress(60)
                        qa_chain = evaluator.setup_qa_chain(chain_type)
                        
                        # Setup evaluation chain
                        progress_bar.progress(80)
                        eval_chain = evaluator.setup_eval_chain()
                        
                        # Run evaluation
                        progress_bar.progress(90)
                        results = evaluator.evaluate_response(query_text, response_text)
                        
                        progress_bar.progress(100)
                        time.sleep(0.5)  # Brief pause to show completion
                        progress_bar.empty()
                    
                    # Display results
                    display_evaluation_results(results)
                    
                    # Action buttons
                    st.markdown("### 🎯 Next Steps")
                    
                    action_cols = st.columns(4)
                    
                    with action_cols[0]:
                        if st.button("🔄 Test Another Question", help="Test with different question"):
                            st.info("💡 Enter a new question and expected answer above!")
                    
                    with action_cols[1]:
                        if st.button("📄 Upload New Document", help="Test with different document"):
                            st.info("💡 Upload a different document to test against!")
                    
                    with action_cols[2]:
                        if st.button("⚙️ Adjust Parameters", help="Modify chunking or chain settings"):
                            st.info("💡 Try different chunking strategies or chain types above!")
                    
                    with action_cols[3]:
                        if st.button("🚀 Custom RAG System", help="Build production RAG system"):
                            st.info("🚀 [Contact GenAI Expertise](https://genaiexpertise.com/contact/) for custom RAG development!")
                
                except Exception as e:
                    st.error(f"❌ Evaluation failed: {str(e)}")
                    st.info("""
                    **Troubleshooting Tips:**
                    - Ensure document contains relevant information
                    - Check that question can be answered from the document  
                    - Verify API key is valid and has sufficient credits
                    - Try with a smaller document or different chunk settings
                    """)
    
    # Add feedback section
    add_feedback_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 Ready for Production RAG Systems?
    
    This evaluation tool demonstrates comprehensive RAG system testing capabilities. We can build:
    
    - **Custom RAG Systems** tailored to your specific documents and use cases
    - **Automated Testing Suites** for continuous quality monitoring
    - **Performance Optimization** for speed and accuracy improvements
    - **Multi-language Support** for global document processing
    - **Enterprise Integration** with your existing knowledge bases
    
    **[Contact GenAI Expertise](https://genaiexpertise.com)** to build your production RAG solution.
    """)

if __name__ == "__main__":
    main()