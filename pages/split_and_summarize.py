import streamlit as st
import os
import time
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import hashlib
from io import StringIO

from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.docstore.document import Document

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
    page_title="Advanced Document Summarizer",
    page_icon="📑",
    layout="wide"
)

# Authentication check
if not check_password():
    st.stop()

class DocumentSummarizer:
    """Enhanced document summarizer with multiple strategies and comprehensive analysis."""
    
    def __init__(self):
        self.api_key = self.get_api_key()
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.max_words = 50000  # Increased from 20,000
        
    def get_api_key(self) -> str:
        """Get API key from environment or secrets."""
        return (
            os.environ.get("OPENAI_API_KEY") or 
            st.secrets.get("APIKEY", {}).get("OPENAI_API_KEY", "")
        )
    
    def validate_file(self, uploaded_file) -> Tuple[bool, str]:
        """Validate uploaded file."""
        if not uploaded_file:
            return False, "Please upload a file to summarize."
        
        # Check file size
        if uploaded_file.size > self.max_file_size:
            return False, f"File too large. Maximum size: {self.max_file_size // (1024*1024)}MB"
        
        # Check file type
        allowed_types = ['txt', 'md', 'csv', 'log']
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in allowed_types:
            return False, f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
        
        return True, ""
    
    def process_file(self, uploaded_file) -> Tuple[str, Dict[str, Any]]:
        """Process uploaded file and extract content with metadata."""
        try:
            # Read file content
            content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            
            # Calculate metadata
            word_count = len(content.split())
            char_count = len(content)
            line_count = len(content.split('\n'))
            paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
            
            # Estimate reading time (200 words per minute average)
            reading_time = max(1, round(word_count / 200))
            
            metadata = {
                'filename': uploaded_file.name,
                'file_size': uploaded_file.size,
                'word_count': word_count,
                'char_count': char_count,
                'line_count': line_count,
                'paragraph_count': paragraph_count,
                'estimated_reading_time': reading_time,
                'file_hash': hashlib.md5(content.encode()).hexdigest()[:8],
                'processing_timestamp': datetime.now().isoformat()
            }
            
            return content, metadata
            
        except UnicodeDecodeError:
            raise Exception("File encoding not supported. Please ensure the file is in UTF-8 format.")
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")
    
    def validate_content(self, content: str) -> Tuple[bool, str]:
        """Validate content length and quality."""
        word_count = len(content.split())
        
        if word_count < 50:
            return False, "Document too short for meaningful summarization (minimum 50 words)."
        
        if word_count > self.max_words:
            return False, f"Document too long ({word_count:,} words). Maximum: {self.max_words:,} words."
        
        # Check for meaningful content
        if len(content.strip()) < 200:
            return False, "Document appears to contain insufficient content for summarization."
        
        return True, ""
    
    def create_text_splitter(self, strategy: str, chunk_size: int, chunk_overlap: int) -> Any:
        """Create appropriate text splitter based on strategy."""
        
        if strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
        elif strategy == "paragraph":
            return RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
        else:  # character
            return CharacterTextSplitter(
                separator="\n",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
    
    def create_custom_prompts(self, summary_type: str) -> Dict[str, PromptTemplate]:
        """Create custom prompts for different summary types."""
        
        prompts = {}
        
        if summary_type == "executive":
            prompts['map'] = PromptTemplate(
                template="""As a business analyst, create a concise summary of the following text focusing on key business insights, decisions, and actionable information:

{text}

EXECUTIVE SUMMARY:""",
                input_variables=["text"]
            )
            
            prompts['reduce'] = PromptTemplate(
                template="""Combine the following executive summaries into a comprehensive business overview:

{text}

Provide a final executive summary that includes:
- Key findings and insights
- Important decisions or recommendations  
- Critical metrics or data points
- Action items or next steps

FINAL EXECUTIVE SUMMARY:""",
                input_variables=["text"]
            )
        
        elif summary_type == "academic":
            prompts['map'] = PromptTemplate(
                template="""As an academic researcher, summarize the following text focusing on main arguments, evidence, methodology, and conclusions:

{text}

ACADEMIC SUMMARY:""",
                input_variables=["text"]
            )
            
            prompts['reduce'] = PromptTemplate(
                template="""Synthesize the following academic summaries into a comprehensive scholarly overview:

{text}

Provide a final academic summary that includes:
- Main thesis and arguments
- Key evidence and methodology
- Important findings and conclusions
- Theoretical implications

FINAL ACADEMIC SUMMARY:""",
                input_variables=["text"]
            )
        
        elif summary_type == "technical":
            prompts['map'] = PromptTemplate(
                template="""As a technical analyst, summarize the following text focusing on technical details, specifications, processes, and implementation aspects:

{text}

TECHNICAL SUMMARY:""",
                input_variables=["text"]
            )
            
            prompts['reduce'] = PromptTemplate(
                template="""Combine the following technical summaries into a comprehensive technical overview:

{text}

Provide a final technical summary that includes:
- Key technical specifications and requirements
- Important processes and methodologies
- Implementation details and considerations
- Technical challenges and solutions

FINAL TECHNICAL SUMMARY:""",
                input_variables=["text"]
            )
        
        else:  # general
            prompts['map'] = PromptTemplate(
                template="""Summarize the following text, capturing the main ideas, key points, and important details:

{text}

SUMMARY:""",
                input_variables=["text"]
            )
            
            prompts['reduce'] = PromptTemplate(
                template="""Combine the following summaries into a comprehensive and coherent final summary:

{text}

Provide a well-structured final summary that captures all the essential information:

FINAL SUMMARY:""",
                input_variables=["text"]
            )
        
        return prompts
    
    def analyze_chunks(self, documents: List[Document]) -> Dict[str, Any]:
        """Analyze document chunks for optimization insights."""
        
        chunk_lengths = [len(doc.page_content) for doc in documents]
        word_counts = [len(doc.page_content.split()) for doc in documents]
        
        analysis = {
            'total_chunks': len(documents),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
            'min_chunk_length': min(chunk_lengths) if chunk_lengths else 0,
            'max_chunk_length': max(chunk_lengths) if chunk_lengths else 0,
            'avg_words_per_chunk': sum(word_counts) / len(word_counts) if word_counts else 0,
            'total_words_in_chunks': sum(word_counts),
            'chunk_size_distribution': {
                'small': len([x for x in chunk_lengths if x < 2000]),
                'medium': len([x for x in chunk_lengths if 2000 <= x < 4000]),
                'large': len([x for x in chunk_lengths if x >= 4000])
            }
        }
        
        return analysis
    
    def summarize_document(
        self, 
        content: str, 
        summary_type: str = "general",
        chunk_strategy: str = "recursive",
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        chain_type: str = "map_reduce"
    ) -> Dict[str, Any]:
        """Perform document summarization with specified parameters."""
        
        if not self.api_key or not self.api_key.startswith("sk-"):
            raise Exception("Valid OpenAI API key required")
        
        try:
            # Initialize LLM
            llm = OpenAI(
                openai_api_key=self.api_key,
                temperature=0.3,  # Slightly higher for more natural summaries
                max_tokens=1500
            )
            
            # Create text splitter
            text_splitter = self.create_text_splitter(chunk_strategy, chunk_size, chunk_overlap)
            
            # Split document
            documents = text_splitter.create_documents([content])
            
            # Analyze chunks
            chunk_analysis = self.analyze_chunks(documents)
            
            # Create custom prompts
            custom_prompts = self.create_custom_prompts(summary_type)
            
            # Create summarization chain
            if chain_type == "map_reduce" and len(custom_prompts) > 1:
                summarize_chain = load_summarize_chain(
                    llm=llm,
                    chain_type=chain_type,
                    map_prompt=custom_prompts.get('map'),
                    combine_prompt=custom_prompts.get('reduce'),
                    verbose=False
                )
            else:
                summarize_chain = load_summarize_chain(
                    llm=llm,
                    chain_type=chain_type,
                    verbose=False
                )
            
            # Generate summary
            summary = summarize_chain.run(documents)
            
            # Calculate summary metrics
            summary_word_count = len(summary.split())
            original_word_count = len(content.split())
            compression_ratio = original_word_count / summary_word_count if summary_word_count > 0 else 0
            
            return {
                'summary': summary,
                'original_word_count': original_word_count,
                'summary_word_count': summary_word_count,
                'compression_ratio': compression_ratio,
                'chunk_analysis': chunk_analysis,
                'parameters': {
                    'summary_type': summary_type,
                    'chunk_strategy': chunk_strategy,
                    'chunk_size': chunk_size,
                    'chunk_overlap': chunk_overlap,
                    'chain_type': chain_type
                },
                'processing_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Summarization failed: {str(e)}")

def display_header():
    """Display the application header."""
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white; text-align: center;'>
        <h1>📑 Advanced Document Summarizer</h1>
        <p>Transform long documents into concise, actionable summaries</p>
        <p><em>Intelligent chunking and multi-strategy summarization for large documents</em></p>
    </div>
    """, unsafe_allow_html=True)

def display_capabilities():
    """Display summarizer capabilities."""
    with st.expander("🚀 Summarization Capabilities", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Document Processing:**
            - Large document handling (up to 50,000 words)
            - Multiple file formats (TXT, MD, CSV, LOG)
            - Intelligent chunking strategies
            - Content validation and optimization
            - Real-time progress tracking
            """)
        
        with col2:
            st.markdown("""
            **🎯 Summary Types:**
            - Executive summaries for business use
            - Academic summaries for research
            - Technical summaries for documentation  
            - General summaries for any content
            - Customizable focus and depth
            """)

def display_file_analysis(metadata: Dict[str, Any]):
    """Display file analysis information."""
    st.markdown("### 📄 Document Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Words", f"{metadata['word_count']:,}")
    
    with col2:
        st.metric("Characters", f"{metadata['char_count']:,}")
    
    with col3:
        st.metric("Paragraphs", metadata['paragraph_count'])
    
    with col4:
        st.metric("Reading Time", f"{metadata['estimated_reading_time']} min")
    
    # Additional details
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("File Size", f"{metadata['file_size']:,} bytes")
        st.metric("Lines", metadata['line_count'])
    
    with col2:
        st.metric("File Hash", metadata['file_hash'])
        processing_time = datetime.fromisoformat(metadata['processing_timestamp'])
        st.metric("Processed At", processing_time.strftime("%H:%M:%S"))

def display_chunk_analysis(chunk_analysis: Dict[str, Any]):
    """Display chunk analysis information."""
    with st.expander("🔍 Chunk Analysis", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Chunks", chunk_analysis['total_chunks'])
            st.metric("Avg Words/Chunk", f"{chunk_analysis['avg_words_per_chunk']:.0f}")
        
        with col2:
            st.metric("Avg Chunk Length", f"{chunk_analysis['avg_chunk_length']:.0f} chars")
            st.metric("Total Words", f"{chunk_analysis['total_words_in_chunks']:,}")
        
        with col3:
            distribution = chunk_analysis['chunk_size_distribution']
            st.markdown("**Chunk Size Distribution:**")
            st.markdown(f"- Small (<2K): {distribution['small']}")
            st.markdown(f"- Medium (2-4K): {distribution['medium']}")
            st.markdown(f"- Large (>4K): {distribution['large']}")

def display_summary_results(results: Dict[str, Any]):
    """Display comprehensive summary results."""
    st.markdown("---")
    st.markdown("## ✨ Summary Results")
    
    # Summary content
    st.markdown("### 📝 Generated Summary")
    st.success(results['summary'])
    
    # Metrics
    st.markdown("### 📊 Summary Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Words", f"{results['original_word_count']:,}")
    
    with col2:
        st.metric("Summary Words", results['summary_word_count'])
    
    with col3:
        compression = results['compression_ratio']
        st.metric("Compression Ratio", f"{compression:.1f}x")
    
    with col4:
        reduction = (1 - 1/compression) * 100 if compression > 0 else 0
        st.metric("Size Reduction", f"{reduction:.1f}%")
    
    # Chunk analysis
    display_chunk_analysis(results['chunk_analysis'])
    
    # Parameters used
    with st.expander("⚙️ Processing Parameters", expanded=False):
        params = results['parameters']
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Summary Type:** {params['summary_type']}")
            st.markdown(f"**Chunk Strategy:** {params['chunk_strategy']}")
            st.markdown(f"**Chain Type:** {params['chain_type']}")
        
        with col2:
            st.markdown(f"**Chunk Size:** {params['chunk_size']:,} characters")
            st.markdown(f"**Chunk Overlap:** {params['chunk_overlap']} characters")
            st.markdown(f"**Processing Time:** {results['processing_time'][:19]}")

def main():
    """Main application logic."""
    
    display_header()
    show_demo_navigation()
    display_api_status()
    
    # Introduction
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎯 Process Large Documents with Intelligence")
        st.markdown("""
        Our advanced document summarizer handles documents that exceed typical AI limits:
        - **Large Document Support** - Process up to 50,000 words efficiently
        - **Intelligent Chunking** - Smart text segmentation for optimal results
        - **Multiple Summary Types** - Executive, academic, technical, or general summaries
        - **Comprehensive Analysis** - Document metrics and processing insights
        - **Quality Optimization** - Multiple strategies for best results
        """)
    
    with col2:
        st.markdown("### 💼 Perfect For:")
        st.markdown("""
        - **Research Papers**
        - **Business Reports**
        - **Technical Documentation**
        - **Legal Documents**
        - **Meeting Transcripts**
        """)
    
    display_capabilities()
    
    # Configuration section
    st.markdown("---")
    st.markdown("## ⚙️ Summarization Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        summary_type = st.selectbox(
            "📋 Summary Type:",
            ["general", "executive", "academic", "technical"],
            help="Choose the type of summary based on your needs and audience"
        )
        
        chunk_strategy = st.selectbox(
            "📄 Chunking Strategy:",
            ["recursive", "paragraph", "character"],
            help="Recursive: Smart splitting. Paragraph: Split on paragraphs. Character: Fixed character splitting."
        )
        
        chain_type = st.selectbox(
            "🔗 Processing Method:",
            ["map_reduce", "stuff", "refine"],
            help="Map_reduce: Process chunks separately then combine. Stuff: Combine all at once. Refine: Iterative improvement."
        )
    
    with col2:
        chunk_size = st.slider(
            "📏 Chunk Size:",
            min_value=1000,
            max_value=8000,
            value=4000,
            step=500,
            help="Size of each text chunk in characters"
        )
        
        chunk_overlap = st.slider(
            "🔄 Chunk Overlap:",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Character overlap between chunks for context preservation"
        )
        
        st.markdown("""
        **💡 Configuration Tips:**
        - **Executive**: Focus on business insights and decisions
        - **Academic**: Emphasize research and methodology
        - **Technical**: Highlight specifications and processes
        - **General**: Balanced overview of all content
        """)
    
    # File upload
    st.markdown("---")
    st.markdown("## 📁 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a document to summarize:",
        type=['txt', 'md', 'csv', 'log'],
        help="Upload text files up to 50MB. Supported formats: TXT, MD, CSV, LOG"
    )
    
    if uploaded_file:
        summarizer = DocumentSummarizer()
        
        # Validate file
        is_valid_file, file_error = summarizer.validate_file(uploaded_file)
        
        if not is_valid_file:
            st.error(file_error)
            st.stop()
        
        try:
            # Process file
            content, metadata = summarizer.process_file(uploaded_file)
            
            # Display file analysis
            display_file_analysis(metadata)
            
            # Validate content
            is_valid_content, content_error = summarizer.validate_content(content)
            
            if not is_valid_content:
                st.error(content_error)
                st.stop()
            
            # Document preview
            with st.expander("📖 Document Preview", expanded=False):
                preview_length = min(1000, len(content))
                st.text_area(
                    "First 1000 characters:",
                    content[:preview_length] + ("..." if len(content) > preview_length else ""),
                    height=200,
                    disabled=True
                )
            
            # Summarize button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                summarize_button = st.button(
                    "🚀 Generate Summary",
                    type="primary",
                    use_container_width=True,
                    help="Process document and generate intelligent summary"
                )
            
            if summarize_button:
                try:
                    with st.spinner("🤖 AI is analyzing and summarizing your document..."):
                        progress_bar = st.progress(0)
                        
                        # Simulate processing steps
                        steps = [
                            ("Analyzing document structure...", 20),
                            ("Creating intelligent chunks...", 40),
                            ("Processing individual sections...", 60),
                            ("Combining insights...", 80),
                            ("Finalizing summary...", 100)
                        ]
                        
                        for step_text, progress in steps:
                            st.text(step_text)
                            progress_bar.progress(progress)
                            time.sleep(0.5)
                        
                        # Perform actual summarization
                        results = summarizer.summarize_document(
                            content=content,
                            summary_type=summary_type,
                            chunk_strategy=chunk_strategy,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            chain_type=chain_type
                        )
                        
                        progress_bar.empty()
                        st.success("✅ Summarization completed!")
                    
                    # Display results
                    display_summary_results(results)
                    
                    # Action buttons
                    st.markdown("### 🎯 Next Steps")
                    
                    action_cols = st.columns(4)
                    
                    with action_cols[0]:
                        if st.button("📋 Copy Summary", help="Copy summary text"):
                            st.code(results['summary'])
                            st.success("✅ Summary ready to copy!")
                    
                    with action_cols[1]:
                        if st.button("🔄 Try Different Settings", help="Modify configuration"):
                            st.info("💡 Adjust the configuration settings above and summarize again!")
                    
                    with action_cols[2]:
                        if st.button("📄 Upload New Document", help="Process different document"):
                            st.info("💡 Upload a different document to summarize!")
                    
                    with action_cols[3]:
                        if st.button("🚀 Custom Solution", help="Build enterprise summarization system"):
                            st.info("🚀 [Contact GenAI Expertise](https://genaiexpertise.com/contact/) for custom document processing solutions!")
                
                except Exception as e:
                    st.error(f"❌ Summarization failed: {str(e)}")
                    st.info("""
                    **Troubleshooting Tips:**
                    - Try reducing chunk size for very long documents
                    - Use different chunking strategy for structured content
                    - Check document format and encoding
                    - Ensure stable internet connection
                    """)
        
        except Exception as e:
            st.error(f"❌ File processing failed: {str(e)}")
    
    else:
        st.info("👆 Upload a document above to begin summarization!")
    
    # Add feedback section
    add_feedback_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 Ready for Enterprise Document Processing?
    
    This summarizer demonstrates advanced document processing capabilities. We can build:
    
    - **Batch Processing Systems** - Process hundreds of documents automatically
    - **Custom Content Types** - Handle PDFs, Word docs, presentations, and more
    - **Industry-Specific Models** - Trained for legal, medical, financial, or technical content
    - **API Integration** - Connect with your existing document management systems
    - **Advanced Analytics** - Document classification, topic extraction, and insight generation
    
    **[Contact GenAI Expertise](https://genaiexpertise.com)** to build your enterprise document intelligence solution.
    """)

if __name__ == "__main__":
    main()