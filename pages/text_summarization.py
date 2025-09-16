import streamlit as st
import os
import time
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import re

from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
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
    page_title="Quick Text Summarizer",
    page_icon="⚡",
    layout="wide"
)

# Authentication check
if not check_password():
    st.stop()

class QuickTextSummarizer:
    """Enhanced quick text summarizer with multiple options and intelligent processing."""
    
    def __init__(self):
        self.api_key = self.get_api_key()
        self.max_words = 10000  # Reasonable limit for quick processing
        
    def get_api_key(self) -> str:
        """Get API key from environment or secrets."""
        return (
            os.environ.get("OPENAI_API_KEY") or 
            st.secrets.get("APIKEY", {}).get("OPENAI_API_KEY", "")
        )
    
    def validate_text(self, text: str) -> Tuple[bool, str]:
        """Validate input text for summarization."""
        if not text.strip():
            return False, "Please enter text to summarize."
        
        if len(text.strip()) < 100:
            return False, "Text should be at least 100 characters for meaningful summarization."
        
        word_count = len(text.split())
        if word_count > self.max_words:
            return False, f"Text too long ({word_count:,} words). Maximum: {self.max_words:,} words."
        
        if word_count < 50:
            return False, "Text should contain at least 50 words for effective summarization."
        
        return True, ""
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text properties for optimization."""
        
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(re.findall(r'[.!?]+', text))
        paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
        
        # Estimate reading time (200 words per minute)
        reading_time = max(1, round(word_count / 200))
        
        # Detect text type based on content
        text_type = self.detect_text_type(text)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'reading_time': reading_time,
            'text_type': text_type,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def detect_text_type(self, text: str) -> str:
        """Detect the type of text for optimized processing."""
        
        text_lower = text.lower()
        
        # Business/formal indicators
        business_keywords = ['meeting', 'agenda', 'proposal', 'report', 'analysis', 'strategic', 'quarterly', 'revenue']
        business_score = sum(1 for keyword in business_keywords if keyword in text_lower)
        
        # Academic indicators
        academic_keywords = ['research', 'study', 'hypothesis', 'methodology', 'conclusion', 'abstract', 'findings']
        academic_score = sum(1 for keyword in academic_keywords if keyword in text_lower)
        
        # Technical indicators
        technical_keywords = ['system', 'implementation', 'algorithm', 'configuration', 'process', 'technical', 'specification']
        technical_score = sum(1 for keyword in technical_keywords if keyword in text_lower)
        
        # News/article indicators
        news_keywords = ['reported', 'according to', 'sources', 'announced', 'breaking', 'update', 'statement']
        news_score = sum(1 for keyword in news_keywords if keyword in text_lower)
        
        scores = {
            'business': business_score,
            'academic': academic_score,
            'technical': technical_score,
            'news': news_score
        }
        
        max_type = max(scores.items(), key=lambda x: x[1])
        return max_type[0] if max_type[1] > 0 else 'general'
    
    def create_optimized_prompt(self, summary_length: str, text_type: str) -> PromptTemplate:
        """Create optimized prompt based on summary requirements."""
        
        length_instructions = {
            'brief': "Create a very concise summary in 2-3 sentences that captures only the most essential points.",
            'standard': "Create a clear and comprehensive summary that covers the main points and key details.",
            'detailed': "Create a thorough summary that includes main points, supporting details, and important context."
        }
        
        type_instructions = {
            'business': "Focus on key business insights, decisions, metrics, and actionable information.",
            'academic': "Emphasize research findings, methodology, conclusions, and theoretical contributions.",
            'technical': "Highlight technical specifications, processes, implementations, and system details.",
            'news': "Focus on key facts, who/what/when/where, and significant developments or impacts.",
            'general': "Capture the main ideas, key points, and most important information."
        }
        
        template = f"""You are an expert text analyst specializing in creating high-quality summaries.

Your task is to summarize the following text with these specifications:
- Length: {length_instructions.get(summary_length, length_instructions['standard'])}
- Focus: {type_instructions.get(text_type, type_instructions['general'])}

Requirements:
1. Maintain the original meaning and context
2. Use clear, professional language
3. Organize information logically
4. Include the most important points first
5. Ensure the summary is self-contained

Text to summarize:
{{text}}

Summary:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["text"]
        )
    
    def create_text_splitter(self, text: str, strategy: str) -> Any:
        """Create appropriate text splitter based on text length and strategy."""
        
        word_count = len(text.split())
        
        # Determine chunk size based on text length
        if word_count < 1000:
            chunk_size = 2000
            chunk_overlap = 100
        elif word_count < 3000:
            chunk_size = 3000
            chunk_overlap = 200
        else:
            chunk_size = 4000
            chunk_overlap = 300
        
        if strategy == "recursive":
            return RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
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
    
    def summarize_text(
        self, 
        text: str, 
        summary_length: str = "standard",
        processing_strategy: str = "smart",
        chain_type: str = "map_reduce"
    ) -> Dict[str, Any]:
        """Generate summary with specified parameters."""
        
        if not self.api_key or not self.api_key.startswith("sk-"):
            raise Exception("Valid OpenAI API key required")
        
        try:
            # Analyze text
            text_analysis = self.analyze_text(text)
            
            # Configure LLM based on summary length
            temperature_map = {'brief': 0.1, 'standard': 0.3, 'detailed': 0.4}
            max_tokens_map = {'brief': 200, 'standard': 500, 'detailed': 800}
            
            llm = OpenAI(
                openai_api_key=self.api_key,
                temperature=temperature_map.get(summary_length, 0.3),
                max_tokens=max_tokens_map.get(summary_length, 500)
            )
            
            # Determine processing approach
            word_count = text_analysis['word_count']
            
            if processing_strategy == "smart":
                # Use direct summarization for shorter texts, chunking for longer ones
                if word_count < 2000:
                    # Direct summarization
                    prompt = self.create_optimized_prompt(summary_length, text_analysis['text_type'])
                    summary = llm(prompt.format(text=text))
                else:
                    # Chunked summarization
                    text_splitter = self.create_text_splitter(text, "recursive")
                    texts = text_splitter.split_text(text)
                    docs = [Document(page_content=t) for t in texts]
                    
                    # Use custom prompt for map_reduce if available
                    if chain_type == "map_reduce":
                        map_prompt = self.create_optimized_prompt(summary_length, text_analysis['text_type'])
                        chain = load_summarize_chain(
                            llm,
                            chain_type=chain_type,
                            map_prompt=map_prompt,
                            verbose=False
                        )
                    else:
                        chain = load_summarize_chain(llm, chain_type=chain_type, verbose=False)
                    
                    summary = chain.run(docs)
            
            else:  # always_chunk
                text_splitter = self.create_text_splitter(text, "character")
                texts = text_splitter.split_text(text)
                docs = [Document(page_content=t) for t in texts]
                
                chain = load_summarize_chain(llm, chain_type=chain_type, verbose=False)
                summary = chain.run(docs)
            
            # Calculate summary metrics
            summary_word_count = len(summary.split())
            compression_ratio = word_count / summary_word_count if summary_word_count > 0 else 0
            
            return {
                'summary': summary.strip(),
                'text_analysis': text_analysis,
                'summary_metrics': {
                    'summary_word_count': summary_word_count,
                    'compression_ratio': compression_ratio,
                    'processing_method': 'direct' if word_count < 2000 and processing_strategy == "smart" else 'chunked'
                },
                'parameters': {
                    'summary_length': summary_length,
                    'processing_strategy': processing_strategy,
                    'chain_type': chain_type,
                    'detected_text_type': text_analysis['text_type']
                },
                'processing_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Summarization failed: {str(e)}")

def display_header():
    """Display the application header."""
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white; text-align: center;'>
        <h1>⚡ Quick Text Summarizer</h1>
        <p>Instant intelligent summaries for any text content</p>
        <p><em>Fast, efficient, and optimized for immediate results</em></p>
    </div>
    """, unsafe_allow_html=True)

def display_sample_texts():
    """Display sample texts for quick testing."""
    with st.expander("📝 Try Sample Texts", expanded=False):
        
        samples = {
            "Business Report": """The Q3 financial results show significant growth across all business units. Revenue increased by 23% compared to the same quarter last year, reaching $45.2 million. The marketing department's digital transformation initiative contributed to a 35% increase in online customer acquisition. Our customer satisfaction scores improved to 4.7/5.0, up from 4.2/5.0 in Q2. The new product line launched in September exceeded initial projections by 18%, generating $3.8 million in sales. Operating expenses were kept under control at 15% below budget, primarily due to automated process improvements. The sales team exceeded their quarterly targets by 12%, with particular strength in the enterprise segment. Looking ahead to Q4, we expect continued growth momentum with the holiday season and planned product releases.""",
            
            "Technical Article": """Machine learning algorithms have revolutionized data processing in modern applications. Deep neural networks, particularly convolutional neural networks (CNNs), excel at image recognition tasks with accuracy rates exceeding 95%. The implementation requires careful consideration of hyperparameters including learning rate, batch size, and network architecture. GPU acceleration significantly improves training times, reducing typical training periods from weeks to days. Data preprocessing steps include normalization, augmentation, and validation splitting. Model evaluation uses cross-validation techniques to ensure generalization. Recent developments in transformer architectures have shown promising results for natural language processing tasks. The attention mechanism allows models to focus on relevant input segments, improving performance on complex linguistic tasks. Production deployment requires considerations for model versioning, A/B testing, and real-time inference capabilities.""",
            
            "News Article": """The city council announced today a major infrastructure investment plan worth $150 million over the next five years. The comprehensive program will focus on upgrading public transportation, improving road conditions, and expanding digital connectivity throughout the metropolitan area. Mayor Johnson stated that the initiative will create approximately 2,500 jobs and significantly improve quality of life for residents. The plan includes $60 million for transit improvements, including new bus routes and modernized stations. Road repairs will receive $45 million, addressing the backlog of maintenance needs identified in last year's infrastructure assessment. Broadband expansion will account for $30 million, aiming to provide high-speed internet access to underserved neighborhoods. The remaining $15 million is allocated for smart city technologies, including traffic management systems and digital public services. Funding will come from a combination of federal grants, state allocations, and municipal bonds."""
        }
        
        for title, sample in samples.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{title}:**")
                st.text_area(f"{title} preview", sample[:200] + "...", height=60, disabled=True, label_visibility="collapsed")
            with col2:
                if st.button(f"Use {title}", key=f"sample_{title}", use_container_width=True):
                    st.session_state.sample_text = sample
                    st.rerun()

def display_text_analysis(analysis: Dict[str, Any]):
    """Display text analysis information."""
    st.markdown("### 📊 Text Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Words", f"{analysis['word_count']:,}")
    
    with col2:
        st.metric("Characters", f"{analysis['char_count']:,}")
    
    with col3:
        st.metric("Sentences", analysis['sentence_count'])
    
    with col4:
        st.metric("Reading Time", f"{analysis['reading_time']} min")
    
    # Additional details
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Paragraphs", analysis['paragraph_count'])
        
    with col2:
        text_type = analysis['text_type'].title()
        st.metric("Detected Type", text_type)

def display_summary_results(results: Dict[str, Any]):
    """Display comprehensive summary results."""
    st.markdown("---")
    st.markdown("## ✨ Summary Results")
    
    # Summary content
    summary_length = results['parameters']['summary_length']
    if summary_length == 'brief':
        st.markdown("### ⚡ Brief Summary")
    elif summary_length == 'detailed':
        st.markdown("### 📋 Detailed Summary")
    else:
        st.markdown("### 📝 Standard Summary")
    
    st.success(results['summary'])
    
    # Metrics
    st.markdown("### 📊 Summary Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    text_analysis = results['text_analysis']
    summary_metrics = results['summary_metrics']
    
    with col1:
        st.metric("Original Words", f"{text_analysis['word_count']:,}")
    
    with col2:
        st.metric("Summary Words", summary_metrics['summary_word_count'])
    
    with col3:
        compression = summary_metrics['compression_ratio']
        st.metric("Compression", f"{compression:.1f}x")
    
    with col4:
        processing_method = summary_metrics['processing_method'].title()
        st.metric("Method", processing_method)
    
    # Processing details
    with st.expander("⚙️ Processing Details", expanded=False):
        params = results['parameters']
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Summary Length:** {params['summary_length'].title()}")
            st.markdown(f"**Processing Strategy:** {params['processing_strategy'].title()}")
            st.markdown(f"**Chain Type:** {params['chain_type']}")
        
        with col2:
            st.markdown(f"**Text Type:** {params['detected_text_type'].title()}")
            st.markdown(f"**Processing Method:** {summary_metrics['processing_method'].title()}")
            processing_time = results['processing_timestamp'][:19]
            st.markdown(f"**Completed At:** {processing_time}")

def main():
    """Main application logic."""
    
    display_header()
    show_demo_navigation()
    display_api_status()
    
    # Introduction
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎯 Instant Text Summarization")
        st.markdown("""
        Get quick, intelligent summaries of any text content with our optimized AI system:
        - **Lightning Fast** - Instant processing for immediate results
        - **Smart Processing** - Automatically adapts to text type and length
        - **Multiple Lengths** - Brief, standard, or detailed summaries
        - **Intelligent Analysis** - Detects content type for optimized results
        - **Quality Focused** - Professional-grade summarization algorithms
        """)
    
    with col2:
        st.markdown("### 💼 Perfect For:")
        st.markdown("""
        - **Meeting Notes**
        - **Article Summaries**
        - **Report Briefs**
        - **Email Digests**
        - **Content Curation**
        """)
    
    # Sample texts
    display_sample_texts()
    
    # Configuration
    st.markdown("---")
    st.markdown("## ⚙️ Summary Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        summary_length = st.selectbox(
            "📏 Summary Length:",
            ["brief", "standard", "detailed"],
            index=1,
            help="Brief: 2-3 sentences. Standard: Balanced coverage. Detailed: Comprehensive summary."
        )
    
    with col2:
        processing_strategy = st.selectbox(
            "🧠 Processing Strategy:",
            ["smart", "always_chunk"],
            help="Smart: Automatically choose best method. Always_chunk: Always use chunking approach."
        )
    
    with col3:
        chain_type = st.selectbox(
            "🔗 Chain Type:",
            ["map_reduce", "stuff", "refine"],
            help="Map_reduce: Best for longer texts. Stuff: Simple combination. Refine: Iterative improvement."
        )
    
    # Text input
    st.markdown("---")
    st.markdown("## 📝 Enter Text to Summarize")
    
    # Check for sample text
    default_text = st.session_state.get("sample_text", "")
    
    text_input = st.text_area(
        "Text Content:",
        value=default_text,
        placeholder="""Paste your text here for instant summarization...

Examples:
• Business reports and meeting notes
• Research articles and papers  
• News articles and blog posts
• Technical documentation
• Email threads and conversations

Try the sample texts above or paste your own content!""",
        height=200,
        help="Enter any text content up to 10,000 words for summarization."
    )
    
    # Clear sample text from session state
    if "sample_text" in st.session_state:
        del st.session_state.sample_text
    
    # Text analysis
    if text_input:
        summarizer = QuickTextSummarizer()
        
        # Validate text
        is_valid, error_message = summarizer.validate_text(text_input)
        
        if not is_valid:
            st.error(error_message)
        else:
            # Display text analysis
            analysis = summarizer.analyze_text(text_input)
            display_text_analysis(analysis)
            
            # Summarize button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                summarize_button = st.button(
                    "⚡ Generate Summary",
                    type="primary",
                    use_container_width=True,
                    help="Create instant intelligent summary"
                )
            
            if summarize_button:
                try:
                    with st.spinner("⚡ Creating your summary..."):
                        # Quick progress indication
                        progress_bar = st.progress(0)
                        for i in range(0, 101, 20):
                            progress_bar.progress(i)
                            time.sleep(0.1)
                        
                        # Generate summary
                        results = summarizer.summarize_text(
                            text=text_input,
                            summary_length=summary_length,
                            processing_strategy=processing_strategy,
                            chain_type=chain_type
                        )
                        
                        progress_bar.empty()
                        st.success("✅ Summary generated!")
                    
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
                        if st.button("🔄 Try Different Length", help="Change summary length"):
                            st.info("💡 Adjust the length setting above and summarize again!")
                    
                    with action_cols[2]:
                        if st.button("📝 New Text", help="Clear and enter new text"):
                            st.info("💡 Clear the text area above and enter new content!")
                    
                    with action_cols[3]:
                        if st.button("🚀 Custom System", help="Build enterprise summarization"):
                            st.info("🚀 [Contact GenAI Expertise](https://genaiexpertise.com/contact/) for custom summarization solutions!")
                
                except Exception as e:
                    st.error(f"❌ Summarization failed: {str(e)}")
                    st.info("""
                    **Troubleshooting Tips:**
                    - Ensure text contains meaningful content
                    - Try with shorter text if very long
                    - Check internet connection stability
                    - Verify API key configuration
                    """)
    
    else:
        st.info("👆 Enter text above to generate an instant summary!")
    
    # Add feedback section
    add_feedback_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 Ready for Production Summarization?
    
    This quick summarizer showcases efficient text processing capabilities. We can build:
    
    - **Real-time Processing** - Instant summaries integrated into your workflows
    - **Bulk Summarization** - Process hundreds of documents simultaneously
    - **Custom Summary Types** - Tailored for your specific industry and needs
    - **API Integration** - Connect with your existing content management systems
    - **Multi-language Support** - Summarization in dozens of languages
    
    **[Contact GenAI Expertise](https://genaiexpertise.com)** to build your enterprise text processing solution.
    """)

if __name__ == "__main__":
    main()