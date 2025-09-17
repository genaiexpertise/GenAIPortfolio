import streamlit as st
import validators
import os
import time
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import re
from urllib.parse import urlparse

from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

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
    page_title="Content Summarizer",
    page_icon="📺",
    layout="wide"
)

# Authentication check
if not check_password():
    st.stop()

class ContentSummarizer:
    """Enhanced content summarizer for YouTube videos and web articles."""
    
    def __init__(self):
        self.supported_platforms = {
            'youtube': ['youtube.com', 'youtu.be', 'm.youtube.com'],
            'video': ['vimeo.com', 'dailymotion.com'],
            'news': ['cnn.com', 'bbc.com', 'reuters.com', 'bloomberg.com'],
            'tech': ['techcrunch.com', 'arstechnica.com', 'wired.com', 'verge.com'],
            'academic': ['arxiv.org', 'scholar.google.com', 'researchgate.net'],
            'business': ['forbes.com', 'wsj.com', 'ft.com', 'economist.com']
        }
        
    def validate_url(self, url: str) -> Tuple[bool, str, str]:
        """Validate URL and determine content type."""
        if not url.strip():
            return False, "Please enter a URL to summarize.", "unknown"
        
        if not validators.url(url):
            return False, "Please enter a valid URL (including https:// or http://)", "unknown"
        
        # Determine content type
        content_type = self.detect_content_type(url)
        
        return True, "", content_type
    
    def detect_content_type(self, url: str) -> str:
        """Detect content type based on URL."""
        url_lower = url.lower()
        
        for content_type, domains in self.supported_platforms.items():
            if any(domain in url_lower for domain in domains):
                return content_type
        
        return "website"
    
    def validate_api_key(self, api_key: str, api_type: str) -> bool:
        """Validate API key format."""
        if not api_key.strip():
            return False
        
        if api_type == "groq":
            return api_key.startswith("gsk_")
        elif api_type == "openai":
            return api_key.startswith("sk-")
        
        return len(api_key) > 20  # Basic length check
    
    def extract_video_info(self, url: str) -> Dict[str, Any]:
        """Extract basic video information from URL."""
        video_info = {'platform': 'unknown', 'video_id': None}
        
        if 'youtube.com' in url or 'youtu.be' in url:
            video_info['platform'] = 'youtube'
            if 'youtu.be' in url:
                video_info['video_id'] = url.split('/')[-1].split('?')[0]
            elif 'watch?v=' in url:
                video_info['video_id'] = url.split('watch?v=')[1].split('&')[0]
        
        return video_info
    
    def create_enhanced_prompt(self, content_type: str, summary_style: str) -> PromptTemplate:
        """Create enhanced prompts based on content type and style."""
        
        base_instructions = {
            'youtube': "You are analyzing a YouTube video transcript. Focus on the main topics discussed, key insights shared, and actionable takeaways.",
            'website': "You are analyzing web content. Extract the main arguments, key information, and important details.",
            'news': "You are analyzing a news article. Focus on the who, what, when, where, why, and the significance of the events.",
            'tech': "You are analyzing technical content. Highlight innovations, technical details, industry implications, and future trends.",
            'academic': "You are analyzing academic content. Focus on research findings, methodology, conclusions, and scholarly contributions.",
            'business': "You are analyzing business content. Emphasize market insights, business strategies, financial implications, and industry impact."
        }
        
        style_instructions = {
            'brief': "Create a concise summary in 2-3 paragraphs focusing on the most essential points.",
            'detailed': "Create a comprehensive summary that covers all major points with supporting details and context.",
            'bullet': "Create a structured summary using bullet points for easy scanning and reference.",
            'executive': "Create an executive summary focusing on key insights, implications, and actionable recommendations."
        }
        
        content_instruction = base_instructions.get(content_type, base_instructions['website'])
        style_instruction = style_instructions.get(summary_style, style_instructions['detailed'])
        
        template = f"""{content_instruction}
        
{style_instruction}

CONTENT ANALYSIS REQUIREMENTS:
1. Main Topic/Theme: What is this content primarily about?
2. Key Points: What are the most important ideas or information presented?
3. Supporting Details: What evidence, examples, or explanations are provided?
4. Conclusions/Takeaways: What should the reader understand or do after consuming this content?
5. Context/Significance: Why is this information important or relevant?

Please structure your response clearly and ensure it captures the essence of the original content while being accessible to someone who hasn't seen/read it.

Content to analyze:
{"{text}"}

Summary:"""

        return PromptTemplate(template=template, input_variables=["text"])
    
    def load_content(self, url: str, content_type: str) -> List[Any]:
        """Load content from URL based on content type."""
        try:
            if content_type == "youtube" or "youtube.com" in url:
                loader = YoutubeLoader.from_youtube_url(
                    url, 
                    add_video_info=True,
                    language=["en", "auto"],
                    translation="en"
                )
            else:
                # Enhanced headers for better compatibility
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                }
                
                loader = UnstructuredURLLoader(
                    urls=[url],
                    ssl_verify=False,
                    headers=headers
                )
            
            documents = loader.load()
            
            if not documents:
                raise Exception("No content could be extracted from the URL")
            
            return documents
            
        except Exception as e:
            raise Exception(f"Failed to load content: {str(e)}")
    
    def analyze_content(self, documents: List[Any]) -> Dict[str, Any]:
        """Analyze loaded content for metrics."""
        if not documents:
            return {}
        
        total_text = " ".join([doc.page_content for doc in documents])
        word_count = len(total_text.split())
        char_count = len(total_text)
        
        # Estimate reading/viewing time
        if word_count > 0:
            reading_time = max(1, round(word_count / 200))  # 200 words per minute
        else:
            reading_time = 0
        
        # Extract metadata if available
        metadata = documents[0].metadata if documents else {}
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'reading_time': reading_time,
            'document_count': len(documents),
            'metadata': metadata,
            'has_content': word_count > 0
        }
    
    def summarize_content(
        self, 
        url: str, 
        api_key: str, 
        api_type: str = "groq",
        summary_style: str = "detailed",
        model_name: str = "mixtral-8x7b-32768"
    ) -> Dict[str, Any]:
        """Perform content summarization with comprehensive analysis."""
        
        try:
            # Validate inputs
            is_valid, error_msg, content_type = self.validate_url(url)
            if not is_valid:
                raise Exception(error_msg)
            
            if not self.validate_api_key(api_key, api_type):
                raise Exception(f"Invalid {api_type.upper()} API key format")
            
            # Load content
            documents = self.load_content(url, content_type)
            
            # Analyze content
            content_analysis = self.analyze_content(documents)
            
            if not content_analysis.get('has_content'):
                raise Exception("No readable content found at the provided URL")
            
            # Initialize LLM
            if api_type == "groq":
                llm = ChatGroq(
                    model=model_name,
                    groq_api_key=api_key,
                    temperature=0.3,
                    max_tokens=2048
                )
            else:  # openai
                llm = OpenAI(
                    openai_api_key=api_key,
                    temperature=0.3,
                    max_tokens=1500
                )
            
            # Create enhanced prompt
            prompt = self.create_enhanced_prompt(content_type, summary_style)
            
            # Create summarization chain
            chain = load_summarize_chain(
                llm=llm,
                chain_type="stuff",
                prompt=prompt,
                verbose=False
            )
            
            # Generate summary
            start_time = time.time()
            summary = chain.run(documents)
            processing_time = time.time() - start_time
            
            # Calculate summary metrics
            summary_word_count = len(summary.split())
            compression_ratio = content_analysis['word_count'] / summary_word_count if summary_word_count > 0 else 0
            
            return {
                'url': url,
                'content_type': content_type,
                'summary': summary.strip(),
                'content_analysis': content_analysis,
                'summary_metrics': {
                    'summary_word_count': summary_word_count,
                    'compression_ratio': compression_ratio,
                    'processing_time': processing_time
                },
                'parameters': {
                    'api_type': api_type,
                    'model_name': model_name,
                    'summary_style': summary_style
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Summarization failed: {str(e)}")

def display_header():
    """Display the application header."""
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white; text-align: center;'>
        <h1>📺 YouTube & Web Content Summarizer</h1>
        <p>Transform hours of content into actionable insights in seconds</p>
        <p><em>AI-powered summarization for videos, articles, and web content</em></p>
    </div>
    """, unsafe_allow_html=True)

def display_supported_platforms():
    """Display supported platforms and content types."""
    with st.expander("🌐 Supported Platforms & Content Types", expanded=False):
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📺 Video Platforms:**
            - YouTube (all video types)
            - YouTube Music
            - Vimeo
            - Dailymotion
            """)
        
        with col2:
            st.markdown("""
            **📰 News & Media:**
            - News websites
            - Blog articles
            - Press releases
            - Online magazines
            """)
        
        with col3:
            st.markdown("""
            **📊 Specialized Content:**
            - Technical documentation
            - Academic papers
            - Business reports
            - Research articles
            """)

def display_sample_urls():
    """Display sample URLs for testing."""
    with st.expander("🔗 Try Sample URLs", expanded=False):
        
        samples = {
            "Tech Tutorial": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "Business Article": "https://www.forbes.com/",
            "News Article": "https://www.bbc.com/news"
        }
        
        col1, col2, col3 = st.columns(3)
        
        for i, (title, url) in enumerate(samples.items()):
            with [col1, col2, col3][i]:
                st.markdown(f"**{title}:**")
                st.text(url[:40] + "..." if len(url) > 40 else url)
                if st.button(f"Use {title}", key=f"sample_{i}", use_container_width=True):
                    st.session_state.sample_url = url
                    st.rerun()

def display_content_analysis(analysis: Dict[str, Any], content_type: str):
    """Display content analysis information."""
    st.markdown("### 📊 Content Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Content Type", content_type.title())
    
    with col2:
        st.metric("Words", f"{analysis.get('word_count', 0):,}")
    
    with col3:
        st.metric("Characters", f"{analysis.get('char_count', 0):,}")
    
    with col4:
        reading_time = analysis.get('reading_time', 0)
        st.metric("Est. Time", f"{reading_time} min")
    
    # Metadata display
    metadata = analysis.get('metadata', {})
    if metadata:
        with st.expander("📋 Content Metadata", expanded=False):
            for key, value in metadata.items():
                if value and str(value).strip():
                    st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")

def display_summary_results(results: Dict[str, Any]):
    """Display comprehensive summary results."""
    st.markdown("---")
    st.markdown("## ✨ Summary Results")
    
    # URL and content type info
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**📍 Source:** {results['url']}")
        
    with col2:
        content_type = results['content_type'].title()
        st.metric("Content Type", content_type)
    
    # Summary display
    st.markdown("### 📝 Generated Summary")
    st.success(results['summary'])
    
    # Metrics
    st.markdown("### 📊 Summary Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    content_analysis = results['content_analysis']
    summary_metrics = results['summary_metrics']
    
    with col1:
        st.metric("Original Words", f"{content_analysis['word_count']:,}")
    
    with col2:
        st.metric("Summary Words", summary_metrics['summary_word_count'])
    
    with col3:
        compression = summary_metrics['compression_ratio']
        st.metric("Compression", f"{compression:.1f}x")
    
    with col4:
        processing_time = summary_metrics['processing_time']
        st.metric("Processing Time", f"{processing_time:.1f}s")
    
    # Processing details
    with st.expander("⚙️ Processing Details", expanded=False):
        params = results['parameters']
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**API Used:** {params['api_type'].upper()}")
            st.markdown(f"**Model:** {params['model_name']}")
            st.markdown(f"**Summary Style:** {params['summary_style'].title()}")
        
        with col2:
            st.markdown(f"**Documents Processed:** {content_analysis['document_count']}")
            timestamp = results['timestamp'][:19].replace('T', ' ')
            st.markdown(f"**Processed At:** {timestamp}")

def main():
    """Main application logic."""
    
    display_header()
    show_demo_navigation()
    display_api_status()
    
    # Introduction
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎯 Automate Your Content Consumption")
        st.markdown("""
        Transform hours of video and reading into actionable insights:
        - **YouTube Videos** - Extract key points from tutorials, lectures, and presentations
        - **Web Articles** - Summarize blog posts, news articles, and research papers
        - **Multiple Formats** - Support for various content types and platforms
        - **Smart Analysis** - Content-aware summarization with context preservation
        - **Fast Processing** - Get summaries in seconds, not hours
        """)
    
    with col2:
        st.markdown("### 💼 Perfect For:")
        st.markdown("""
        - **Research & Learning**
        - **Content Curation**
        - **Market Intelligence**
        - **Educational Content**
        - **Business Analysis**
        """)
    
    display_supported_platforms()
    display_sample_urls()
    
    # API Configuration
    st.markdown("---")
    st.markdown("## ⚙️ API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_type = st.selectbox(
            "🔑 Select AI Provider:",
            ["groq", "openai"],
            help="Choose your preferred AI service provider"
        )
        
        if api_type == "groq":
            api_key = st.text_input(
                "Groq API Key:",
                type="password",
                help="Enter your Groq API key (starts with gsk_)"
            )
            model_options = ["mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"]
        else:
            api_key = st.text_input(
                "OpenAI API Key:", 
                type="password",
                help="Enter your OpenAI API key (starts with sk-)"
            )
            model_options = ["gpt-3.5-turbo", "gpt-4"]
        
        model_name = st.selectbox(
            "🤖 Select Model:",
            model_options,
            help="Choose the AI model for summarization"
        )
    
    with col2:
        summary_style = st.selectbox(
            "📋 Summary Style:",
            ["detailed", "brief", "bullet", "executive"],
            help="Choose the format and depth of your summary"
        )
        
        st.markdown("""
        **📝 Summary Styles:**
        - **Brief**: 2-3 paragraphs with key points
        - **Detailed**: Comprehensive coverage with context
        - **Bullet**: Structured points for easy scanning
        - **Executive**: Business-focused insights and recommendations
        """)
    
    # Content Input
    st.markdown("---")
    st.markdown("## 🔗 Enter Content URL")
    
    # Check for sample URL
    default_url = st.session_state.get("sample_url", "")
    
    url_input = st.text_input(
        "Content URL:",
        value=default_url,
        placeholder="https://www.youtube.com/watch?v=... or https://example.com/article",
        help="Enter a YouTube video URL or website URL to summarize"
    )
    
    # Clear sample URL from session state
    if "sample_url" in st.session_state:
        del st.session_state.sample_url
    
    # URL validation and analysis
    if url_input:
        summarizer = ContentSummarizer()
        is_valid, error_msg, content_type = summarizer.validate_url(url_input)
        
        if not is_valid:
            st.error(error_msg)
        else:
            # Show URL analysis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Content Type:** {content_type.title()}")
            
            with col2:
                domain = urlparse(url_input).netloc
                st.info(f"**Domain:** {domain}")
            
            with col3:
                if content_type == "youtube":
                    video_info = summarizer.extract_video_info(url_input)
                    if video_info.get('video_id'):
                        st.info(f"**Video ID:** {video_info['video_id'][:10]}...")
                    else:
                        st.info("**Status:** Ready to process")
                else:
                    st.info("**Status:** Ready to process")
    
    # Summarization
    if url_input and api_key:
        summarizer = ContentSummarizer()
        is_valid, error_msg, content_type = summarizer.validate_url(url_input)
        
        if not is_valid:
            st.error(error_msg)
        elif not summarizer.validate_api_key(api_key, api_type):
            st.error(f"Invalid {api_type.upper()} API key format")
        else:
            # Summarize button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                summarize_button = st.button(
                    "🚀 Summarize Content",
                    type="primary",
                    use_container_width=True,
                    help="Analyze and summarize the content from the provided URL"
                )
            
            if summarize_button:
                try:
                    with st.spinner("🤖 Processing content and generating summary..."):
                        progress_bar = st.progress(0)
                        
                        # Progress steps
                        steps = [
                            ("Loading content from URL...", 25),
                            ("Analyzing content structure...", 50),
                            ("Generating AI summary...", 75),
                            ("Finalizing results...", 100)
                        ]
                        
                        for step_text, progress in steps:
                            st.text(step_text)
                            progress_bar.progress(progress)
                            time.sleep(0.5)
                        
                        # Perform summarization
                        results = summarizer.summarize_content(
                            url=url_input,
                            api_key=api_key,
                            api_type=api_type,
                            summary_style=summary_style,
                            model_name=model_name
                        )
                        
                        progress_bar.empty()
                        st.success("✅ Content successfully summarized!")
                    
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
                        if st.button("🔄 Different Style", help="Try different summary style"):
                            st.info("💡 Change the summary style above and process again!")
                    
                    with action_cols[2]:
                        if st.button("🔗 New URL", help="Summarize different content"):
                            st.info("💡 Enter a new URL above to summarize different content!")
                    
                    with action_cols[3]:
                        if st.button("🚀 Custom System", help="Build enterprise content processing"):
                            st.info("🚀 [Contact GenAI Expertise](https://genaiexpertise.com/contact/) for custom content processing solutions!")
                
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    st.info("""
                    **Troubleshooting Tips:**
                    - Ensure the URL is accessible and contains readable content
                    - Check that your API key is valid and has sufficient credits
                    - Try with a different URL if the content is behind a paywall
                    - For YouTube videos, ensure they have available captions/transcripts
                    """)
    
    elif url_input and not api_key:
        st.warning(f"⚠️ Please enter your {api_type.upper()} API key to proceed")
    elif not url_input:
        st.info("👆 Enter a content URL above to begin summarization!")
    
    # Sidebar information
    with st.sidebar:
        st.markdown("### 🚀 Key Features")
        st.markdown("""
        - **YouTube Integration**: Extract insights from video content
        - **Web Scraping**: Process articles and blog posts  
        - **Smart Analysis**: Content-aware summarization
        - **Multiple Formats**: Brief, detailed, bullet, executive styles
        - **Fast Processing**: Get results in seconds
        """)
        
        st.markdown("### 💡 Pro Tips")
        st.markdown("""
        - Use **brief** style for quick overviews
        - Choose **detailed** for comprehensive analysis
        - Select **bullet** points for easy scanning
        - Pick **executive** style for business insights
        """)
    
    # Add feedback section
    add_feedback_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 Ready for Enterprise Content Intelligence?
    
    This summarizer demonstrates advanced content processing capabilities. We can build:
    
    - **Automated Content Monitoring** - Track and summarize content from multiple sources
    - **Batch Processing Systems** - Process hundreds of URLs simultaneously
    - **Custom Integration** - Connect with your CMS, research tools, or business systems
    - **Advanced Analytics** - Content classification, sentiment analysis, and trend detection
    - **Multi-language Support** - Process content in dozens of languages
    
    **[Contact GenAI Expertise](https://genaiexpertise.com)** to build your enterprise content intelligence platform.
    """)

if __name__ == "__main__":
    main()