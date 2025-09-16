import streamlit as st
from pathlib import Path
import sys
import streamlit.components.v1 as components

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from dependencies import check_password

# Page configuration
st.set_page_config(
    page_title="GenAI Expertise - AI Solutions Portfolio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://genaiexpertise.com/contact/",    
        "Report a bug": "https://github.com/genaiexpertise/RAGApps/issues",
        "About": "https://genaiexpertise.com",
    },
)

# Custom CSS for better visual appeal
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .demo-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        height: 100%;
    }
    .demo-section h3 {
        color: #333;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    .demo-section h4 {
        color: #666;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    .demo-section ul {
        margin: 0.5rem 0;
        padding-left: 1.2rem;
    }
    .demo-section li {
        margin-bottom: 0.3rem;
        color: #555;
    }
    .demo-section p {
        color: #666;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .tech-badge {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Logo and header
if Path("assets/logo.png").exists():
    st.logo(
        'assets/logo.png',
        link="https://genaiexpertise.com",
        icon_image="assets/icon.png" if Path("assets/icon.png").exists() else None,
    )

# Authentication check
if not check_password():
    st.stop()

# Main header
st.markdown("""
<div class="main-header">
    <h1>🤖 GenAI Expertise Portfolio</h1>
    <p>Advanced AI Solutions for Modern Businesses</p>
    <p><em>Showcasing cutting-edge Language Models, RAG Systems, and Intelligent Applications</em></p>
</div>
""", unsafe_allow_html=True)

# Introduction section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## Welcome to Our AI Solutions Showcase
    
    Explore our comprehensive suite of AI-powered applications designed to transform your business operations. 
    Each demo represents production-ready technology that can be customized for your specific needs.
    
    **🎯 What You'll Discover:**
    - Advanced Natural Language Processing
    - Intelligent Document Analysis
    - Multi-language Support
    - Real-time AI Interactions
    - Quality Assurance Systems
    """)

with col2:
    st.markdown("""
    ### 🚀 Quick Start
    Navigate through our demos using the sidebar. Each application includes:
    - **Live Demo**: Interactive experience
    - **Use Cases**: Business applications
    - **Technical Details**: Implementation insights
    
    ### 📞 Ready to Build?
    [Contact GenAI Expertise](https://genaiexpertise.com/contact/) 
    to discuss your custom AI solution.
    """)

# Technology stack
st.markdown("---")
st.markdown("## 🛠️ Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Core AI Technologies:**
    <div class="tech-badge">OpenAI GPT</div>
    <div class="tech-badge">LangChain</div>
    <div class="tech-badge">Vector Databases</div>
    <div class="tech-badge">RAG Systems</div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    **Development Framework:**
    <div class="tech-badge">Streamlit</div>
    <div class="tech-badge">FastAPI</div>
    <div class="tech-badge">Python</div>
    <div class="tech-badge">Pandas</div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    **Advanced Features:**
    <div class="tech-badge">Real-time Processing</div>
    <div class="tech-badge">Multi-language</div>
    <div class="tech-badge">Quality Evaluation</div>
    <div class="tech-badge">Memory Management</div>
    """, unsafe_allow_html=True)

# Application showcase
st.markdown("---")
st.markdown("## 🎨 Application Portfolio")

# Create feature cards for each application
applications = [
    {
        "title": "🎯 Smart Text Redaction & Style Transfer",
        "description": "Transform any text to match your brand voice with intelligent tone and dialect conversion.",
        "features": ["Tone Conversion (Formal/Informal)", "Dialect Adaptation (US/UK)", "Content Redaction", "Brand Voice Consistency"],
        "use_cases": ["Marketing Content", "International Communications", "Brand Standardization"],
        "page": "smart_text_style.py"
    },
    {
        "title": "📝 AI Blog Post Generator", 
        "description": "Generate professional, SEO-optimized blog content tailored to your industry and audience.",
        "features": ["Topic-based Generation", "Word Count Control", "SEO Optimization", "Industry Expertise"],
        "use_cases": ["Content Marketing", "Thought Leadership", "SEO Strategy"],
        "page": "blog_post_generator.py"
    },
    {
        "title": "💬 Intelligent Chatbot",
        "description": "Context-aware conversational AI with persistent memory for enhanced customer interactions.",
        "features": ["Conversation Memory", "Context Awareness", "Natural Responses", "Session Management"],
        "use_cases": ["Customer Support", "Sales Assistance", "Internal Help Desk"],
        "page": "chatbot.py"
    },
    {
        "title": "📊 RAG System Evaluator",
        "description": "Quality assurance tool for Retrieval-Augmented Generation systems with accuracy metrics.",
        "features": ["Answer Validation", "Quality Scoring", "Hallucination Detection", "Performance Analytics"],
        "use_cases": ["AI Quality Assurance", "System Validation", "Performance Monitoring"],
        "page": "evaluate_q_and_a_from_long_document.py"
    },
    {
        "title": "🔍 Smart Review Analyzer",
        "description": "Extract actionable insights from customer reviews with sentiment and delivery analysis.",
        "features": ["Sentiment Analysis", "Delivery Tracking", "Price Perception", "Structured Output"],
        "use_cases": ["Customer Insights", "Product Analysis", "Market Research"],
        "page": "extract_json_from_review.py"
    },
    {
        "title": "📑 Advanced Document Summarizer",
        "description": "Process long documents with intelligent chunking and comprehensive summarization.",
        "features": ["Large Document Processing", "Intelligent Chunking", "Context Preservation", "Multiple Formats"],
        "use_cases": ["Research Analysis", "Report Generation", "Knowledge Extraction"],
        "page": "split_and_summarize.py"
    },
    {
        "title": "⚡ Quick Text Summarizer",
        "description": "Instant text summarization for rapid content processing and analysis.",
        "features": ["Real-time Processing", "Multiple Summary Lengths", "Key Point Extraction", "Fast Performance"],
        "use_cases": ["Content Curation", "Meeting Notes", "Research Briefs"],
        "page": "text_summarization.py"
    },
    {
        "title": "🌍 Multi-Language Translator",
        "description": "Specialized translation service with focus on Nigerian languages and cultural context.",
        "features": ["Nigerian Languages", "Cultural Context", "Technical Translation", "Real-time Processing"],
        "use_cases": ["International Business", "Cultural Communication", "Educational Content"],
        "page": "translator.py"
    }
]

# Display applications in a grid using HTML blocks
for i in range(0, len(applications), 2):
    col1, col2 = st.columns(2)
    
    for j, col in enumerate([col1, col2]):
        if i + j < len(applications):
            app = applications[i + j]
            html_block = f"""
            <div class="demo-section">
                <h3>{app['title']}</h3>
                <p>{app['description']}</p>
                
                <h4>🔧 Key Features:</h4>
                <ul>
                    {''.join([f'<li>{feature}</li>' for feature in app['features']])}
                </ul>
                
                <h4>💼 Business Use Cases:</h4>
                <ul>
                    {''.join([f'<li>{use_case}</li>' for use_case in app['use_cases']])}
                </ul>
            </div>
            """
            with col:
                components.html(html_block, height=400, scrolling=True)

# Call to action
st.markdown("---")
st.markdown("""
## 🚀 Ready to Transform Your Business?

Our AI solutions are designed to integrate seamlessly into your existing workflows, providing immediate value and long-term competitive advantages.

### Next Steps:
1. **Explore** the demos using the sidebar navigation
2. **Experience** the power of each AI application
3. **Contact us** to discuss your specific requirements
4. **Get a custom solution** tailored to your business needs

### Why Choose GenAI Expertise?
- **Production-Ready Solutions**: All demos represent deployable technology
- **Custom Development**: Tailored to your specific business requirements  
- **Ongoing Support**: Comprehensive maintenance and updates
- **Scalable Architecture**: Grows with your business needs
""")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📧 Contact Information**
    - Website: [GenAIExpertise.com](https://genaiexpertise.com)
    - Email: Contact via website
    - Support: Professional AI consulting
    """)

with col2:
    st.markdown("""
    **🛠️ Services Offered**
    - Custom AI Development
    - RAG System Implementation
    - AI Strategy Consulting
    - Training & Support
    """)

with col3:
    st.markdown("""
    **🎯 Industries Served**
    - E-commerce & Retail
    - Healthcare & Medical
    - Education & Training
    - Financial Services
    """)

st.markdown("""
---
<div style='text-align: center; color: #666; padding: 1rem;'>
    <em>© 2024 GenAI Expertise. Transforming businesses through intelligent AI solutions.</em>
</div>
""", unsafe_allow_html=True)
