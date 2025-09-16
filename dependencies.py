import streamlit as st
import hashlib
from typing import Optional

def password_entered():
    """Checks whether a password entered by the user is correct."""
    if (
        st.session_state["password"] 
        == st.secrets.get("app_password", "demo123")  # Fallback for demo
    ):
        st.session_state["password_correct"] = True
        # Clear the password from session state for security
        del st.session_state["password"]
    else:
        st.session_state["password_correct"] = False

def check_password() -> bool:
    """
    Enhanced password authentication with better UX and security.
    Returns True if the user has entered a correct password.
    """
    # Check if password has been validated
    if "password_correct" not in st.session_state:
        # First run, show login form
        show_login_form()
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show error and login form
        show_login_form(show_error=True)
        return False
    else:
        # Password correct
        return True

def show_login_form(show_error: bool = False):
    """Display the login form with optional error message."""
    
    # Custom CSS for login form
    st.markdown("""
    <style>
        .login-container {
            max-width: 400px;
            margin: 2rem auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background: white;
            text-align: center;
        }
        .login-header {
            color: #667eea;
            margin-bottom: 1.5rem;
        }
        .error-message {
            color: #ff4757;
            background: #ffe3e3;
            padding: 0.8rem;
            border-radius: 5px;
            margin: 1rem 0;
            border-left: 4px solid #ff4757;
        }
        .demo-info {
            background: #e8f4f8;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
            font-size: 0.9rem;
            color: #2c3e50;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create centered login container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-container">
            <div class="login-header">
                <h2>🔐 Access GenAI Portfolio</h2>
                <p>Enter password to explore our AI solutions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show error message if password was incorrect
        if show_error:
            st.markdown("""
            <div class="error-message">
                ❌ <strong>Access Denied:</strong> Incorrect password. Please try again.
            </div>
            """, unsafe_allow_html=True)
        
        # Password input
        st.text_input(
            "Password",
            type="password",
            placeholder="Enter access password",
            on_change=password_entered,
            key="password",
            help="Contact GenAI Expertise if you need access credentials"
        )
        
        # Demo information
        st.markdown("""
        <div class="demo-info">
            <strong>📋 Demo Access Information:</strong><br>
            This portfolio showcases production-ready AI solutions. 
            Each demo represents technology that can be customized for your business needs.
            <br><br>
            <strong>🔑 Need Access?</strong><br>
            Contact us at <a href="https://genaiexpertise.com" target="_blank">GenAIExpertise.com</a> 
            for demonstration credentials or to discuss your AI project requirements.
        </div>
        """, unsafe_allow_html=True)

def get_client_info() -> dict:
    """
    Collect basic client information for demo tracking (optional).
    This can help with follow-up and customization.
    """
    if "client_info_collected" not in st.session_state:
        st.session_state["client_info_collected"] = False
    
    if not st.session_state["client_info_collected"]:
        with st.sidebar:
            st.markdown("### 👋 Welcome!")
            st.markdown("Help us customize your demo experience:")
            
            with st.form("client_info"):
                company = st.text_input("Company (optional)")
                industry = st.selectbox(
                    "Industry (optional)",
                    ["", "Technology", "Healthcare", "Finance", "Education", 
                     "Retail", "Manufacturing", "Other"]
                )
                use_case = st.text_area("What AI solutions interest you? (optional)")
                
                if st.form_submit_button("Continue to Demo"):
                    st.session_state["client_info"] = {
                        "company": company,
                        "industry": industry,
                        "use_case": use_case
                    }
                    st.session_state["client_info_collected"] = True
                    st.rerun()
            
            return {}
    
    return st.session_state.get("client_info", {})

def show_demo_navigation():
    """Enhanced sidebar navigation with descriptions."""
    st.sidebar.markdown("## 🎯 AI Solutions Demo")
    
    pages = {
        "🏠 Portfolio Overview": "main_portfolio.py",
        "🎨 Text Style Transfer": "main.py", 
        "📝 Blog Generator": "blog_post_generator.py",
        "💬 AI Chatbot": "chatbot.py",
        "📊 RAG Evaluator": "evaluate_q_and_a_from_long_document.py",
        "🔍 Review Analyzer": "extract_json_from_review.py",
        "📑 Document Summarizer": "split_and_summarize.py",
        "⚡ Quick Summarizer": "text_summarization.py",
        "🌍 Translator": "translator.py"
    }
    
    st.sidebar.markdown("**Select a demo to explore:**")
    
    for display_name, file_name in pages.items():
        if st.sidebar.button(
            display_name,
            use_container_width=True,
            help=f"Explore the {display_name.split(' ', 1)[1]} functionality"
        ):
            st.switch_page(file_name)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **💡 Pro Tip:** Each demo showcases different AI capabilities. 
    Try multiple demos to see the full range of possibilities!
    """)

def display_api_status():
    """Show API connection status in sidebar."""
    import os
    
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        # Check API key status
        api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("APIKEY", {}).get("OPENAI_API_KEY")
        
        if api_key and api_key.startswith("sk-"):
            st.success("✅ AI Services Connected")
        else:
            st.warning("⚠️ API Configuration Needed")
            st.info("Demo may have limited functionality")

def add_feedback_section():
    """Add feedback collection in sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📢 Feedback")
        
        feedback = st.text_area(
            "Share your thoughts:",
            placeholder="What did you think of this demo?",
            height=80
        )
        
        if st.button("Send Feedback", use_container_width=True):
            if feedback.strip():
                # Here you could integrate with a feedback collection system
                st.success("Thank you for your feedback!")
                # In production, you might send this to a database or email
            else:
                st.warning("Please enter your feedback first.")

def setup_page_config(
    page_title: str,
    page_icon: str = "🤖",
    layout: str = "centered"
):
    """Standardized page configuration for all demo pages."""
    st.set_page_config(
        page_title=f"{page_title} - GenAI Expertise",
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://genaiexpertise.com/contact/",    
            "Report a bug": "https://github.com/genaiexpertise/RAGApps/issues",
            "About": "https://genaiexpertise.com",
        },
    )
    
    # Add logo if available
    try:
        st.logo(
            'assets/logo.png',
            link="https://genaiexpertise.com",
            icon_image="assets/icon.png",
        )
    except:
        pass  # Logo files not found, continue without them