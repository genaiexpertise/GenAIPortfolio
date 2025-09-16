import os
import streamlit as st
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid

from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
    page_title="Intelligent AI Chatbot",
    page_icon="💬",
    layout="wide"
)

# Authentication check
if not check_password():
    st.stop()

class EnhancedChatbot:
    """Enhanced chatbot with better memory management and features."""
    
    def __init__(self):
        self.api_key = self.get_api_key()
        self.chatbot_memory = {}
        self.initialize_chatbot()
    
    def get_api_key(self) -> str:
        """Get API key from environment or secrets."""
        return (
            os.environ.get("OPENAI_API_KEY") or 
            st.secrets.get("APIKEY", {}).get("OPENAI_API_KEY", "")
        )
    
    def initialize_chatbot(self):
        """Initialize the chatbot with proper configuration."""
        if not self.api_key or not self.api_key.startswith("sk-"):
            st.error("⚠️ Valid OpenAI API key required for this demo")
            st.stop()
        
        # Enhanced system prompt for better conversations
        system_prompt = """You are an intelligent AI assistant created by GenAI Expertise. You are helpful, knowledgeable, and professional.

Your capabilities include:
- Answering questions across various topics
- Helping with problem-solving and analysis
- Providing explanations and tutorials
- Offering creative assistance
- Supporting business and technical discussions

Guidelines for responses:
- Be helpful, accurate, and concise
- Ask clarifying questions when needed
- Provide practical, actionable advice
- Maintain a professional yet friendly tone
- If you're unsure about something, say so honestly
- Remember the conversation context for better continuity

Current conversation context: This is a demonstration of AI chatbot capabilities for potential clients of GenAI Expertise."""

        # Create prompt template with system message and chat history
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Initialize the language model
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=self.api_key,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Create the chain
        self.chain = self.prompt_template | self.llm
        
        # Create runnable with message history
        self.chatbot_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Get or create session history for a given session ID."""
        if session_id not in self.chatbot_memory:
            self.chatbot_memory[session_id] = ChatMessageHistory()
        return self.chatbot_memory[session_id]
    
    def get_session_id(self) -> str:
        """Get or create a unique session ID for the user."""
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id
    
    def clear_conversation(self):
        """Clear the current conversation."""
        session_id = self.get_session_id()
        if session_id in self.chatbot_memory:
            self.chatbot_memory[session_id].clear()
        st.session_state.messages = []
        self.add_welcome_message()
    
    def add_welcome_message(self):
        """Add a welcome message to start the conversation."""
        welcome_msg = """👋 Hello! I'm your AI assistant powered by GenAI Expertise's technology. 

I'm here to help you with:
• **Questions & Research** - Get answers on various topics
• **Problem Solving** - Work through challenges together  
• **Creative Tasks** - Brainstorming, writing, and ideation
• **Technical Support** - Coding, analysis, and explanations
• **Business Insights** - Strategy, planning, and decision support

What can I help you with today?"""
        
        st.session_state.messages = [
            {"role": "assistant", "content": welcome_msg, "timestamp": datetime.now()}
        ]
    
    def generate_response(self, user_input: str) -> str:
        """Generate a response to user input."""
        try:
            session_id = self.get_session_id()
            
            response = self.chatbot_with_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            return response.content
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again or rephrase your question."

def initialize_chat_state():
    """Initialize chat state if not already done."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = EnhancedChatbot()
    
    # Add welcome message if no messages exist
    if not st.session_state.messages:
        st.session_state.chatbot.add_welcome_message()

def display_chat_header():
    """Display the chat header with features and controls."""
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white; text-align: center;'>
        <h1>💬 Intelligent AI Chatbot</h1>
        <p>Experience natural conversations with advanced AI</p>
        <p><em>Context-aware responses with persistent memory</em></p>
    </div>
    """, unsafe_allow_html=True)

def display_chat_features():
    """Display chatbot features and capabilities."""
    with st.expander("🚀 Chatbot Capabilities", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🧠 Core Intelligence:**
            - Natural language understanding
            - Context-aware responses
            - Multi-turn conversations
            - Memory of chat history
            - Reasoning and analysis
            """)
        
        with col2:
            st.markdown("""
            **💼 Business Applications:**
            - Customer support automation
            - Employee assistance
            - Knowledge base queries
            - Training and onboarding
            - 24/7 availability
            """)

def display_conversation_controls():
    """Display conversation management controls."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Clear Chat", help="Start a new conversation", use_container_width=True):
            st.session_state.chatbot.clear_conversation()
            st.rerun()
    
    with col2:
        conversation_count = len(st.session_state.messages)
        st.metric("Messages", conversation_count)
    
    with col3:
        # Calculate approximate conversation length
        total_chars = sum(len(msg.get("content", "")) for msg in st.session_state.messages)
        st.metric("Characters", f"{total_chars:,}")
    
    with col4:
        session_id = st.session_state.get("session_id", "N/A")[:8]
        st.metric("Session", f"...{session_id}")

def display_suggested_prompts():
    """Display suggested conversation starters."""
    st.markdown("### 💡 Try These Conversation Starters:")
    
    suggestions = [
        "How can AI improve customer service?",
        "Explain machine learning in simple terms",
        "Help me brainstorm blog post ideas",
        "What are the benefits of automation?",
        "How do I implement AI in my business?",
        "Create a marketing strategy outline"
    ]
    
    cols = st.columns(3)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 3]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                # Add the suggestion as a user message
                st.session_state.user_input = suggestion
                st.rerun()

def format_message_timestamp(timestamp):
    """Format message timestamp for display."""
    if timestamp:
        return timestamp.strftime("%H:%M")
    return ""

def display_chat_messages():
    """Display all chat messages with enhanced formatting."""
    
    # Custom CSS for better message styling
    st.markdown("""
    <style>
        .chat-message {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 10px;
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        .assistant-message {
            background-color: #f5f5f5;
            border-left: 4px solid #4caf50;
        }
        .message-timestamp {
            font-size: 0.8rem;
            color: #666;
            margin-top: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Display conversation
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        timestamp = message.get("timestamp")
        
        with st.chat_message(role):
            st.markdown(content)
            if timestamp:
                st.caption(f"🕒 {format_message_timestamp(timestamp)}")

def handle_user_input():
    """Handle user input and generate responses."""
    
    # Check for suggested prompt
    if "user_input" in st.session_state and st.session_state.user_input:
        prompt = st.session_state.user_input
        del st.session_state.user_input  # Clear it
    else:
        prompt = st.chat_input("Type your message here...", key="chat_input")
    
    if prompt:
        # Add user message to history
        user_message = {
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now()
        }
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"🕒 {format_message_timestamp(user_message['timestamp'])}")
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                response = st.session_state.chatbot.generate_response(prompt)
                
                # Simulate typing effect
                placeholder = st.empty()
                displayed_text = ""
                for char in response:
                    displayed_text += char
                    placeholder.markdown(displayed_text + "▌")
                    time.sleep(0.01)  # Adjust speed as needed
                
                placeholder.markdown(response)
                
                # Add timestamp
                response_time = datetime.now()
                st.caption(f"🕒 {format_message_timestamp(response_time)}")
        
        # Add assistant response to history
        assistant_message = {
            "role": "assistant", 
            "content": response,
            "timestamp": response_time
        }
        st.session_state.messages.append(assistant_message)
        
        st.rerun()

def display_chat_analytics():
    """Display conversation analytics."""
    if len(st.session_state.messages) > 2:  # More than just welcome message
        with st.sidebar:
            st.markdown("### 📊 Conversation Stats")
            
            user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
            assistant_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Your Messages", len(user_messages))
            with col2:
                st.metric("AI Responses", len(assistant_messages))
            
            # Average response length
            if assistant_messages:
                avg_length = sum(len(msg["content"]) for msg in assistant_messages) // len(assistant_messages)
                st.metric("Avg Response Length", f"{avg_length} chars")

def main():
    """Main application logic."""
    
    # Initialize chat state
    initialize_chat_state()
    
    # Display header and navigation
    display_chat_header()
    show_demo_navigation()
    display_api_status()
    
    # Chat features explanation
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎯 Experience Intelligent Conversations")
        st.markdown("""
        This AI chatbot demonstrates advanced conversational capabilities with:
        - **Context Awareness** - Remembers your conversation history
        - **Natural Responses** - Human-like communication style
        - **Multi-topic Expertise** - Knowledgeable across various domains
        - **Professional Integration** - Ready for business deployment
        """)
    
    with col2:
        st.markdown("### 💼 Business Benefits:")
        st.markdown("""
        - **24/7 Availability**
        - **Consistent Quality**
        - **Scalable Support**
        - **Cost Reduction**
        - **Customer Satisfaction**
        """)
    
    # Display features and controls
    display_chat_features()
    display_conversation_controls()
    
    # Main chat area
    st.markdown("---")
    
    # Show suggested prompts if conversation is new
    if len(st.session_state.messages) <= 1:
        display_suggested_prompts()
        st.markdown("---")
    
    # Display chat messages
    display_chat_messages()
    
    # Handle user input (this will trigger rerun if there's input)
    handle_user_input()
    
    # Display analytics in sidebar
    display_chat_analytics()
    
    # Add feedback section
    add_feedback_section()
    
    # Footer information
    st.markdown("---")
    st.markdown("""
    ### 🚀 Ready for Custom Chatbot Solutions?
    
    This demonstration showcases core conversational AI capabilities. We can build:
    
    - **Custom Knowledge Bases** - Train on your specific data and documents
    - **Multi-channel Integration** - Deploy across web, mobile, and messaging platforms
    - **Advanced Analytics** - Track conversations, satisfaction, and performance
    - **Specialized Functions** - Integrate with your APIs and business systems
    - **Enterprise Security** - GDPR compliance, data encryption, and audit trails
    
    **[Contact GenAI Expertise](https://genaiexpertise.com)** to discuss your custom chatbot solution.
    """)

if __name__ == "__main__":
    main()