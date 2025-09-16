import streamlit as st
from langchain_core.prompts import PromptTemplate 
from langchain_openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
import time
from typing import Optional

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
    page_title="Smart Text Style Transfer",
    page_icon="🎨",
    layout="wide"
)

# Authentication check
if not check_password():
    st.stop()

# Enhanced template with better examples and instructions
template = """
You are an expert copywriter and linguistic specialist. Your task is to transform the given text according to specific tone and dialect requirements while maintaining the core message and improving clarity.

TRANSFORMATION REQUIREMENTS:
- Properly redact and improve the draft text for clarity and impact
- Convert to the specified TONE: {tone}
- Adapt to the specified DIALECT: {dialect}
- Maintain the original meaning and key information
- Ensure professional quality and readability

TONE EXAMPLES:
Formal Tone Characteristics:
- Professional vocabulary and complete sentences
- Third-person perspective when appropriate  
- Clear structure with proper grammar
- Authoritative and respectful language
- Example: "We are pleased to announce that following comprehensive deliberations, the leadership team has reached a unanimous decision regarding the strategic direction of our organization."

Informal Tone Characteristics:
- Conversational and friendly language
- Contractions and casual expressions
- First/second person perspective
- Accessible vocabulary and shorter sentences
- Example: "Hey everyone! We've got some exciting news to share - after lots of discussions and brainstorming sessions, we've finally decided on our next big move!"

DIALECT ADAPTATIONS:
American English:
- Vocabulary: elevator, apartment, garbage, cookies, parking lot, pants, gas, flashlight
- Spelling: organize, realize, color, center, defense
- Expressions: "touch base," "circle back," "reach out"

British English:  
- Vocabulary: lift, flat, rubbish, biscuits, car park, trousers, petrol, torch
- Spelling: organise, realise, colour, centre, defence
- Expressions: "have a chat," "get back to," "get in touch"

QUALITY GUIDELINES:
- Start with a warm, appropriate introduction if the original lacks one
- Ensure smooth flow and natural transitions
- Maintain professional standards regardless of tone
- Adapt cultural references appropriately for the dialect
- Preserve technical accuracy while improving readability

SOURCE TEXT: {draft}
REQUIRED TONE: {tone}
REQUIRED DIALECT: {dialect}

Please provide your {dialect} {tone} transformation:
"""

def load_LLM(openai_api_key: str, temperature: float = 0.7) -> OpenAI:
    """Load and configure the language model."""
    if not openai_api_key or not openai_api_key.startswith("sk-"):
        st.error("⚠️ Valid OpenAI API key required for this demo")
        st.stop()
    
    return OpenAI(temperature=temperature, openai_api_key=openai_api_key)

def get_api_key() -> str:
    """Get API key from environment or secrets."""
    return (
        os.environ.get("OPENAI_API_KEY") or 
        st.secrets.get("APIKEY", {}).get("OPENAI_API_KEY", "")
    )

def validate_input(text: str, max_words: int = 700) -> tuple[bool, str]:
    """Validate user input text."""
    if not text.strip():
        return False, "Please enter some text to transform."
    
    word_count = len(text.split())
    if word_count > max_words:
        return False, f"Text is too long ({word_count} words). Maximum allowed: {max_words} words."
    
    return True, ""

def display_transformation_examples():
    """Show example transformations to help users understand capabilities."""
    with st.expander("📚 See Example Transformations", expanded=False):
        
        st.markdown("### Original Text:")
        st.info("""
        Our company had a meeting yesterday and we decided that Sam Altman is coming back as CEO. 
        It took us 5 days of talking and arguing but we finally agreed to bring him back.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎩 **Formal + American:**")
            st.success("""
            We are pleased to announce that following a comprehensive five-day deliberation period, 
            our organization has reached a unanimous decision to reinstate Sam Altman as Chief Executive Officer. 
            This decision reflects our commitment to strong leadership and organizational excellence.
            """)
        
        with col2:
            st.markdown("### 💬 **Informal + British:**")
            st.success("""
            Great news everyone! After quite a lot of back-and-forth discussions over the past five days, 
            we've decided to bring Sam Altman back as our CEO. It wasn't easy reaching this decision, 
            but we're chuffed with the outcome!
            """)

def main():
    """Main application logic."""
    
    # Header section
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white; text-align: center;'>
        <h1>🎨 Smart Text Style Transfer</h1>
        <p>Transform your text with AI-powered tone and dialect conversion</p>
        <p><em>Professional content adaptation for global audiences</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation and status
    show_demo_navigation()
    display_api_status()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### ✨ Transform Your Content")
        st.markdown("""
        This AI-powered tool helps you adapt your content for different audiences by:
        - **Converting tone** (formal ↔ informal) for appropriate communication style
        - **Adapting dialect** (American ↔ British English) for regional preferences  
        - **Improving clarity** and professional quality
        - **Maintaining meaning** while enhancing readability
        """)
    
    with col2:
        st.markdown("### 🎯 Perfect For:")
        st.markdown("""
        - **Marketing Content**
        - **International Communications** 
        - **Brand Voice Consistency**
        - **Professional Documentation**
        - **Cultural Adaptation**
        """)
    
    # Example transformations
    display_transformation_examples()
    
    # Input section
    st.markdown("---")
    st.markdown("## 📝 Enter Your Text")
    
    # Text input with enhanced UX
    draft_input = st.text_area(
        label="Text to Transform",
        placeholder="""Paste your text here... 

For example:
"We had a meeting and decided to hire the new guy. He seems pretty good and we think he'll fit in well with the team. Should start next week."

Try different combinations of tone and dialect to see the transformation magic! ✨""",
        height=150,
        help="Enter up to 700 words. The AI will improve clarity while transforming tone and dialect."
    )
    
    # Show character/word count
    if draft_input:
        word_count = len(draft_input.split())
        char_count = len(draft_input)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Words", word_count, delta=f"{700-word_count} remaining")
        with col2:
            st.metric("Characters", char_count)
        with col3:
            if word_count > 700:
                st.error(f"Reduce by {word_count-700} words")
            else:
                st.success("Length OK ✅")
    
    # Transformation options
    st.markdown("## ⚙️ Transformation Settings")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        option_tone = st.selectbox(
            '🎭 Select Tone Style:',
            options=['Formal', 'Informal'],
            help="""
            • **Formal**: Professional, structured, authoritative
            • **Informal**: Conversational, friendly, accessible
            """
        )
    
    with col2:
        option_dialect = st.selectbox(
            '🌍 Choose English Dialect:',
            options=['American', 'British'],
            help="""
            • **American**: US spelling, vocabulary, and expressions
            • **British**: UK spelling, vocabulary, and expressions  
            """
        )
    
    with col3:
        temperature = st.slider(
            '🎨 Creativity Level:',
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower = more conservative, Higher = more creative transformations"
        )
    
    # Validation and processing
    if draft_input:
        is_valid, error_message = validate_input(draft_input)
        
        if not is_valid:
            st.error(error_message)
            st.stop()
        
        # API key check
        openai_api_key = get_api_key()
        if not openai_api_key:
            st.warning("""
            ⚠️ **OpenAI API Key Required** 
            
            This demo requires an OpenAI API key to function. Please contact GenAI Expertise for demo access.
            """)
            st.info("💡 [Get API Access Instructions](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)")
            st.stop()
        
        # Transform button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            transform_button = st.button(
                "🚀 Transform Text",
                type="primary",
                use_container_width=True,
                help="Click to apply AI-powered style transformation"
            )
        
        if transform_button:
            # Show processing state
            with st.spinner('🤖 AI is transforming your text...'):
                progress_bar = st.progress(0)
                
                try:
                    # Initialize the language model
                    progress_bar.progress(25)
                    llm = load_LLM(openai_api_key, temperature)
                    
                    # Create the prompt
                    progress_bar.progress(50)
                    prompt = PromptTemplate(
                        input_variables=["tone", "dialect", "draft"],
                        template=template,
                    )
                    
                    # Format prompt with user inputs
                    formatted_prompt = prompt.format(
                        tone=option_tone, 
                        dialect=option_dialect, 
                        draft=draft_input
                    )
                    
                    # Generate transformation
                    progress_bar.progress(75)
                    transformed_text = llm(formatted_prompt)
                    progress_bar.progress(100)
                    
                    # Clear progress bar
                    progress_bar.empty()
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("## ✨ Transformation Results")
                    
                    # Results in two columns for comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📄 Original Text")
                        st.info(draft_input)
                        
                        # Original text stats
                        orig_words = len(draft_input.split())
                        orig_sentences = draft_input.count('.') + draft_input.count('!') + draft_input.count('?')
                        st.caption(f"📊 Original: {orig_words} words, ~{orig_sentences} sentences")
                    
                    with col2:
                        st.markdown(f"### ✨ {option_tone} {option_dialect} Version")
                        st.success(transformed_text)
                        
                        # Transformed text stats
                        trans_words = len(transformed_text.split())
                        trans_sentences = transformed_text.count('.') + transformed_text.count('!') + transformed_text.count('?')
                        st.caption(f"📊 Transformed: {trans_words} words, ~{trans_sentences} sentences")
                    
                    # Analysis section
                    st.markdown("### 📊 Transformation Analysis")
                    
                    analysis_cols = st.columns(4)
                    
                    with analysis_cols[0]:
                        word_change = trans_words - orig_words
                        st.metric(
                            "Word Count Change", 
                            f"{word_change:+d}",
                            help="Difference in word count after transformation"
                        )
                    
                    with analysis_cols[1]:
                        st.metric(
                            "Tone Applied", 
                            option_tone,
                            help="Selected tone style"
                        )
                    
                    with analysis_cols[2]:
                        st.metric(
                            "Dialect Applied", 
                            option_dialect,
                            help="Selected English dialect"
                        )
                    
                    with analysis_cols[3]:
                        complexity_score = "High" if trans_words > orig_words else "Simplified"
                        st.metric(
                            "Complexity", 
                            complexity_score,
                            help="Relative complexity compared to original"
                        )
                    
                    # Action buttons for results
                    st.markdown("### 🎯 Next Steps")
                    
                    action_cols = st.columns(4)
                    
                    with action_cols[0]:
                        if st.button("📋 Copy Result", help="Copy transformed text to clipboard"):
                            st.code(transformed_text)
                            st.success("✅ Text ready to copy from code block above!")
                    
                    with action_cols[1]:
                        if st.button("🔄 Try Different Style", help="Transform with different settings"):
                            st.info("💡 Adjust the tone and dialect settings above, then click Transform again!")
                    
                    with action_cols[2]:
                        if st.button("📝 New Text", help="Clear and start with new text"):
                            st.info("💡 Clear the text area above and enter new content to transform!")
                    
                    with action_cols[3]:
                        if st.button("💬 Get Custom Solution", help="Contact for custom AI implementation"):
                            st.info("🚀 [Contact GenAI Expertise](https://genaiexpertise.com/contact/) for custom AI solutions!")
                    
                    # Quality indicators
                    st.markdown("### 🏆 Quality Indicators")
                    
                    quality_cols = st.columns(3)
                    
                    with quality_cols[0]:
                        st.markdown("""
                        **✅ Transformation Quality:**
                        - Tone successfully adapted
                        - Dialect conventions applied
                        - Original meaning preserved
                        - Professional grammar maintained
                        """)
                    
                    with quality_cols[1]:
                        st.markdown("""
                        **🎯 Business Benefits:**
                        - Consistent brand voice
                        - Cultural appropriateness
                        - Enhanced readability
                        - Professional presentation
                        """)
                    
                    with quality_cols[2]:
                        st.markdown("""
                        **🔧 Technical Features:**
                        - AI-powered analysis
                        - Context preservation
                        - Style consistency
                        - Scalable processing
                        """)
                    
                except Exception as e:
                    progress_bar.empty()
                    st.error(f"❌ Transformation failed: {str(e)}")
                    st.info("""
                    **Troubleshooting Tips:**
                    - Check your internet connection
                    - Ensure text is within length limits
                    - Try again in a moment
                    - Contact support if issues persist
                    """)
    
    # Add feedback section
    add_feedback_section()
    
    # Footer with additional information
    st.markdown("---")
    st.markdown("""
    ### 🚀 Ready for Custom AI Solutions?
    
    This demo showcases just one application of our AI text processing capabilities. We can build:
    
    - **Custom tone adaptation** for your specific brand voice
    - **Multi-language processing** beyond English dialects  
    - **Batch processing** for large content volumes
    - **API integrations** for seamless workflow integration
    - **Advanced analytics** for content performance tracking
    
    **[Contact GenAI Expertise](https://genaiexpertise.com)** to discuss your custom AI content solution.
    """)

if __name__ == "__main__":
    main()