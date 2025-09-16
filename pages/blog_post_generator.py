import streamlit as st
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv
import os
import time
import re
from typing import Optional, Dict, Any

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
    page_title="AI Blog Post Generator",
    page_icon="📝",
    layout="wide"
)

# Authentication check
if not check_password():
    st.stop()

def get_api_key() -> str:
    """Get API key from environment or secrets."""
    return (
        os.environ.get("OPENAI_API_KEY") or 
        st.secrets.get("APIKEY", {}).get("OPENAI_API_KEY", "")
    )

def validate_topic(topic: str) -> tuple[bool, str]:
    """Validate the blog topic input."""
    if not topic.strip():
        return False, "Please enter a topic for your blog post."
    
    if len(topic.strip()) < 3:
        return False, "Topic should be at least 3 characters long."
    
    if len(topic.split()) > 20:
        return False, "Topic is too long. Please keep it concise (max 20 words)."
    
    return True, ""

def get_enhanced_template(blog_type: str, tone: str, target_audience: str) -> str:
    """Create an enhanced template based on user selections."""
    
    base_template = f"""You are an expert content writer and digital marketing specialist with extensive experience in creating engaging blog content.

BLOG SPECIFICATIONS:
- Type: {blog_type}
- Tone: {tone} 
- Target Audience: {target_audience}
- Word Count: Approximately 400-500 words
- Topic: {{topic}}

CONTENT REQUIREMENTS:
1. Create an attention-grabbing headline that includes the main topic
2. Write an engaging introduction that hooks the reader
3. Develop 3-4 main points with supporting details and examples
4. Include actionable insights or takeaways
5. End with a compelling conclusion that encourages engagement
6. Use appropriate headings and structure for readability

TONE GUIDELINES:
"""

    if tone == "Professional":
        base_template += """- Use authoritative language and industry expertise
- Include relevant statistics or data when applicable  
- Maintain credible and trustworthy voice
- Use formal but accessible language"""
    elif tone == "Conversational":
        base_template += """- Write as if speaking directly to a friend
- Use personal pronouns and relatable examples
- Include rhetorical questions to engage readers
- Keep language casual but informative"""
    elif tone == "Educational":
        base_template += """- Focus on teaching and explaining concepts clearly
- Use step-by-step approaches when relevant
- Include definitions for technical terms
- Structure content for easy learning"""
    else:  # Persuasive
        base_template += """- Use compelling arguments and strong calls-to-action
- Include social proof and credibility indicators
- Address potential objections
- Create urgency and motivation to act"""

    base_template += f"""

AUDIENCE CONSIDERATIONS for {target_audience}:
"""
    
    if target_audience == "Business Professionals":
        base_template += "- Focus on ROI, efficiency, and strategic insights\n- Use business terminology appropriately\n- Include practical applications for work environments"
    elif target_audience == "General Public":
        base_template += "- Use accessible language without too much jargon\n- Include relatable examples from everyday life\n- Focus on broad appeal and general interest"
    elif target_audience == "Industry Experts":
        base_template += "- Use advanced terminology and concepts\n- Include technical details and industry-specific insights\n- Assume high level of background knowledge"
    else:  # Tech Enthusiasts
        base_template += "- Include technical details and innovation aspects\n- Reference latest trends and developments\n- Use appropriate technical terminology"

    base_template += """

FORMATTING:
- Start with an engaging headline (use # for main title)
- Use ## for section headings
- Include bullet points or numbered lists where appropriate
- End with a clear call-to-action

After the blog post, provide:
1. Word count: "Word Count: [X] words"
2. Reading time: "Estimated Reading Time: [X] minutes"
3. Three relevant hashtags for social media

Generate a high-quality, engaging blog post following these specifications:"""

    return base_template

def analyze_blog_content(content: str) -> Dict[str, Any]:
    """Analyze the generated blog content for metrics."""
    
    # Count words
    words = len(content.split())
    
    # Estimate reading time (average 200 words per minute)
    reading_time = max(1, round(words / 200))
    
    # Count sentences
    sentences = len(re.findall(r'[.!?]+', content))
    
    # Count paragraphs
    paragraphs = len([p for p in content.split('\n') if p.strip()])
    
    # Count headings
    headings = len(re.findall(r'^#{1,6}\s', content, re.MULTILINE))
    
    # Check for key elements
    has_introduction = bool(re.search(r'^#{1,2}\s*(intro|introduction)', content, re.IGNORECASE | re.MULTILINE))
    has_conclusion = bool(re.search(r'^#{1,2}\s*(conclusion|summary|final)', content, re.IGNORECASE | re.MULTILINE))
    has_call_to_action = bool(re.search(r'(call.{0,10}action|cta|contact|subscribe|learn more|get started)', content, re.IGNORECASE))
    
    return {
        'words': words,
        'reading_time': reading_time,
        'sentences': sentences,
        'paragraphs': paragraphs,
        'headings': headings,
        'has_introduction': has_introduction,
        'has_conclusion': has_conclusion,
        'has_call_to_action': has_call_to_action
    }

def generate_blog_post(topic: str, blog_type: str, tone: str, target_audience: str, creativity: float) -> str:
    """Generate blog post with enhanced template and error handling."""
    
    openai_api_key = get_api_key()
    
    if not openai_api_key or not openai_api_key.startswith("sk-"):
        st.error("⚠️ Valid OpenAI API key required for this demo")
        return ""
    
    try:
        # Initialize the language model
        llm = OpenAI(
            openai_api_key=openai_api_key,
            temperature=creativity,
            max_tokens=2048
        )
        
        # Create enhanced template
        template = get_enhanced_template(blog_type, tone, target_audience)
        
        # Create prompt
        prompt = PromptTemplate(
            input_variables=["topic"],
            template=template
        )
        
        # Format and generate
        formatted_prompt = prompt.format(topic=topic)
        response = llm(formatted_prompt)
        
        return response
        
    except Exception as e:
        st.error(f"❌ Generation failed: {str(e)}")
        return ""

def main():
    """Main application logic."""
    
    # Header section
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white; text-align: center;'>
        <h1>📝 AI Blog Post Generator</h1>
        <p>Create professional, engaging blog content in seconds</p>
        <p><em>SEO-optimized content tailored to your audience and goals</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation and status
    show_demo_navigation()
    display_api_status()
    
    # Introduction section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### ✨ Transform Ideas into Engaging Content")
        st.markdown("""
        Our AI blog post generator creates professional, SEO-friendly content tailored to your specific needs:
        - **Topic-focused content** with relevant insights and examples
        - **Audience-specific tone** for maximum engagement
        - **Professional structure** with proper headings and flow
        - **SEO optimization** with natural keyword integration
        - **Call-to-action integration** to drive conversions
        """)
    
    with col2:
        st.markdown("### 🎯 Perfect For:")
        st.markdown("""
        - **Content Marketing**
        - **Thought Leadership** 
        - **SEO Strategy**
        - **Blog Consistency**
        - **Social Media Content**
        """)
    
    # Example topics
    with st.expander("💡 Need Topic Inspiration? Click for Examples", expanded=False):
        st.markdown("### 🚀 Technology Topics:")
        st.markdown("• The Future of Artificial Intelligence in Business • Cybersecurity Best Practices for Remote Teams • Blockchain Technology Explained Simply")
        
        st.markdown("### 💼 Business Topics:")
        st.markdown("• Building a Strong Company Culture • Digital Marketing Trends for 2024 • Effective Leadership in Remote Work")
        
        st.markdown("### 🎨 Creative Topics:")
        st.markdown("• The Psychology of Color in Branding • Content Creation Tips for Social Media • Building Your Personal Brand Online")
    
    # Input section
    st.markdown("---")
    st.markdown("## 📝 Blog Post Configuration")
    
    # Topic input
    topic_input = st.text_area(
        "🎯 Enter your blog post topic:",
        placeholder="e.g., 'The Benefits of AI in Customer Service' or 'How to Build a Successful Remote Team'",
        help="Be specific about what you want to write about. The more detailed your topic, the better the content will be.",
        height=80
    )
    
    # Configuration options
    st.markdown("### ⚙️ Content Customization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        blog_type = st.selectbox(
            "📄 Blog Post Type:",
            [
                "How-to Guide",
                "Industry Analysis", 
                "Opinion Piece",
                "Product Review",
                "Case Study",
                "Listicle",
                "Tutorial",
                "News Commentary"
            ],
            help="Choose the style and structure of your blog post"
        )
        
        tone = st.selectbox(
            "🎭 Writing Tone:",
            [
                "Professional",
                "Conversational", 
                "Educational",
                "Persuasive"
            ],
            help="Select the tone that best fits your brand and audience"
        )
    
    with col2:
        target_audience = st.selectbox(
            "👥 Target Audience:",
            [
                "General Public",
                "Business Professionals",
                "Industry Experts", 
                "Tech Enthusiasts"
            ],
            help="Who are you writing this blog post for?"
        )
        
        creativity = st.slider(
            "🎨 Creativity Level:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower = more focused and factual, Higher = more creative and varied"
        )
    
    # Validation and generation
    if topic_input:
        is_valid, error_message = validate_topic(topic_input)
        
        if not is_valid:
            st.error(error_message)
        else:
            # Show topic preview
            st.info(f"📋 **Topic**: {topic_input}")
            st.info(f"🎯 **Style**: {tone} {blog_type} for {target_audience}")
            
            # Generate button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                generate_button = st.button(
                    "🚀 Generate Blog Post",
                    type="primary",
                    use_container_width=True,
                    help="Create your AI-powered blog content"
                )
            
            if generate_button:
                # Show processing state
                with st.spinner('🤖 AI is crafting your blog post...'):
                    progress_bar = st.progress(0)
                    
                    for i in range(0, 101, 25):
                        progress_bar.progress(i)
                        time.sleep(0.3)
                    
                    # Generate the blog post
                    blog_content = generate_blog_post(
                        topic_input, 
                        blog_type, 
                        tone, 
                        target_audience, 
                        creativity
                    )
                    
                    progress_bar.empty()
                    
                    if blog_content:
                        # Display results
                        st.markdown("---")
                        st.markdown("## ✨ Your Generated Blog Post")
                        
                        # Blog content display
                        st.markdown("### 📄 Content:")
                        st.markdown(blog_content)
                        
                        # Analysis
                        analysis = analyze_blog_content(blog_content)
                        
                        st.markdown("### 📊 Content Analysis")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Word Count", f"{analysis['words']}")
                        
                        with col2:
                            st.metric("Reading Time", f"{analysis['reading_time']} min")
                        
                        with col3:
                            st.metric("Paragraphs", f"{analysis['paragraphs']}")
                        
                        with col4:
                            st.metric("Headings", f"{analysis['headings']}")
                        
                        # Quality indicators
                        st.markdown("### 🏆 Quality Check")
                        
                        quality_cols = st.columns(3)
                        
                        with quality_cols[0]:
                            intro_status = "✅" if analysis['has_introduction'] else "⚠️"
                            st.markdown(f"{intro_status} **Introduction**: {'Found' if analysis['has_introduction'] else 'Not detected'}")
                        
                        with quality_cols[1]:
                            conclusion_status = "✅" if analysis['has_conclusion'] else "⚠️"
                            st.markdown(f"{conclusion_status} **Conclusion**: {'Found' if analysis['has_conclusion'] else 'Not detected'}")
                        
                        with quality_cols[2]:
                            cta_status = "✅" if analysis['has_call_to_action'] else "⚠️"
                            st.markdown(f"{cta_status} **Call-to-Action**: {'Found' if analysis['has_call_to_action'] else 'Not detected'}")
                        
                        # Action buttons
                        st.markdown("### 🎯 Next Steps")
                        
                        action_cols = st.columns(4)
                        
                        with action_cols[0]:
                            if st.button("📋 Copy Content", help="Copy blog post for use"):
                                st.code(blog_content, language=None)
                                st.success("✅ Content ready to copy from above!")
                        
                        with action_cols[1]:
                            if st.button("🔄 Generate Another", help="Create a different version"):
                                st.info("💡 Adjust the settings above and click Generate again!")
                        
                        with action_cols[2]:
                            if st.button("📝 New Topic", help="Start with a different topic"):
                                st.info("💡 Enter a new topic in the text area above!")
                        
                        with action_cols[3]:
                            if st.button("🚀 Get Custom CMS", help="Learn about custom content solutions"):
                                st.info("🚀 [Contact GenAI Expertise](https://genaiexpertise.com/contact/) for custom content management solutions!")
                        
                        # SEO and social media tips
                        st.markdown("### 📈 Optimization Tips")
                        
                        tip_cols = st.columns(2)
                        
                        with tip_cols[0]:
                            st.markdown("""
                            **🔍 SEO Enhancement:**
                            - Add relevant keywords naturally
                            - Include meta description (150-160 chars)
                            - Use internal and external links
                            - Optimize images with alt text
                            - Include schema markup
                            """)
                        
                        with tip_cols[1]:
                            st.markdown("""
                            **📱 Social Media Ready:**
                            - Create compelling social snippets
                            - Design shareable quotes
                            - Add relevant hashtags
                            - Include engaging visuals
                            - Schedule optimal posting times
                            """)
    
    else:
        st.info("👆 Enter a topic above to generate your blog post!")
    
    # Add feedback section
    add_feedback_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 Scale Your Content Marketing
    
    This AI blog generator demonstrates just one aspect of intelligent content creation. We can build:
    
    - **Content Management Systems** with AI-powered writing assistance
    - **SEO Optimization Tools** for better search rankings
    - **Multi-language Content** for global audiences
    - **Brand Voice Consistency** across all content
    - **Automated Content Calendars** with scheduled generation
    
    **[Contact GenAI Expertise](https://genaiexpertise.com)** to build your custom content solution.
    """)

if __name__ == "__main__":
    main()