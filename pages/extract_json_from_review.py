import streamlit as st
import os
import time
import json
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
import re

from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

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
    page_title="Smart Review Analyzer",
    page_icon="🔍",
    layout="wide"
)

# Authentication check
if not check_password():
    st.stop()

class ReviewAnalyzer:
    """Enhanced review analyzer with comprehensive sentiment and data extraction."""
    
    def __init__(self):
        self.api_key = self.get_api_key()
        self.analysis_categories = {
            "sentiment": ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"],
            "price_perception": ["Very Expensive", "Expensive", "Fair", "Cheap", "Very Cheap"],
            "delivery_speed": ["Very Fast", "Fast", "Average", "Slow", "Very Slow"],
            "product_quality": ["Excellent", "Good", "Average", "Poor", "Very Poor"]
        }
    
    def get_api_key(self) -> str:
        """Get API key from environment or secrets."""
        return (
            os.environ.get("OPENAI_API_KEY") or 
            st.secrets.get("APIKEY", {}).get("OPENAI_API_KEY", "")
        )
    
    def validate_review(self, review_text: str) -> Tuple[bool, str]:
        """Validate the review input."""
        if not review_text.strip():
            return False, "Please enter a product review to analyze."
        
        if len(review_text.strip()) < 10:
            return False, "Review should be at least 10 characters long for meaningful analysis."
        
        word_count = len(review_text.split())
        if word_count > 1000:
            return False, f"Review is too long ({word_count} words). Maximum allowed: 1000 words."
        
        return True, ""
    
    def create_enhanced_template(self, analysis_type: str = "comprehensive") -> str:
        """Create enhanced analysis template based on type."""
        
        base_template = """You are an expert customer review analyst with extensive experience in e-commerce and customer sentiment analysis.

Your task is to analyze the following product review and extract key insights with high accuracy.

ANALYSIS REQUIREMENTS:
Extract and analyze the following information from the review:

1. SENTIMENT ANALYSIS:
   - Overall sentiment: Very Positive, Positive, Neutral, Negative, or Very Negative
   - Confidence level: How certain are you? (High/Medium/Low)
   - Key emotional indicators: What words/phrases indicate the sentiment?

2. DELIVERY ANALYSIS:
   - Delivery timeframe: Extract specific days/weeks if mentioned
   - Delivery satisfaction: Very Fast, Fast, Average, Slow, Very Slow, or Not Mentioned
   - Delivery issues: Any problems mentioned (damaged, late, etc.)

3. PRICE PERCEPTION:
   - Price sentiment: Very Expensive, Expensive, Fair, Cheap, Very Cheap, or Not Mentioned  
   - Value assessment: Does customer think it's worth the price?
   - Comparison mentions: Any price comparisons to competitors?

4. PRODUCT QUALITY:
   - Quality rating: Excellent, Good, Average, Poor, Very Poor, or Not Mentioned
   - Specific features mentioned: What aspects are praised/criticized?
   - Durability mentions: Any comments on longevity?

5. ADDITIONAL INSIGHTS:
   - Purchase occasion: Gift, personal use, business, etc.
   - Recommendation likelihood: Would they recommend? (Yes/No/Maybe)
   - Return/exchange mentions: Any issues with returns?
   - Customer type: First-time buyer, repeat customer, etc.

FORMAT YOUR RESPONSE AS JSON:
{{
    "sentiment": {{
        "overall": "[Very Positive/Positive/Neutral/Negative/Very Negative]",
        "confidence": "[High/Medium/Low]",
        "key_indicators": "[specific words/phrases]"
    }},
    "delivery": {{
        "timeframe": "[specific timeframe or 'Not mentioned']",
        "satisfaction": "[Very Fast/Fast/Average/Slow/Very Slow/Not Mentioned]",
        "issues": "[any delivery problems or 'None mentioned']"
    }},
    "price": {{
        "perception": "[Very Expensive/Expensive/Fair/Cheap/Very Cheap/Not Mentioned]",
        "value_assessment": "[Worth it/Overpriced/Good value/Not mentioned]",
        "comparisons": "[any competitor comparisons or 'None mentioned']"
    }},
    "quality": {{
        "rating": "[Excellent/Good/Average/Poor/Very Poor/Not Mentioned]",
        "features_praised": "[specific positive features]",
        "features_criticized": "[specific negative features]"
    }},
    "additional_insights": {{
        "purchase_occasion": "[Gift/Personal/Business/Not mentioned]",
        "recommendation": "[Yes/No/Maybe/Not mentioned]",
        "customer_type": "[First-time/Repeat/Not clear]"
    }}
}}

REVIEW TO ANALYZE: {review}

Provide your detailed JSON analysis:"""

        if analysis_type == "basic":
            return """For the following product review, extract key information in a structured format:

SENTIMENT: Is the customer satisfied? (Very Positive/Positive/Neutral/Negative/Very Negative)
DELIVERY: How long did delivery take? (Extract specific timeframe or "Not mentioned")
PRICE: How does the customer feel about pricing? (Very Expensive/Expensive/Fair/Cheap/Very Cheap/Not Mentioned)

Format as JSON:
{{
    "sentiment": "...",
    "delivery": "...", 
    "price": "..."
}}

Review: {review}

Analysis:"""
        
        return base_template
    
    def parse_analysis_result(self, result: str) -> Dict[str, Any]:
        """Parse and structure the analysis result."""
        try:
            # Try to parse as JSON first
            if result.strip().startswith('{') and result.strip().endswith('}'):
                return json.loads(result)
            
            # Fallback: extract key information manually
            analysis = {}
            
            # Extract sentiment
            sentiment_match = re.search(r'sentiment["\s]*:\s*["\s]*([^",\n]+)', result, re.IGNORECASE)
            if sentiment_match:
                analysis['sentiment'] = sentiment_match.group(1).strip('"')
            
            # Extract delivery
            delivery_match = re.search(r'delivery["\s]*:\s*["\s]*([^",\n]+)', result, re.IGNORECASE)
            if delivery_match:
                analysis['delivery'] = delivery_match.group(1).strip('"')
            
            # Extract price
            price_match = re.search(r'price["\s]*:\s*["\s]*([^",\n]+)', result, re.IGNORECASE)
            if price_match:
                analysis['price'] = price_match.group(1).strip('"')
            
            return analysis
            
        except Exception as e:
            return {"error": f"Failed to parse analysis: {str(e)}", "raw_result": result}
    
    def analyze_review(self, review_text: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Perform comprehensive review analysis."""
        
        if not self.api_key or not self.api_key.startswith("sk-"):
            raise Exception("Valid OpenAI API key required")
        
        try:
            # Initialize LLM
            llm = OpenAI(
                openai_api_key=self.api_key,
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1500
            )
            
            # Create prompt
            template = self.create_enhanced_template(analysis_type)
            prompt = PromptTemplate(
                input_variables=["review"],
                template=template
            )
            
            # Generate analysis
            formatted_prompt = prompt.format(review=review_text)
            result = llm(formatted_prompt)
            
            # Parse result
            parsed_analysis = self.parse_analysis_result(result)
            
            # Add metadata
            parsed_analysis['metadata'] = {
                'analysis_timestamp': datetime.now().isoformat(),
                'review_word_count': len(review_text.split()),
                'review_char_count': len(review_text),
                'analysis_type': analysis_type
            }
            
            return parsed_analysis
            
        except Exception as e:
            raise Exception(f"Analysis failed: {str(e)}")

def display_header():
    """Display the application header."""
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white; text-align: center;'>
        <h1>🔍 Smart Review Analyzer</h1>
        <p>Extract actionable insights from customer reviews with AI-powered analysis</p>
        <p><em>Comprehensive sentiment analysis and business intelligence extraction</em></p>
    </div>
    """, unsafe_allow_html=True)

def display_sample_reviews():
    """Display sample reviews for testing."""
    with st.expander("📝 Try These Sample Reviews", expanded=False):
        
        samples = {
            "Positive Review": """This laptop is absolutely fantastic! I ordered it on Monday and it arrived by Wednesday - super fast delivery. 
            The price was a bit high at first, but after using it for a week, I can say it's totally worth every penny. 
            The build quality is excellent, the screen is crisp, and the battery lasts all day. 
            I bought this for work and it handles everything I throw at it. Highly recommend!""",
            
            "Negative Review": """Very disappointed with this purchase. Took over 2 weeks to arrive and when it did, 
            the product was damaged. The price was way too high for the quality - I've seen better products for half the price. 
            Customer service was unhelpful when I tried to return it. The product feels cheap and broke after just a few days. 
            Would not recommend and will not be buying from this company again.""",
            
            "Mixed Review": """The product itself is decent - good quality materials and works as described. 
            However, the delivery was slower than expected (took about 10 days) and the price seems a bit steep. 
            It's not bad, but I'm not sure if I'd buy it again. The packaging was nice and it arrived in good condition. 
            Overall, it's okay but there might be better value options out there."""
        }
        
        col1, col2, col3 = st.columns(3)
        
        for i, (title, sample) in enumerate(samples.items()):
            with [col1, col2, col3][i]:
                st.markdown(f"**{title}:**")
                if st.button(f"Use {title}", key=f"sample_{i}", use_container_width=True):
                    st.session_state.sample_review = sample
                    st.rerun()

def display_analysis_results(analysis: Dict[str, Any]):
    """Display comprehensive analysis results."""
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    if "error" in analysis:
        st.error(f"Analysis Error: {analysis['error']}")
        if "raw_result" in analysis:
            with st.expander("Raw Analysis Output"):
                st.text(analysis["raw_result"])
        return
    
    # Main metrics
    if "sentiment" in analysis:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_data = analysis.get("sentiment", {})
            if isinstance(sentiment_data, dict):
                sentiment = sentiment_data.get("overall", "Not analyzed")
                confidence = sentiment_data.get("confidence", "Unknown")
            else:
                sentiment = str(sentiment_data)
                confidence = "Unknown"
            
            # Color code sentiment
            if "Positive" in sentiment:
                st.success(f"😊 **Sentiment**: {sentiment}")
            elif "Negative" in sentiment:
                st.error(f"😞 **Sentiment**: {sentiment}")
            else:
                st.info(f"😐 **Sentiment**: {sentiment}")
            
            st.caption(f"Confidence: {confidence}")
        
        with col2:
            delivery_data = analysis.get("delivery", {})
            if isinstance(delivery_data, dict):
                timeframe = delivery_data.get("timeframe", "Not mentioned")
                satisfaction = delivery_data.get("satisfaction", "Not mentioned")
            else:
                timeframe = str(delivery_data)
                satisfaction = "Unknown"
            
            st.info(f"🚚 **Delivery**: {timeframe}")
            st.caption(f"Satisfaction: {satisfaction}")
        
        with col3:
            price_data = analysis.get("price", {})
            if isinstance(price_data, dict):
                perception = price_data.get("perception", "Not mentioned")
                value = price_data.get("value_assessment", "Not mentioned")
            else:
                perception = str(price_data)
                value = "Unknown"
            
            # Color code price perception
            if "Cheap" in perception or "Fair" in perception:
                st.success(f"💰 **Price**: {perception}")
            elif "Expensive" in perception:
                st.warning(f"💰 **Price**: {perception}")
            else:
                st.info(f"💰 **Price**: {perception}")
            
            st.caption(f"Value: {value}")
    
    # Detailed insights
    if isinstance(analysis.get("sentiment"), dict):
        st.markdown("### 🎯 Detailed Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Sentiment Analysis:**")
            sentiment_data = analysis["sentiment"]
            st.markdown(f"- **Overall Sentiment**: {sentiment_data.get('overall', 'N/A')}")
            st.markdown(f"- **Confidence Level**: {sentiment_data.get('confidence', 'N/A')}")
            st.markdown(f"- **Key Indicators**: {sentiment_data.get('key_indicators', 'N/A')}")
            
            if "quality" in analysis:
                quality_data = analysis["quality"]
                st.markdown("**⭐ Product Quality:**")
                st.markdown(f"- **Rating**: {quality_data.get('rating', 'N/A')}")
                st.markdown(f"- **Praised Features**: {quality_data.get('features_praised', 'N/A')}")
                st.markdown(f"- **Criticized Features**: {quality_data.get('features_criticized', 'N/A')}")
        
        with col2:
            delivery_data = analysis.get("delivery", {})
            st.markdown("**🚚 Delivery Analysis:**")
            st.markdown(f"- **Timeframe**: {delivery_data.get('timeframe', 'N/A')}")
            st.markdown(f"- **Satisfaction**: {delivery_data.get('satisfaction', 'N/A')}")
            st.markdown(f"- **Issues**: {delivery_data.get('issues', 'N/A')}")
            
            price_data = analysis.get("price", {})
            st.markdown("**💰 Price Analysis:**")
            st.markdown(f"- **Perception**: {price_data.get('perception', 'N/A')}")
            st.markdown(f"- **Value Assessment**: {price_data.get('value_assessment', 'N/A')}")
            st.markdown(f"- **Comparisons**: {price_data.get('comparisons', 'N/A')}")
    
    # Additional insights
    if "additional_insights" in analysis:
        insights = analysis["additional_insights"]
        st.markdown("### 💡 Additional Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Purchase Occasion", insights.get("purchase_occasion", "Unknown"))
        
        with col2:
            recommendation = insights.get("recommendation", "Unknown")
            if recommendation == "Yes":
                st.success(f"Recommendation: {recommendation}")
            elif recommendation == "No":
                st.error(f"Recommendation: {recommendation}")
            else:
                st.info(f"Recommendation: {recommendation}")
        
        with col3:
            st.metric("Customer Type", insights.get("customer_type", "Unknown"))
    
    # Metadata
    if "metadata" in analysis:
        metadata = analysis["metadata"]
        st.markdown("### 📋 Analysis Metadata")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Word Count", metadata.get("review_word_count", 0))
        
        with col2:
            st.metric("Character Count", metadata.get("review_char_count", 0))
        
        with col3:
            analysis_time = metadata.get("analysis_timestamp", "Unknown")
            if analysis_time != "Unknown":
                try:
                    dt = datetime.fromisoformat(analysis_time.replace('Z', '+00:00'))
                    st.metric("Analyzed At", dt.strftime("%H:%M:%S"))
                except:
                    st.metric("Analyzed At", "Just now")
            else:
                st.metric("Analyzed At", "Unknown")

def main():
    """Main application logic."""
    
    display_header()
    show_demo_navigation()
    display_api_status()
    
    # Introduction
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎯 Transform Customer Feedback into Actionable Insights")
        st.markdown("""
        Our AI-powered review analyzer extracts comprehensive insights from customer feedback:
        - **Sentiment Analysis** - Understand customer satisfaction levels
        - **Delivery Assessment** - Track shipping performance and issues
        - **Price Perception** - Monitor pricing competitiveness and value
        - **Quality Insights** - Identify product strengths and weaknesses
        - **Business Intelligence** - Extract actionable data for decision-making
        """)
    
    with col2:
        st.markdown("### 💼 Business Benefits:")
        st.markdown("""
        - **Customer Insights**
        - **Product Improvement**
        - **Competitive Analysis**
        - **Quality Monitoring**
        - **Strategic Planning**
        """)
    
    # Sample reviews
    display_sample_reviews()
    
    # Configuration
    st.markdown("---")
    st.markdown("## ⚙️ Analysis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox(
            "📊 Analysis Depth:",
            ["comprehensive", "basic"],
            help="Comprehensive: Detailed multi-dimensional analysis. Basic: Quick sentiment, delivery, and price extraction."
        )
    
    with col2:
        st.markdown("""
        **📋 What We Extract:**
        - Sentiment & emotional indicators
        - Delivery timeframe & satisfaction
        - Price perception & value assessment
        - Product quality & feature feedback
        - Purchase context & recommendations
        """)
    
    # Review input
    st.markdown("---")
    st.markdown("## 📝 Enter Product Review")
    
    # Check for sample review
    default_review = st.session_state.get("sample_review", "")
    
    review_input = st.text_area(
        "Product Review Text:",
        value=default_review,
        placeholder="""Paste your product review here...

Example: "I absolutely love this product! It arrived in just 2 days and the quality exceeded my expectations. The price was fair for what you get, and I've already recommended it to my friends. Great purchase!"

Try the sample reviews above or enter your own review text.""",
        height=150,
        help="Enter customer review text to analyze. Maximum 1000 words."
    )
    
    # Clear sample review from session state after use
    if "sample_review" in st.session_state:
        del st.session_state.sample_review
    
    # Show review stats
    if review_input:
        word_count = len(review_input.split())
        char_count = len(review_input)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Words", word_count, delta=f"{1000-word_count} remaining")
        with col2:
            st.metric("Characters", char_count)
        with col3:
            if word_count > 1000:
                st.error(f"Reduce by {word_count-1000} words")
            else:
                st.success("Length OK ✅")
    
    # Analysis execution
    if review_input:
        analyzer = ReviewAnalyzer()
        is_valid, error_message = analyzer.validate_review(review_input)
        
        if not is_valid:
            st.error(error_message)
        else:
            # Analysis button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_button = st.button(
                    "🚀 Analyze Review",
                    type="primary",
                    use_container_width=True,
                    help="Extract comprehensive insights from the review"
                )
            
            if analyze_button:
                try:
                    with st.spinner("🤖 AI is analyzing the review..."):
                        progress_bar = st.progress(0)
                        
                        for i in range(0, 101, 25):
                            progress_bar.progress(i)
                            time.sleep(0.3)
                        
                        # Perform analysis
                        analysis_results = analyzer.analyze_review(review_input, analysis_type)
                        
                        progress_bar.empty()
                    
                    # Display results
                    display_analysis_results(analysis_results)
                    
                    # Action buttons
                    st.markdown("### 🎯 Next Steps")
                    
                    action_cols = st.columns(4)
                    
                    with action_cols[0]:
                        if st.button("📋 Export Analysis", help="Export analysis results"):
                            st.code(json.dumps(analysis_results, indent=2))
                            st.success("✅ Analysis ready to copy from above!")
                    
                    with action_cols[1]:
                        if st.button("🔄 Analyze Another", help="Clear and analyze new review"):
                            st.info("💡 Clear the text area above and enter a new review!")
                    
                    with action_cols[2]:
                        if st.button("📊 Try Samples", help="Use provided sample reviews"):
                            st.info("💡 Use the sample reviews provided above!")
                    
                    with action_cols[3]:
                        if st.button("🚀 Custom System", help="Build custom review analysis system"):
                            st.info("🚀 [Contact GenAI Expertise](https://genaiexpertise.com/contact/) for custom review analysis solutions!")
                
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("""
                    **Troubleshooting Tips:**
                    - Ensure the review contains meaningful content
                    - Check that the text is in English
                    - Try with a shorter review if it's very long
                    - Verify your internet connection
                    """)
    
    else:
        st.info("👆 Enter a product review above to analyze!")
    
    # Add feedback section
    add_feedback_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 Ready for Enterprise Review Analytics?
    
    This analyzer demonstrates advanced review processing capabilities. We can build:
    
    - **Bulk Review Processing** - Analyze thousands of reviews automatically
    - **Real-time Monitoring** - Track review sentiment as they come in
    - **Custom Categories** - Extract insights specific to your industry
    - **Competitor Analysis** - Compare your reviews against competitors
    - **Dashboard Integration** - Connect with your existing analytics systems
    
    **[Contact GenAI Expertise](https://genaiexpertise.com)** to build your custom review intelligence platform.
    """)

if __name__ == "__main__":
    main()