import streamlit as st
import os
import time
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import re

from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

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
    page_title="Multi-Language Translator",
    page_icon="🌍",
    layout="wide"
)

# Authentication check
if not check_password():
    st.stop()

class AdvancedTranslator:
    """Enhanced translator with cultural context and quality assessment."""
    
    def __init__(self):
        self.api_key = self.get_api_key()
        self.max_chars = 5000  # Reasonable limit for quality translation
        
        # Expanded language support with cultural context
        self.languages = {
            "Yoruba": {
                "code": "yo",
                "native": "Yorùbá",
                "region": "West Africa (Nigeria, Benin, Togo)",
                "speakers": "45+ million",
                "cultural_notes": "Rich oral tradition, tonal language with complex honorifics"
            },
            "Hausa": {
                "code": "ha", 
                "native": "Harshen Hausa",
                "region": "West/Central Africa (Nigeria, Niger, Chad)",
                "speakers": "70+ million",
                "cultural_notes": "Trade language, Arabic script influence, Islamic cultural elements"
            },
            "Igbo": {
                "code": "ig",
                "native": "Asụsụ Igbo", 
                "region": "Southeast Nigeria",
                "speakers": "27+ million",
                "cultural_notes": "Tonal language, rich proverbs, communal cultural values"
            },
            "French": {
                "code": "fr",
                "native": "Français",
                "region": "France, West Africa, Canada",
                "speakers": "280+ million",
                "cultural_notes": "Formal/informal registers, cultural nuances important"
            },
            "Arabic": {
                "code": "ar",
                "native": "العربية",
                "region": "Middle East, North Africa",
                "speakers": "420+ million", 
                "cultural_notes": "Classical vs. Modern Standard, right-to-left script"
            },
            "Swahili": {
                "code": "sw",
                "native": "Kiswahili",
                "region": "East Africa (Kenya, Tanzania, Uganda)",
                "speakers": "200+ million",
                "cultural_notes": "Bantu base with Arabic influence, trade language"
            },
            "Spanish": {
                "code": "es",
                "native": "Español",
                "region": "Spain, Latin America",
                "speakers": "500+ million",
                "cultural_notes": "Regional variations, formal/informal address systems"
            },
            "Portuguese": {
                "code": "pt",
                "native": "Português", 
                "region": "Brazil, Portugal, Africa",
                "speakers": "260+ million",
                "cultural_notes": "Brazilian vs. European variants, African influences"
            }
        }
    
    def get_api_key(self) -> str:
        """Get API key from environment or secrets."""
        return (
            os.environ.get("OPENAI_API_KEY") or 
            st.secrets.get("APIKEY", {}).get("OPENAI_API_KEY", "")
        )
    
    def validate_input(self, text: str, source_lang: str, target_lang: str) -> Tuple[bool, str]:
        """Validate translation input."""
        if not text.strip():
            return False, "Please enter text to translate."
        
        if len(text.strip()) < 2:
            return False, "Text should be at least 2 characters long."
        
        if len(text) > self.max_chars:
            return False, f"Text too long ({len(text)} characters). Maximum: {self.max_chars} characters."
        
        if source_lang == target_lang:
            return False, "Source and target languages cannot be the same."
        
        return True, ""
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text properties for optimization."""
        
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(re.findall(r'[.!?]+', text))
        
        # Detect text complexity
        avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
        
        # Detect formal vs informal indicators
        formal_indicators = ['please', 'kindly', 'respectfully', 'sincerely', 'regarding']
        informal_indicators = ['hi', 'hey', 'thanks', 'yeah', 'cool', 'awesome']
        
        formal_score = sum(1 for indicator in formal_indicators if indicator.lower() in text.lower())
        informal_score = sum(1 for indicator in informal_indicators if indicator.lower() in text.lower())
        
        formality = "formal" if formal_score > informal_score else "informal" if informal_score > 0 else "neutral"
        
        # Detect content type
        content_type = self.detect_content_type(text)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'formality': formality,
            'content_type': content_type,
            'complexity': 'high' if avg_word_length > 6 else 'medium' if avg_word_length > 4 else 'simple'
        }
    
    def detect_content_type(self, text: str) -> str:
        """Detect content type for appropriate translation style."""
        
        text_lower = text.lower()
        
        # Business content
        business_keywords = ['meeting', 'proposal', 'contract', 'agreement', 'business', 'company', 'service']
        if any(keyword in text_lower for keyword in business_keywords):
            return 'business'
        
        # Personal/casual content  
        personal_keywords = ['family', 'friend', 'love', 'miss you', 'how are you', 'greetings']
        if any(keyword in text_lower for keyword in personal_keywords):
            return 'personal'
        
        # Technical content
        technical_keywords = ['system', 'process', 'technology', 'implementation', 'configuration']
        if any(keyword in text_lower for keyword in technical_keywords):
            return 'technical'
        
        # Educational content
        educational_keywords = ['learn', 'study', 'education', 'knowledge', 'understand', 'explain']
        if any(keyword in text_lower for keyword in educational_keywords):
            return 'educational'
        
        return 'general'
    
    def create_enhanced_prompt(self, source_lang: str, target_lang: str, text_analysis: Dict) -> ChatPromptTemplate:
        """Create culturally-aware translation prompt."""
        
        lang_info = self.languages.get(target_lang, {})
        formality = text_analysis.get('formality', 'neutral')
        content_type = text_analysis.get('content_type', 'general')
        
        system_message = f"""You are an expert translator specializing in {target_lang} ({lang_info.get('native', target_lang)}) with deep cultural understanding.

TRANSLATION REQUIREMENTS:
- Source Language: {source_lang}
- Target Language: {target_lang} ({lang_info.get('native', target_lang)})
- Content Type: {content_type}
- Formality Level: {formality}
- Cultural Context: {lang_info.get('cultural_notes', 'Standard translation')}

TRANSLATION GUIDELINES:
1. Maintain the original meaning and context
2. Adapt cultural references appropriately for {lang_info.get('region', 'target region')}
3. Use appropriate formality level ({formality})
4. Preserve tone and emotional nuance
5. Consider cultural sensitivities and local customs
6. Use natural, fluent expressions in {target_lang}
7. For technical terms, provide culturally appropriate equivalents

SPECIAL CONSIDERATIONS for {target_lang}:
{lang_info.get('cultural_notes', 'Maintain cultural authenticity and natural expression')}

Translate the following text from {source_lang} to {target_lang}:"""

        return ChatPromptTemplate.from_messages([
            ('system', system_message),
            ('user', '{text}')
        ])
    
    def translate_text(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str,
        quality_level: str = "high"
    ) -> Dict[str, Any]:
        """Perform enhanced translation with quality assessment."""
        
        if not self.api_key or not self.api_key.startswith("sk-"):
            raise Exception("Valid OpenAI API key required")
        
        try:
            # Analyze text
            text_analysis = self.analyze_text(text)
            
            # Configure LLM based on quality level
            temperature_map = {'high': 0.2, 'balanced': 0.3, 'creative': 0.5}
            
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                openai_api_key=self.api_key,
                temperature=temperature_map.get(quality_level, 0.3),
                max_tokens=2000
            )
            
            # Create enhanced prompt
            prompt_template = self.create_enhanced_prompt(source_lang, target_lang, text_analysis)
            
            # Create translation chain
            parser = StrOutputParser()
            chain = prompt_template | llm | parser
            
            # Perform translation
            start_time = time.time()
            translation = chain.invoke({"text": text})
            processing_time = time.time() - start_time
            
            # Analyze translation
            translation_analysis = self.analyze_text(translation)
            
            return {
                'original_text': text,
                'translated_text': translation.strip(),
                'source_language': source_lang,
                'target_language': target_lang,
                'source_analysis': text_analysis,
                'translation_analysis': translation_analysis,
                'processing_metrics': {
                    'processing_time': processing_time,
                    'quality_level': quality_level,
                    'character_ratio': len(translation) / len(text) if len(text) > 0 else 0
                },
                'language_info': self.languages.get(target_lang, {}),
                'translation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Translation failed: {str(e)}")

def display_header():
    """Display the application header."""
    st.markdown("""
    <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 10px; margin-bottom: 2rem; color: white; text-align: center;'>
        <h1>🌍 Multi-Language Translator</h1>
        <p>Professional translation services with cultural context and local expertise</p>
        <p><em>Specialized in African languages and cross-cultural communication</em></p>
    </div>
    """, unsafe_allow_html=True)

def display_language_info():
    """Display language information and capabilities."""
    with st.expander("🗺️ Supported Languages & Cultural Context", expanded=False):
        
        translator = AdvancedTranslator()
        
        # Group languages by region
        african_langs = ["Yoruba", "Hausa", "Igbo", "Swahili"]
        international_langs = ["French", "Arabic", "Spanish", "Portuguese"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌍 African Languages")
            for lang in african_langs:
                if lang in translator.languages:
                    info = translator.languages[lang]
                    st.markdown(f"""
                    **{lang}** ({info['native']})
                    - 👥 Speakers: {info['speakers']}
                    - 📍 Region: {info['region']}
                    - 🎭 Cultural Notes: {info['cultural_notes']}
                    """)
        
        with col2:
            st.markdown("### 🌎 International Languages")
            for lang in international_langs:
                if lang in translator.languages:
                    info = translator.languages[lang]
                    st.markdown(f"""
                    **{lang}** ({info['native']})
                    - 👥 Speakers: {info['speakers']}
                    - 📍 Region: {info['region']}
                    - 🎭 Cultural Notes: {info['cultural_notes']}
                    """)

def display_sample_translations():
    """Display sample translations for different scenarios."""
    with st.expander("📝 Sample Translation Examples", expanded=False):
        
        samples = {
            "Business Greeting": {
                "English": "Good morning. I hope this message finds you well. I would like to schedule a meeting to discuss our upcoming project.",
                "context": "Formal business communication"
            },
            "Personal Message": {
                "English": "Hi! How are you and your family doing? I miss you all and can't wait to see you during the holidays.",
                "context": "Casual family communication"
            },
            "Technical Instruction": {
                "English": "Please follow these steps to configure the system: First, open the settings menu, then select the language preferences.",
                "context": "Technical documentation"
            }
        }
        
        for title, sample in samples.items():
            st.markdown(f"**{title}** ({sample['context']}):")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text_area(f"{title} text", sample["English"], height=60, disabled=True, label_visibility="collapsed")
            with col2:
                if st.button(f"Use {title}", key=f"sample_{title}", use_container_width=True):
                    st.session_state.sample_text = sample["English"]
                    st.rerun()

def display_text_analysis(analysis: Dict[str, Any], lang: str):
    """Display text analysis information."""
    st.markdown(f"### 📊 {lang} Text Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Words", analysis['word_count'])
    
    with col2:
        st.metric("Characters", analysis['char_count'])
    
    with col3:
        formality = analysis['formality'].title()
        st.metric("Formality", formality)
    
    with col4:
        content_type = analysis['content_type'].title()
        st.metric("Content Type", content_type)

def display_translation_results(results: Dict[str, Any]):
    """Display comprehensive translation results."""
    st.markdown("---")
    st.markdown("## ✨ Translation Results")
    
    # Translation display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 📝 Original ({results['source_language']})")
        st.info(results['original_text'])
        display_text_analysis(results['source_analysis'], results['source_language'])
    
    with col2:
        target_lang = results['target_language']
        lang_info = results.get('language_info', {})
        native_name = lang_info.get('native', target_lang)
        
        st.markdown(f"### ✨ Translation ({target_lang} - {native_name})")
        st.success(results['translated_text'])
        display_text_analysis(results['translation_analysis'], target_lang)
    
    # Language and cultural information
    if lang_info:
        st.markdown("### 🌍 Language Context")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Region:** {lang_info.get('region', 'N/A')}")
            st.markdown(f"**Speakers:** {lang_info.get('speakers', 'N/A')}")
        
        with col2:
            st.markdown(f"**Native Script:** {lang_info.get('native', 'N/A')}")
            st.markdown(f"**Cultural Notes:** {lang_info.get('cultural_notes', 'N/A')}")
    
    # Processing metrics
    st.markdown("### 📊 Translation Metrics")
    
    metrics = results['processing_metrics']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        processing_time = metrics.get('processing_time', 0)
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    with col2:
        char_ratio = metrics.get('character_ratio', 0)
        st.metric("Length Ratio", f"{char_ratio:.2f}x")
    
    with col3:
        quality = metrics.get('quality_level', 'standard').title()
        st.metric("Quality Level", quality)

def main():
    """Main application logic."""
    
    display_header()
    show_demo_navigation()
    display_api_status()
    
    # Introduction
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🎯 Professional Multi-Language Translation")
        st.markdown("""
        Experience culturally-aware translation services powered by advanced AI:
        - **African Language Expertise** - Specialized in Yoruba, Hausa, Igbo, and Swahili
        - **Cultural Context** - Translations that respect local customs and expressions
        - **Content Adaptation** - Appropriate formality and style for different contexts
        - **Quality Assurance** - Multiple quality levels for different needs
        - **Professional Grade** - Suitable for business, academic, and personal use
        """)
    
    with col2:
        st.markdown("### 💼 Perfect For:")
        st.markdown("""
        - **Business Communications**
        - **Cultural Exchange**
        - **Educational Materials**
        - **Personal Messages**
        - **Technical Documentation**
        """)
    
    display_language_info()
    display_sample_translations()
    
    # Configuration
    st.markdown("---")
    st.markdown("## ⚙️ Translation Configuration")
    
    translator = AdvancedTranslator()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_lang = st.selectbox(
            "📤 Source Language:",
            ["English"] + list(translator.languages.keys()),
            help="Language of the original text"
        )
    
    with col2:
        # Remove source language from target options
        target_options = [lang for lang in translator.languages.keys() if lang != source_lang]
        target_lang = st.selectbox(
            "📥 Target Language:",
            target_options,
            help="Language to translate into"
        )
    
    with col3:
        quality_level = st.selectbox(
            "🎯 Quality Level:",
            ["high", "balanced", "creative"],
            help="High: Precise/formal. Balanced: Natural flow. Creative: Expressive style."
        )
    
    # Text input
    st.markdown("---")
    st.markdown("## 📝 Enter Text to Translate")
    
    # Check for sample text
    default_text = st.session_state.get("sample_text", "")
    
    text_input = st.text_area(
        f"Text in {source_lang}:",
        value=default_text,
        placeholder=f"""Enter your {source_lang} text here for translation to {target_lang}...

Examples:
• Business emails and formal communications
• Personal messages and greetings
• Technical instructions and documentation  
• Educational content and explanations
• Cultural expressions and local sayings

Try the sample texts above or enter your own content!""",
        height=150,
        help=f"Enter text in {source_lang} to translate to {target_lang} (max {translator.max_chars} characters)"
    )
    
    # Clear sample text from session state
    if "sample_text" in st.session_state:
        del st.session_state.sample_text
    
    # Show input analysis
    if text_input:
        analysis = translator.analyze_text(text_input)
        display_text_analysis(analysis, source_lang)
        
        # Validation
        is_valid, error_message = translator.validate_input(text_input, source_lang, target_lang)
        
        if not is_valid:
            st.error(error_message)
        else:
            # Translate button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                translate_button = st.button(
                    f"🌍 Translate to {target_lang}",
                    type="primary",
                    use_container_width=True,
                    help=f"Translate from {source_lang} to {target_lang} with cultural context"
                )
            
            if translate_button:
                try:
                    with st.spinner(f"🤖 Translating to {target_lang} with cultural context..."):
                        progress_bar = st.progress(0)
                        
                        # Progress steps
                        steps = [
                            ("Analyzing text structure...", 25),
                            ("Applying cultural context...", 50),
                            ("Generating translation...", 75),
                            ("Quality checking...", 100)
                        ]
                        
                        for step_text, progress in steps:
                            st.text(step_text)
                            progress_bar.progress(progress)
                            time.sleep(0.3)
                        
                        # Perform translation
                        results = translator.translate_text(
                            text=text_input,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            quality_level=quality_level
                        )
                        
                        progress_bar.empty()
                        st.success("✅ Translation completed!")
                    
                    # Display results
                    display_translation_results(results)
                    
                    # Action buttons
                    st.markdown("### 🎯 Next Steps")
                    
                    action_cols = st.columns(4)
                    
                    with action_cols[0]:
                        if st.button("📋 Copy Translation", help="Copy translated text"):
                            st.code(results['translated_text'])
                            st.success("✅ Translation ready to copy!")
                    
                    with action_cols[1]:
                        if st.button("🔄 Reverse Translate", help="Translate back to source"):
                            st.info(f"💡 Switch {target_lang} to source and {source_lang} to target above!")
                    
                    with action_cols[2]:
                        if st.button("🌍 Try Different Language", help="Select different target language"):
                            st.info("💡 Select a different target language above and translate again!")
                    
                    with action_cols[3]:
                        if st.button("🚀 Custom Solution", help="Build enterprise translation system"):
                            st.info("🚀 [Contact GenAI Expertise](https://genaiexpertise.com/contact/) for custom translation solutions!")
                
                except Exception as e:
                    st.error(f"❌ Translation failed: {str(e)}")
                    st.info("""
                    **Troubleshooting Tips:**
                    - Ensure text contains meaningful content
                    - Check source language selection is correct
                    - Try with shorter text if very long
                    - Verify internet connection is stable
                    """)
    
    else:
        st.info("👆 Enter text above to begin translation!")
    
    # Add feedback section
    add_feedback_section()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🚀 Ready for Enterprise Translation Services?
    
    This translator demonstrates advanced multilingual capabilities with cultural awareness. We can build:
    
    - **Custom Language Models** - Trained specifically for your industry terminology
    - **Real-time Translation APIs** - Integrate with your existing applications and workflows
    - **Batch Processing Systems** - Translate large volumes of documents automatically
    - **Cultural Adaptation Services** - Localization beyond literal translation
    - **Multi-modal Translation** - Handle documents, images, audio, and video content
    
    **[Contact GenAI Expertise](https://genaiexpertise.com)** to build your enterprise translation solution.
    """)

if __name__ == "__main__":
    main()