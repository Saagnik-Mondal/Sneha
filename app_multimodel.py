import streamlit as st
import os
import re
import sys
import io
import contextlib
import time
import torch
from enum import Enum
from llama_cpp import Llama
from llama_index.core import Settings
from llama_index.core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set page configuration
st.set_page_config(
    page_title="SNEHA - Offline Healthcare Assistant",
    page_icon="👩‍⚕️",
    layout="wide"
)

# Constants
# Model paths
GENERAL_MODEL_PATH = "/Volumes/X9 Pro/LMModels/TheBloke/zephyr-7B-beta-GGUF/zephyr-7b-beta.Q4_0.gguf"
SMALL_MODEL_PATH = "/Volumes/X9 Pro/LMModels/TheBloke/phi-2-GGUF/phi-2.Q8_0.gguf"
MEDICAL_MODEL_PATH = "/Volumes/X9 Pro/LMModels/mradermacher/Med-Alpaca-2-7b-chat-GGUF/Med-Alpaca-2-7b-chat.Q3_K_M.gguf"
SUPPORT_MODEL_PATH = "/Volumes/X9 Pro/LMModels/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/openhermes-2.5-mistral-7b.Q3_K_S.gguf"

SNEHA_VERSION = "2.0.0"  # Updated version number for multi-model capabilities
SNEHA_CODENAME = "SNEHA MultiModel"  # Updated internal development codename
SNEHA_RELEASE_DATE = "April 19, 2025"  # Today's date for the latest version
DISCLAIMER = "⚠️ SNEHA is an AI assistant. She does not provide medical diagnoses or treatments. However, SNEHA is not a licensed medical professional, and the information provided by the assistant should not be used as a substitute for professional medical advice, diagnosis, or treatment. Please consult a licensed doctor for serious health concerns."

# Custom CSS with animations and modern styling
st.markdown("""
<style>
    /* Base styling */
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Welcome screen animation */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes floatIcon {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh;
        text-align: center;
        animation: fadeIn 1.5s ease-in-out;
    }
    
    .welcome-logo {
        margin-bottom: 2rem;
        animation: pulse 2s infinite ease-in-out, floatIcon 3s infinite ease-in-out;
    }
    
    .welcome-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #2e7d32;
        animation: slideUp 1s ease-out;
    }
    
    .welcome-subtitle {
        font-size: 1.8rem;
        color: #546e7a;
        margin-bottom: 1.5rem;
        animation: slideUp 1s ease-out 0.3s;
        animation-fill-mode: both;
    }
    
    .welcome-author {
        font-size: 1.2rem;
        color: #78909c;
        animation: slideUp 1s ease-out 0.6s;
        animation-fill-mode: both;
    }
    
    .start-button {
        margin-top: 2rem;
        animation: slideUp 1s ease-out 0.9s;
        animation-fill-mode: both;
    }
    
    /* App styling */
    .app-header {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .app-header:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .app-header h1 {
        color: #2e7d32;
        margin-bottom: 0.5rem;
    }
    
    .disclaimer {
        background-color: #fff8e1;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1.5rem;
        font-size: 0.9rem;
        border-left: 4px solid #ffc107;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .sneha-version {
        color: #78909c;
        font-size: 0.9rem;
        text-align: right;
        margin-top: 5px;
    }
    
    .sneha-response {
        background-color: #f5f7fa;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        animation: fadeIn 0.5s ease-out;
    }
    
    .stTextArea>div>div>textarea {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 10px !important;
        font-size: 1rem !important;
        transition: all 0.3s ease;
    }
    
    .stTextArea>div>div>textarea:focus {
        border-color: #4caf50 !important;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2) !important;
    }
    
    .stButton>button {
        background-color: #4caf50 !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
        border-radius: 50px !important;
        border: none !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        background-color: #43a047 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f5f7fa !important;
    }
    
    .sidebar-content {
        padding: 1rem;
        background: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    .sidebar-feature {
        display: flex;
        align-items: center;
        margin-bottom: 0.8rem;
        padding: 0.5rem;
        background-color: #f1f8e9;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .sidebar-feature:hover {
        background-color: #e8f5e9;
        transform: translateX(5px);
    }
    
    .feature-icon {
        margin-right: 0.5rem;
        color: #4caf50;
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-dot {
        width: 12px;
        height: 12px;
        background: #4caf50;
        border-radius: 50%;
        margin: 0 6px;
        display: inline-block;
    }
    
    .loading-dot:nth-child(1) {
        animation: pulse 0.6s ease-in-out infinite alternate;
        animation-delay: 0s;
    }
    
    .loading-dot:nth-child(2) {
        animation: pulse 0.6s ease-in-out infinite alternate;
        animation-delay: 0.2s;
    }
    
    .loading-dot:nth-child(3) {
        animation: pulse 0.6s ease-in-out infinite alternate;
        animation-delay: 0.4s;
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Model badge styles */
    .model-badge {
        display: inline-block;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 10px;
        margin-left: 8px;
        font-weight: bold;
    }
    .model-general {
        background-color: #e3f2fd;
        color: #1976d2;
    }
    .model-medical {
        background-color: #e8f5e9;
        color: #388e3c;
    }
    .model-research {
        background-color: #f3e5f5;
        color: #7b1fa2;
    }
    .model-support {
        background-color: #fff3e0;
        color: #e65100;
    }
    .model-small {
        background-color: #f1f1f1;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

def sanitize_response(response_text):
    """Remove any doctor names, hospital names, or locations from the response."""
    
    # Remove common opening phrases that sound like form letters
    opening_phrases = [
        r"Thank you for writing to us at .+\.",
        r"Thank you for reaching out to .+\.",
        r"Thank you for contacting .+\.",
        r"Thanks for reaching out to .+\."
    ]
    for phrase in opening_phrases:
        response_text = re.sub(phrase, "", response_text, flags=re.IGNORECASE)
    
    # Replace doctor names with "healthcare provider"
    response_text = re.sub(r'Dr\.\s+[A-Za-z]+', 'healthcare provider', response_text)
    
    # Generic replacements
    replacements = {
        "hospital": "medical facility",
        "clinic": "medical facility", 
        "center": "medical facility",
        "healthcaremagic": "", 
        "healthcare magic": "",
        "health care magic": "",
        "medical service": "",
        "medical platform": "",
        "our team": "",
        "our medical team": "",
        "our healthcare team": "",
        "our experts": "",
        "our staff": ""
    }
    
    for term, replacement in replacements.items():
        response_text = re.sub(fr'\b{term}\b', replacement, response_text, flags=re.IGNORECASE)
    
    # Remove any remaining references to specific healthcare services
    response_text = re.sub(r'at\s+[A-Z][A-Za-z\s]+(Health|Medical|Care|Clinic)', '', response_text)
    
    # Clean up any double spaces or leading spaces after sanitization
    response_text = re.sub(r'\s{2,}', ' ', response_text)
    response_text = re.sub(r'^\s+', '', response_text)
    
    return response_text

@contextlib.contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stderr(fnull) as err, contextlib.redirect_stdout(fnull) as out:
            yield (err, out)

class ModelType(Enum):
    """Enum for different model types with specific purposes"""
    GENERAL = "general"         # Zephyr-7B: General, friendly conversation
    SMALL = "small"             # Phi-2: Fast, small footprint chatbot
    MEDICAL = "medical"         # Med-Alpaca-2: Medical Q&A
    SUPPORT = "support"         # OpenHermes: Empathetic, safe general support
    RESEARCH = "research"       # BioGPT: Biomedical research, academic knowledge

def initialize_models():
    """Initialize multiple models for different purposes in SNEHA."""
    models = {}
    errors = []
    
    # Setting environment variable to optimize for CPU
    os.environ["LLAMA_METAL"] = "0"  # Disable Metal (Apple GPU) support
    os.environ["LLAMA_CUBLAS"] = "0" # Disable CUDA
    
    # Initialize GGUF models with llama.cpp
    model_paths = {
        ModelType.GENERAL: GENERAL_MODEL_PATH,
        ModelType.SMALL: SMALL_MODEL_PATH,
        ModelType.MEDICAL: MEDICAL_MODEL_PATH,
        ModelType.SUPPORT: SUPPORT_MODEL_PATH,
    }
    
    for model_type, model_path in model_paths.items():
        if not os.path.exists(model_path):
            errors.append(f"Model file not found at: {model_path}")
            continue
        
        try:
            with suppress_stdout_stderr():
                # Initialize model with optimized parameters
                model = Llama(
                    model_path=model_path,
                    n_ctx=512,               # Context window for prompt + response
                    n_gpu_layers=0,          # CPU-only for reliability
                    n_threads=3,             # Reduced thread count to avoid overwhelming CPU
                    n_batch=128,             # Batch size for faster initial responses
                    use_mlock=True,          # Keep model in memory
                    use_mmap=True,           # Use memory mapping for faster loading
                    verbose=False,           # Reduce logging overhead
                    seed=42,                 # Fixed seed for deterministic responses
                )
                models[model_type] = model
        except Exception as e:
            errors.append(f"Error initializing {model_type.value} model: {str(e)}")
    
    # Initialize BioGPT with transformers
    try:
        with suppress_stdout_stderr():
            tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
            bio_model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
            models[ModelType.RESEARCH] = {"model": bio_model, "tokenizer": tokenizer}
    except Exception as e:
        errors.append(f"Error initializing BioGPT model: {str(e)}")
    
    if errors:
        st.error("\n".join(errors))
        if not models:
            st.stop()
    
    return models

def select_model_for_query(query, models):
    """Select the most appropriate model based on query content, with specialized routing based on query type."""
    query_lower = query.lower()
    
    # 1. Mental Health/Emotional Support (OpenHermes) - Enhanced detection for support needs
    mental_health_keywords = [
        # Emotional states
        "anxious", "worried", "scared", "afraid", "stressed", "depressed", "sad", "upset", 
        "nervous", "concerned", "help me", "feeling down", "mental health", "anxiety", 
        "depression", "panic", "suicidal", "hopeless", "overwhelmed", "trauma", "ptsd",
        "grief", "lonely", "alone", "isolated", "worthless", "exhausted", "burnout",
        
        # Support seeking phrases
        "need someone to talk", "can't cope", "therapy", "therapist", "counseling", 
        "feeling lost", "no one understands", "don't know what to do", "can't take it anymore",
        "emotional support", "hurting inside", "need help with feelings"
    ]
    
    # Check for crisis signals that need careful handling
    crisis_signals = ["suicidal", "kill myself", "end my life", "don't want to live", "harming myself", "self-harm"]
    if any(signal in query_lower for signal in crisis_signals):
        if ModelType.SUPPORT in models:
            return models[ModelType.SUPPORT], ModelType.SUPPORT
        else:
            # Fallback to GENERAL model if SUPPORT is not available
            return models[ModelType.GENERAL], ModelType.GENERAL
    
    if any(keyword in query_lower for keyword in mental_health_keywords):
        if ModelType.SUPPORT in models:
            return models[ModelType.SUPPORT], ModelType.SUPPORT
        else:
            # Fallback to GENERAL model if SUPPORT is not available
            return models[ModelType.GENERAL], ModelType.GENERAL
    
    # 2. Medical Q&A (Med-Alpaca) - Enhanced for clinical information and health advice
    medical_keywords = [
        # Common conditions and symptoms
        "diagnosis", "treatment", "symptom", "disease", "condition", "medication",
        "prescription", "doctor", "hospital", "pain", "fever", "nausea", "headache",
        "blood", "heart", "brain", "lung", "kidney", "liver", "infection", "diabetes", 
        "allergy", "allergic", "asthma", "cancer", "tumor", "chronic", "arthritis", 
        "hypertension", "inflammation", "surgery", "wound", "injury", "fracture",
        
        # Medical questions
        "should i see a doctor", "medical advice", "health problem", "side effect",
        "taking medication", "drug interaction", "dosage", "underlying cause"
    ]
    
    if any(keyword in query_lower for keyword in medical_keywords):
        if ModelType.MEDICAL in models:
            return models[ModelType.MEDICAL], ModelType.MEDICAL
        else:
            # Try to fall back to any available model in priority order
            for fallback_type in [ModelType.GENERAL, ModelType.SUPPORT, ModelType.SMALL]:
                if fallback_type in models:
                    return models[fallback_type], fallback_type
            # Ultimate fallback - first available model
            if models:
                first_model_type = list(models.keys())[0]
                return models[first_model_type], first_model_type
    
    # 3. Research/Medical Literature (BioGPT) - Academic and research-based questions
    research_keywords = [
        # Academic/research terms
        "research", "study", "paper", "journal", "publication", "clinical trial", 
        "evidence", "literature", "scientific", "mechanism", "pathophysiology",
        "published", "peer-reviewed", "meta-analysis", "systematic review", "data", 
        "statistics", "findings", "conclusion", "methodology", "protocol",
        
        # Scientific understanding
        "molecular", "cellular", "pathway", "gene", "genetic", "biomarker",
        "etiology", "physiology", "pharmacology", "epidemiology", "prevalence",
        
        # Research questions
        "latest research", "recent studies", "evidence-based", "according to research",
        "scientific consensus", "breakthrough", "innovation", "discovery"
    ]
    
    if any(keyword in query_lower for keyword in research_keywords):
        if ModelType.RESEARCH in models:
            return models[ModelType.RESEARCH], ModelType.RESEARCH
        else:
            # Try to fall back to any available model in priority order
            for fallback_type in [ModelType.MEDICAL, ModelType.GENERAL, ModelType.SUPPORT]:
                if fallback_type in models:
                    return models[fallback_type], fallback_type
            # Ultimate fallback - first available model
            if models:
                first_model_type = list(models.keys())[0]
                return models[first_model_type], first_model_type
    
    # 4. Daily Health Coach/Conversational (Zephyr) - Lifestyle and wellness conversations
    lifestyle_keywords = [
        # Daily habits
        "exercise", "workout", "fitness", "nutrition", "diet", "eating", "food",
        "sleep", "rest", "water", "hydration", "meditation", "mindfulness", "yoga",
        "routine", "habit", "lifestyle", "wellness", "wellbeing", "healthy living",
        
        # Conversational health queries
        "what should i eat", "how much water", "better sleep", "improve my health",
        "stay healthy", "feel better", "energy level", "tired all the time", 
        "daily routine", "healthy habits", "weight management", "strength training",
        "cardio", "stretching", "motivate"
    ]
    
    if any(keyword in query_lower for keyword in lifestyle_keywords):
        return models[ModelType.GENERAL], ModelType.GENERAL
    
    # 5. Quick Triage/Simple Answers (Phi-2) - Short, direct queries needing fast responses
    # Short queries and simple yes/no questions (already implemented)
    if len(query.split()) <= 10 or query.lower().endswith("?") and len(query.split()) <= 15:
        if any(keyword in query_lower for keyword in medical_keywords + mental_health_keywords):
            # If medical or mental health, use specialized models even for short queries
            pass
        elif ModelType.SMALL in models:
            return models[ModelType.SMALL], ModelType.SMALL
    
    # Default to general model for conversational health topics if available
    if ModelType.GENERAL in models:
        return models[ModelType.GENERAL], ModelType.GENERAL
    
    # Ultimate fallback - return the first available model 
    if models:
        first_model_type = list(models.keys())[0]
        return models[first_model_type], first_model_type
    
    # If no models at all, raise an informative error
    raise RuntimeError("No language models are available. Check model paths and initialization.")

def fix_response(sanitized_response):
    """Apply post-processing fixes to the response"""
    if not sanitized_response:
        return "I'm sorry, I couldn't generate a response. Please try asking your question again."
    
    # Check if response seems complete
    has_proper_ending = any(sanitized_response.rstrip().endswith(p) for p in ['.', '!', '?', ':', ';'])
    
    # Check if the last sentence seems cut off (ends with certain words that suggest incompleteness)
    incomplete_endings = ['and', 'but', 'or', 'so', 'because', 'however', 'as', 'if', 'the', 'a', 'an', 'to', 'for', 'with', 'that', 'this', 'these', 'those', 'when', 'where', 'which', 'who', 'whom', 'whose', 'how', 'why', 'what', 'whatever', 'whichever', 'whoever', 'whomever', 'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would']
    words = sanitized_response.rstrip('.').split()
    last_word = words[-1].lower() if words else ""
    
    # Check if response appears to be cut off mid-paragraph
    ends_abruptly = not has_proper_ending or last_word in incomplete_endings
    likely_mid_thought = len(words) > 5 and len(words[-1]) <= 3
    
    # Check for incomplete or awkward phrases like "I also recommend you to." and fix them
    awkward_phrases = [
        r'I also recommend you to\.\s*',
        r'I suggest you to\.\s*',
        r'I advise you to\.\s*',
        r'You should\.\s*',
        r'It is recommended to\.\s*',
        r'It is suggested to\.\s*'
    ]
    
    for phrase in awkward_phrases:
        sanitized_response = re.sub(phrase, '', sanitized_response, flags=re.IGNORECASE)
    
    # Remove formal and cold openings
    formal_openings = [
        r'^(Hello|Hi|Greetings|Hey),?\s+I\s+am\s+SNEHA\.?\s*',
        r'^Thank you for your query\.?\s*',
        r'^Thank you for your question\.?\s*',
        r'^Thanks for reaching out\.?\s*',
        r'^Thanks for your message\.?\s*',
        r'^Thank you for contacting\.?\s*'
    ]
    for opening in formal_openings:
        sanitized_response = re.sub(opening, '', sanitized_response, flags=re.IGNORECASE)
    
    # Fix incomplete sentences with missing objects
    incomplete_sentences = [
        r'Take plenty\.', 
        r'Drink plenty\.', 
        r'Get plenty\.',
        r'Use plenty\.',
        r'Apply plenty\.'
    ]
    for phrase in incomplete_sentences:
        if re.search(phrase, sanitized_response):
            if 'water' in sanitized_response.lower() or 'fluid' in sanitized_response.lower() or 'hydrat' in sanitized_response.lower():
                sanitized_response = re.sub(phrase, 'Stay well hydrated.', sanitized_response)
            else:
                sanitized_response = re.sub(phrase, 'Make sure to stay hydrated and get plenty of rest.', sanitized_response)
                
    # Fix incomplete professional consultation suggestions
    consult_patterns = [
        (r'better to consult an\.', 'better to consult a healthcare provider.'),
        (r'consult an\.', 'consult a healthcare provider.'),
        (r'consult a\.', 'consult a healthcare provider.'),
        (r'talk to a\.', 'talk to a healthcare provider.'),
        (r'see a\.', 'see a healthcare provider.'),
        (r'visit a\.', 'visit a healthcare provider.'),
        (r'check with an\.', 'check with a healthcare provider.'),
        (r'check with a\.', 'check with a healthcare provider.'),
    ]
    
    for pattern, replacement in consult_patterns:
        sanitized_response = re.sub(pattern, replacement, sanitized_response, flags=re.IGNORECASE)
        
    # Add specific advice for common conditions when missing
    if ('sneez' in sanitized_response.lower() and 'nasal' in sanitized_response.lower()) or 'nose' in sanitized_response.lower():
        if not any(tip in sanitized_response.lower() for tip in ['saline', 'steam', 'humidifier', 'warm', 'rest', 'antihistamine']):
            # Find a good point to insert advice
            if '. ' in sanitized_response and not sanitized_response.endswith('. '):
                # Insert after a sentence that's not the end
                parts = sanitized_response.split('. ')
                if len(parts) >= 2:
                    sanitized_response = '. '.join(parts[:-1]) + '. You might find relief with saline nasal sprays, breathing in steam from a warm shower, or using a humidifier. ' + parts[-1]
    
    # Fix mixed tense issues and unclear medication advice
    mixed_tense_patterns = [
        (r'You must have taken the medicine.+and also take it now', 'If you have medication that helps with these symptoms, you might want to take it as directed'),
        (r'You should have taken.+and also take', 'Consider taking'),
        (r'You must take.+you have already taken', 'If you haven\'t already taken medication for this, you might consider taking')
    ]
    
    for pattern, replacement in mixed_tense_patterns:
        sanitized_response = re.sub(pattern, replacement, sanitized_response, flags=re.IGNORECASE)
    
    # Fix incomplete responses
    if ends_abruptly or likely_mid_thought:
        # Check for potentially serious symptoms (headache + vomiting, chest pain, etc.)
        potentially_serious = False
        headache_vomiting = False
        
        # These combinations might indicate more serious conditions
        serious_combos = [
            ['headache', 'vomit'],
            ['chest', 'pain'],
            ['breath', 'difficult'],
            ['fever', 'rash'],
            ['dizz', 'faint'],
            ['sudden', 'vision'],
            ['severe', 'pain']
        ]
        
        for combo in serious_combos:
            if all(word in sanitized_response.lower() for word in combo):
                potentially_serious = True
                if combo == ['headache', 'vomit']:
                    headache_vomiting = True
                break
        
        # Special case for headache with vomiting - use the preferred example style
        if headache_vomiting:
            # Replace the entire response with a better structured one
            sanitized_response = "I'm really sorry you're feeling this way. Headaches with vomiting can be quite uncomfortable. Make sure to stay hydrated, especially if you've been vomiting. Rest in a quiet, dimly lit room since bright lights and noise can make headaches worse. If you haven't already and it's appropriate for you, you might consider taking an over-the-counter pain reliever. If your symptoms continue or get worse, it would be best to check with a healthcare provider to be safe. Is this the first time you've experienced these symptoms together, or has this happened before?"
            return sanitized_response
        
        # If response relates to sleep or tiredness
        if any(word in sanitized_response.lower() for word in ['sleep', 'tired', 'rest', 'insomnia', 'bed', 'night']) and not potentially_serious:
            # First clean up any poor transitions or repetitions in the existing response
            sanitized_response = sanitized_response.rstrip(" .")
            
            # Check if the response already covers common advice
            has_sleep_routine = any(phrase in sanitized_response.lower() for phrase in ["schedule", "routine", "regular"])
            has_screen_advice = any(phrase in sanitized_response.lower() for phrase in ["screen", "phone", "device", "blue light"])
            has_environment = any(phrase in sanitized_response.lower() for phrase in ["environment", "comfortable", "cozy", "dark", "quiet"])
            
            # Build a natural completion that doesn't repeat advice
            completion = ". "
            if not has_sleep_routine:
                completion += "It might help to stick to a regular sleep schedule, even on weekends. "
            if not has_screen_advice:
                completion += "Try to avoid screens at least an hour before bed as the blue light can disrupt your sleep. "
            if not has_environment:
                completion += "Making your bedroom comfortable, quiet, and slightly cool can also improve sleep quality. "
                
            completion += "I hope you're able to get some restful sleep soon. Would any of these suggestions work for you, or is there something specific keeping you up at night?"
            
            sanitized_response = sanitized_response + completion
        
        # If potentially serious symptoms are mentioned
        elif potentially_serious:
            sanitized_response = sanitized_response.rstrip(" .") + ". These symptoms can have various causes. It's important to rest, stay hydrated, and monitor how you feel. If your symptoms persist for more than a day, get worse, or if you're concerned, please don't hesitate to contact a healthcare provider. Your health and safety come first. How are you feeling right now? Is there anything specific I can help you understand about managing these symptoms?"            # If response relates to pain or discomfort
        elif any(word in sanitized_response.lower() for word in ['pain', 'ache', 'hurt', 'discomfort', 'sore']):
            sanitized_response = sanitized_response.rstrip(" .") + ". Gentle stretching, proper body positioning, and applying warmth might provide some relief. If the discomfort persists for more than a few days, it might be worth checking in with a healthcare provider. How are you feeling right now?"
            
            # If response relates to loneliness, isolation, or mental health concerns
        elif any(word in sanitized_response.lower() for word in ['lonely', 'alone', 'isolated', 'disconnected', 'loneliness', 'depression', 'anxiety', 'sad']):
            sanitized_response = sanitized_response.rstrip(" .") + ". Some things that might help include: reaching out to a friend or family member for a brief chat, joining community groups with shared interests, or practicing self-care activities that boost your mood. Small steps like a daily walk or keeping a gratitude journal can also help improve your emotional wellbeing. Would you like to talk more about specific strategies that might work for your situation?"
            
            # Generic completion for other topics
        else:
            sanitized_response = sanitized_response.rstrip(" .") + ". I hope this information is helpful. Is there anything specific you'd like me to explain further or any other questions you have?"
    # Add a period if the response doesn't end with a sentence-ending punctuation
    elif not has_proper_ending:
        sanitized_response = sanitized_response.rstrip() + "."

    return sanitized_response

def process_query(query, models):
    """Process user query using SNEHA's multi-model approach."""
    if not query.strip():
        return "Please enter a health-related question.", None
    
    # Handle basic questions without using the model 
    # (for faster responses to common queries)
    
    # Basic identity questions
    name_questions = ["what is your name", "what's your name", "who are you", "your name", "tell me your name", "name?"]
    if any(question in query.lower() for question in name_questions):
        return f"Hi!👋 My name is SNEHA (version {SNEHA_VERSION}). I'm a healthcare assistant developed by Saagnik Mondal from India. I'm here to help you with health-related questions in a caring and supportive way. How can I help you today?", ModelType.GENERAL
    
    # Basic capabilities questions
    capability_questions = ["what can you do", "how can you help", "what do you do", "your capabilities", "what can u do"]
    if any(question in query.lower() for question in capability_questions):
        return f"Hi! As SNEHA {SNEHA_VERSION}, I can help answer your health-related questions, provide information about symptoms, discuss wellness tips, and offer supportive guidance for health concerns. I use multiple specialized AI models to provide the most helpful information for your specific questions. I don't diagnose medical conditions, but I can provide general health information and self-care tips. What would you like to know about today?", ModelType.GENERAL
    
    # Simple greetings
    greetings = ["hi", "hello", "hey", "greetings", "hi there", "good morning", "good afternoon", "good evening"]
    if query.lower().strip() in greetings or len(query.split()) <= 2:
        return f"Hello! I'm SNEHA {SNEHA_VERSION}, your healthcare assistant. I'm here to help with health-related questions and concerns. How can I assist you with your health today?", ModelType.SMALL
    
    # Simple non-health topic detection
    non_health_keywords = ["movie", "weather", "politics", "sports", "game", "music", "cook", "recipe"]
    if any(keyword in query.lower() for keyword in non_health_keywords):
        return f"I'm SNEHA {SNEHA_VERSION}, specialized in healthcare topics. I don't have expertise in {query.lower()}. How can I help you with a health-related question instead?", ModelType.GENERAL

    # Select the appropriate model based on query content
    selected_model, model_type = select_model_for_query(query, models)
    
    # Use the appropriate prompt template based on model type
    if model_type == ModelType.RESEARCH:
        # For biomedical research queries using BioGPT
        try:
            inputs = selected_model["tokenizer"](f"Answer this biomedical question: {query}", return_tensors="pt")
            with suppress_stdout_stderr():
                with torch.no_grad():
                    outputs = selected_model["model"].generate(
                        inputs["input_ids"],
                        max_length=512,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        early_stopping=True
                    )
            response_text = selected_model["tokenizer"].decode(outputs[0], skip_special_tokens=True)
            sanitized_response = sanitize_response(response_text)
            return fix_response(sanitized_response), model_type
        except Exception as e:
            # Fallback to medical model if BioGPT fails
            st.error(f"Error with BioGPT: {str(e)}. Falling back to medical model.")
            selected_model = models[ModelType.MEDICAL]
            model_type = ModelType.MEDICAL
    
    # For GGUF models (General, Medical, Support, Small)
    prompt_templates = {
        ModelType.GENERAL: f"""<|system|>
You are SNEHA, a friendly health coach and conversational healthcare assistant. Your specialty is daily wellness, healthy habits, and lifestyle guidance.

APPROACH:
- Be warm, friendly, and conversational - like chatting with a supportive friend
- Focus on practical, everyday health advice that's easy to implement
- Encourage small, sustainable habit changes rather than dramatic lifestyle shifts
- Be motivational without being pushy
- Ask follow-up questions about daily habits, exercise routines, or health goals
- Emphasize the connection between physical health, mental wellbeing, and daily habits
- Provide casual, realistic advice about nutrition, hydration, sleep, and exercise
- Keep technical terms to a minimum and use relatable examples
- End conversations with a simple encouragement or actionable suggestion

IMPORTANT GUIDELINES:
- Never claim medical expertise or make diagnostic statements
- For any concerning health issues, suggest consulting a healthcare provider
- Always acknowledge that lifestyle changes complement but don't replace medical care

SPECIFIC FOCUS AREAS:
- Daily habit formation and tracking
- Hydration, sleep hygiene, and balanced nutrition
- Basic fitness guidance and staying active
- Stress management through lifestyle choices
- Creating sustainable wellness routines
<|user|>
{query}
<|assistant|>
""",
        ModelType.MEDICAL: f"""<Instructions>
As SNEHA, you are a clinical-style medical assistant with expertise in health conditions, symptoms, treatments, and medical knowledge. You provide structured, reliable health information.

APPROACH FOR THIS QUERY: "{query}"

1. ASSESSMENT:
- Interpret the user's health question carefully
- Identify any described symptoms, conditions, or health concerns
- Determine if there are potential serious medical implications

2. RESPONSE STRUCTURE:
- Begin with a brief empathetic acknowledgment
- Provide clear, factual information about the condition or symptoms described
- Explain common causes, contributing factors, or relevant mechanisms
- Suggest 2-3 evidence-based self-care approaches when appropriate
- Include specific guidance on when professional medical care is needed

3. TONE AND STYLE:
- Use a reassuring but authoritative tone
- Balance medical accuracy with accessibility
- Include specific details that show medical knowledge
- Keep responses organized and focused on the medical question
- Blend clinical precision with patient-friendly language

4. SAFETY PROTOCOLS:
- For potentially serious symptoms (chest pain, severe headache, breathing difficulty, etc.), emphasize the importance of medical evaluation
- Never discourage seeking professional medical care
- Be clear about the limitations of self-diagnosis
- When recommending remedies, provide specific, detailed instructions

5. WRAP-UP:
- End with a relevant follow-up question about their specific situation
- Demonstrate continuity of care by referencing their particular concern

Remember: Maintain the balance between being informative and acknowledging your role as a supportive healthcare assistant, not a doctor.
</Instructions>

Response:""",
        ModelType.SUPPORT: f"""<|im_start|>system
You are SNEHA, a mental health support assistant providing empathetic, compassionate responses to people experiencing emotional distress or mental health concerns. 

PRIMARY GOAL: Create a safe, supportive space for the person to feel heard and validated while guiding them toward appropriate resources.

CRISIS DETECTION AND RESPONSE:
- Recognize signs of acute distress, suicidal ideation, or self-harm
- For severe crisis situations, gently but clearly emphasize the importance of immediate professional help
- Always provide crisis resources: suggest calling a mental health helpline, texting a crisis service, or contacting a trusted person
- Never minimize expressions of suicidal thoughts or self-harm intentions

COMMUNICATION APPROACH:
- Lead with validation of feelings (e.g., "What you're feeling is understandable")
- Use reflective listening techniques to show you understand their concerns
- Maintain a warm, non-judgmental tone that conveys genuine care
- Ask clarifying questions to better understand their emotional state
- Balance empathy with appropriate professional boundaries

APPROPRIATE SUPPORT STRATEGIES:
- Suggest simple grounding and mindfulness techniques for immediate distress
- Frame mental health challenges as common, normal human experiences
- Provide psychoeducation about common mental health concepts when relevant
- Emphasize that seeking help is a sign of strength, not weakness
- Suggest accessible self-care strategies while acknowledging their limitations

PROFESSIONAL BOUNDARIES:
- Clearly acknowledge you are not a licensed therapist or counselor
- Encourage professional mental health support when appropriate
- Never attempt to diagnose specific mental health conditions
- Avoid overly simplistic "quick fixes" for complex mental health issues

For this interaction, focus on creating a supportive, hopeful conversation while guiding toward appropriate resources if needed.
<|im_end|>

<|im_start|>user
{query}
<|im_end|>

<|im_start|>assistant
""",
        ModelType.SMALL: f"""<|system|>
You are SNEHA, a quick-response healthcare assistant designed for rapid symptom assessment and triage. Your role is to provide concise, targeted responses that help users decide their next steps.

YOUR APPROACH:
- Provide very brief, focused answers (2-5 sentences maximum)
- Start with a direct answer to the main question
- Use simple, clear language with no technical jargon
- Be decisive and specific rather than general
- For urgent symptoms, clearly state "This needs medical attention"
- For non-urgent issues, provide 1-2 specific self-care suggestions
- Use bullet points when appropriate for clarity and speed

QUESTION TYPES TO HANDLE:
- Yes/no health questions
- Quick symptom assessment
- "Should I see a doctor?" queries
- Basic first aid guidance
- Simple medication questions
- General health facts

Your goal is to be the "quick triage" system that helps users determine if they need urgent care, routine care, or simple self-care.
<|user|>
{query}
<|assistant|>
"""
    }
    
    # Get appropriate prompt for the selected model
    prompt = prompt_templates[model_type]
    
    # Special handling for BioGPT research model
    if model_type == ModelType.RESEARCH:
        research_prompt = f"""Answer this biomedical or healthcare research question based on scientific literature. 
Focus on providing evidence-based information with proper context. Include relevant research findings when available.
If appropriate, mention study types, limitations of evidence, or scientific consensus.
For mechanisms and processes, explain in clear, structured terms.
Remember to cite general sources of information (like "According to clinical research..." or "Studies in the field suggest...").
Question: {query}
"""
        try:
            inputs = selected_model["tokenizer"](research_prompt, return_tensors="pt")
            with suppress_stdout_stderr():
                with torch.no_grad():
                    outputs = selected_model["model"].generate(
                        inputs["input_ids"],
                        max_length=512,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        early_stopping=True
                    )
            response_text = selected_model["tokenizer"].decode(outputs[0], skip_special_tokens=True)
            sanitized_response = sanitize_response(response_text)
            return fix_response(sanitized_response), model_type
        except Exception as e:
            st.error(f"Error with BioGPT: {str(e)}. Falling back to medical model.")
            selected_model = models[ModelType.MEDICAL]
            model_type = ModelType.MEDICAL
            prompt = prompt_templates[model_type]
    
    
    # Get appropriate prompt for the selected model
    prompt = prompt_templates[model_type]
    
    try:
        # Process with the selected model
        with suppress_stdout_stderr():
            response = selected_model(
                prompt,
                max_tokens=300,           # Token limit for complete responses
                temperature=0.5,          # Balanced between creativity and consistency
                echo=False,
                top_k=20,                 # Limited for faster processing
                top_p=0.9,                # Standard sampling for reliability
                repeat_penalty=1.1        # Light penalty to avoid repetition
            )
        
        # Extract response text
        response_text = response["choices"][0]["text"] if "choices" in response else ""
        sanitized_response = sanitize_response(response_text)
        
        # Apply post-processing fixes
        final_response = fix_response(sanitized_response)
        return final_response, model_type
    
    except Exception as e:
        # Fallback to the general model if there's an error
        st.error(f"Error with {model_type.value} model: {str(e)}. Falling back to general model.")
        try:
            general_model = models[ModelType.GENERAL]
            general_prompt = prompt_templates[ModelType.GENERAL]
            
            with suppress_stdout_stderr():
                response = general_model(
                    general_prompt,
                    max_tokens=300,
                    temperature=0.5,
                    echo=False,
                    top_k=20,
                    top_p=0.9,
                    repeat_penalty=1.1
                )
            
            response_text = response["choices"][0]["text"] if "choices" in response else ""
            sanitized_response = sanitize_response(response_text)
            return fix_response(sanitized_response), ModelType.GENERAL
        
        except:
            # Last resort fallback
            return "I'm sorry, I'm having trouble processing your question at the moment. Could you please try again or rephrase your question?", ModelType.GENERAL

def display_welcome_screen():
    """Display an animated welcome screen"""
    welcome_html = """
    <div id="welcome-screen" class="welcome-container">
        <div class="welcome-logo">
            <svg width="150" height="150" viewBox="0 0 512 512">
                <circle cx="256" cy="256" r="230" fill="#e8f5e9" stroke="#4caf50" stroke-width="12"/>
                <path d="M256,120 C190,120 130,180 130,260 C130,320 170,380 256,380 C342,380 382,320 382,260 C382,180 322,120 256,120 Z" fill="#4caf50" opacity="0.3"/>
                <path d="M200,240 L220,260 L260,220" stroke="#4caf50" stroke-width="15" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M200,300 L220,320 L260,280" stroke="#4caf50" stroke-width="15" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M310,240 C318,240 325,247 325,255 C325,263 318,270 310,270 C302,270 295,263 295,255 C295,247 302,240 310,240 Z" fill="#4caf50"/>
                <path d="M310,300 C318,300 325,307 325,315 C325,323 318,330 310,330 C302,330 295,323 295,315 C295,307 302,300 310,300 Z" fill="#4caf50"/>
            </svg>
        </div>
        <h1 class="welcome-title">SNEHA AI</h1>
        <h2 class="welcome-subtitle">Your Offline Healthcare Assistant</h2>
        <p class="welcome-author">Presented by Saagnik Mondal</p>
    </div>
    <div id="main-app" style="display:none;"></div>
    """
    return welcome_html

def display_animated_loading():
    """Display an animated loading indicator"""
    loading_html = """
    <div class="loading-container">
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
        <div class="loading-dot"></div>
    </div>
    """
    return loading_html

def display_sidebar():
    """Display the sidebar with animations"""
    st.sidebar.image("https://img.icons8.com/color/96/000000/caduceus.png", width=120)
    
    st.sidebar.markdown("""
    <div class="sidebar-content">
        <h2 style="color: #2e7d32; margin-bottom: 1rem;">About SNEHA {}</h2>
        <p style="margin-bottom: 1.5rem;">
            SNEHA is a healthcare intelligence system designed for fast, efficient responses using multiple specialized models.
            Developed by Saagnik Mondal from India, SNEHA provides caring health assistance with a warm, personalized approach.
        </p>
    </div>
    """.format(SNEHA_VERSION), unsafe_allow_html=True)
    
    st.sidebar.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div class="sidebar-content">
        <h3 style="color: #2e7d32; margin-bottom: 1rem;">SNEHA's Models</h3>
        
        <div class="sidebar-feature">
            <span class="feature-icon">🩺</span>
            <span><b>Medical</b>: Specialized clinical knowledge</span>
        </div>
        
        <div class="sidebar-feature">
            <span class="feature-icon">📚</span>
            <span><b>Research</b>: Biomedical research information</span>
        </div>
        
        <div class="sidebar-feature">
            <span class="feature-icon">🤗</span>
            <span><b>Support</b>: Empathetic mental health responses</span>
        </div>
        
        <div class="sidebar-feature">
            <span class="feature-icon">💬</span>
            <span><b>General</b>: Healthcare conversations</span>
        </div>
        
        <div class="sidebar-feature">
            <span class="feature-icon">⚡</span>
            <span><b>Small</b>: Fast responses for simple questions</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Version information
    st.sidebar.markdown(f"""
    <div style="margin-top: 2rem; padding: 1rem; background-color: #f5f7fa; border-radius: 10px; text-align: center;">
        <p style="color: #78909c; font-size: 0.9rem; margin-bottom: 0.2rem;">Version {SNEHA_VERSION}</p>
        <p style="color: #90a4ae; font-size: 0.8rem; margin-bottom: 0;">"{SNEHA_CODENAME}"</p>
    </div>
    """, unsafe_allow_html=True)
    
def get_model_badge(model_type):
    """Get badge HTML for the model type"""
    badges = {
        ModelType.GENERAL: "<span class='model-badge model-general'>Zephyr</span>",
        ModelType.MEDICAL: "<span class='model-badge model-medical'>Med-Alpaca</span>",
        ModelType.RESEARCH: "<span class='model-badge model-research'>BioGPT</span>",
        ModelType.SUPPORT: "<span class='model-badge model-support'>OpenHermes</span>",
        ModelType.SMALL: "<span class='model-badge model-small'>Phi-2</span>",
    }
    return badges.get(model_type, "")
    
def main():
    """Main function to run the SNEHA healthcare assistant app with modern UI"""
    
    # Session state to track if this is the first load
    if 'first_load' not in st.session_state:
        st.session_state.first_load = True
        st.session_state.show_welcome = True
        st.session_state.models_loaded = False
        st.session_state.chat_history = []
    
    # Display sidebar 
    display_sidebar()
    
    # Welcome screen or main app
    if st.session_state.show_welcome:
        # Display animated welcome screen
        st.markdown(display_welcome_screen(), unsafe_allow_html=True)
        
        if st.button("Start Consultation", key="welcome_button"):
            st.session_state.show_welcome = False
            st.rerun()
    else:
        # Main app content
        st.markdown(f"""
        <div class="app-header">
            <h1>SNEHA - Your Offline Healthcare Assistant</h1>
            <div class="sneha-version">Version {SNEHA_VERSION} "{SNEHA_CODENAME}"</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize models if not already loaded
        if not st.session_state.models_loaded:
            with st.spinner("Loading SNEHA's knowledge..."):
                try:
                    st.markdown(display_animated_loading(), unsafe_allow_html=True)
                    models = initialize_models()
                    st.session_state.models = models
                    st.session_state.models_loaded = True
                    time.sleep(1)  # Brief pause for animation effect
                except Exception as e:
                    st.error(f"Error initializing healthcare models: {str(e)}")
                    st.stop()
        else:
            models = st.session_state.models
        
        # User input area
        with st.container():
            st.write("##### Ask me about any health-related topics:")
            user_query = st.text_area(
                label="Health Question", 
                placeholder="For example: I've had a headache for two days, what should I do?", 
                height=100,
                label_visibility="collapsed"  # Hide the label visually but keep it for accessibility
            )
            
            col1, col2, col3 = st.columns([1, 1, 6])
            with col1:
                submit_button = st.button("Send", key="send_query", use_container_width=True)
            with col2:
                clear_button = st.button("Clear", key="clear_chat", use_container_width=True)
        
        # Process query when submit button is clicked
        if submit_button and user_query:
            with st.spinner(""):
                st.markdown(display_animated_loading(), unsafe_allow_html=True)
                response, model_type = process_query(user_query, models)
                
                # Add to chat history
                st.session_state.chat_history.append(("user", user_query))
                st.session_state.chat_history.append(("sneha", response, model_type))
        
        # Clear chat history
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
            
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### Conversation")
            for i, entry in enumerate(st.session_state.chat_history):
                if entry[0] == "user":
                    st.markdown(f"""
                    <div style="background-color: #f1f1f1; border-radius: 15px; padding: 10px 15px; margin: 5px 0 5px auto; max-width: 80%; text-align: right; float: right; clear: both;">
                        <p style="color: #333; margin: 0;">{entry[1]}</p>
                    </div>
                    <div style="clear: both;"></div>
                    """, unsafe_allow_html=True)
                else:  # sneha response
                    model_badge = get_model_badge(entry[2]) if len(entry) > 2 else ""
                    st.markdown(f"""
                    <div class="sneha-response">
                        <p style="color: #333; margin: 0;">{entry[1]}</p>
                        <div style="text-align: right; margin-top: 5px; opacity: 0.7;">{model_badge}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Add disclaimer at the bottom
        st.markdown(f"""<div class="disclaimer">{DISCLAIMER}</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
