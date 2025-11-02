"""
Pulmo.ai - AI-Powered Lung Cancer Screening & Health Advisory System
Advanced CT scan analysis with personalized health recommendations.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import tempfile

# Add submission folder to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

# Import CT-CLIP inference
from ct_description.ct_clip_inference import CTClipInferenceSingle
from ct_description.ensemble_inference import ensemble_inference
from ct_description.report_generator import generate_report, PATHOLOGY_SPECIFIC_THRESHOLDS
from ct_description.lung_cancer_report_generator import generate_lung_cancer_focused_report

# Import agent and utilities from health_recommendations
from health_recommendations.agent import create_agent, AgentState
from health_recommendations.utils import (
    validate_user_profile,
    create_patient_summary
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Pulmo.ai - AI Lung Cancer Screening",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Remove file upload size limit for large CT scans (500+ MB)
# Default Streamlit limit is 200 MB, we need to support larger volumes
try:
    # This must be set before any file_uploader is created
    from streamlit.web.server.server import Server
    Server._max_upload_size_mb = 2000  # 2 GB limit
except:
    # If the above doesn't work, try config.toml approach
    pass

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .urgency-immediate {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .urgency-urgent {
        background-color: #ff8800;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .urgency-moderate {
        background-color: #ffbb33;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .urgency-routine {
        background-color: #00C851;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .patient-summary {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
    }
    .findings-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 15px 0;
    }
    .model-info {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "patient_profile" not in st.session_state:
        st.session_state.patient_profile = {}
    
    if "ct_file_path" not in st.session_state:
        st.session_state.ct_file_path = None
    
    if "ct_predictions" not in st.session_state:
        st.session_state.ct_predictions = None
    
    if "ct_findings_text" not in st.session_state:
        st.session_state.ct_findings_text = None
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "ensemble"
    
    if "agent_state" not in st.session_state:
        st.session_state.agent_state = None
    
    if "agent" not in st.session_state:
        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("MODEL_NAME", "gpt-4o")
        st.session_state.agent = create_agent(
            model_name=model_name,
            temperature=0.7,
            api_key=api_key
        )
    
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = None
    
    if "urgency_level" not in st.session_state:
        st.session_state.urgency_level = None
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    if "profile_complete" not in st.session_state:
        st.session_state.profile_complete = False
    
    if "ct_complete" not in st.session_state:
        st.session_state.ct_complete = False


def render_sidebar():
    """Render the sidebar with user profile input."""
    with st.sidebar:
        st.markdown("## 👤 Patient Profile")
        
        with st.form("patient_profile_form"):
            st.markdown("### Basic Information")
            
            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=st.session_state.patient_profile.get("age", 50),
                help="Patient's age in years"
            )
            
            gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Other"],
                index=["Male", "Female", "Other"].index(
                    st.session_state.patient_profile.get("gender", "Male")
                )
            )
            
            st.markdown("### Lifestyle & Risk Factors")
            
            smoking_status = st.selectbox(
                "Smoking Status",
                ["Never Smoker", "Former Smoker", "Current Smoker", "Heavy Smoker (>20/day)"],
                index=0
            )
            
            family_history = st.text_area(
                "Family History",
                value=st.session_state.patient_profile.get("family_history", ""),
                placeholder="e.g., Father had lung cancer at age 60",
                help="Any relevant family medical history"
            )
            
            symptoms = st.text_area(
                "Current Symptoms",
                value=st.session_state.patient_profile.get("symptoms", ""),
                placeholder="e.g., Persistent cough, fatigue, chest pain",
                help="List any symptoms you're experiencing"
            )
            
            st.markdown("### Background Information")
            
            occupation = st.text_input(
                "Occupation/Environment",
                value=st.session_state.patient_profile.get("occupation", ""),
                placeholder="e.g., Construction worker, urban area"
            )
            
            location = st.text_input(
                "Location",
                value=st.session_state.patient_profile.get("location", ""),
                placeholder="e.g., Mumbai, Delhi NCR"
            )
            
            medical_history = st.text_area(
                "Medical History",
                value=st.session_state.patient_profile.get("medical_history", ""),
                placeholder="e.g., Hypertension, diabetes, asthma"
            )
            
            additional_context = st.text_area(
                "Additional Context",
                value=st.session_state.patient_profile.get("additional_context", ""),
                placeholder="Any other relevant information"
            )
            
            submitted = st.form_submit_button("💾 Save Profile", type="primary")
            
            if submitted:
                profile_data = {
                    "age": age,
                    "gender": gender,
                    "smoking_status": smoking_status,
                    "family_history": family_history,
                    "symptoms": symptoms,
                    "occupation": occupation,
                    "location": location,
                    "medical_history": medical_history,
                    "additional_context": additional_context
                }
                
                is_valid, error_msg = validate_user_profile(profile_data)
                
                if is_valid:
                    st.session_state.patient_profile = profile_data
                    st.session_state.profile_complete = True
                    st.success("✅ Profile saved successfully!")
                else:
                    st.error(f"❌ {error_msg}")
        
        # Display saved profile summary
        if st.session_state.profile_complete:
            st.markdown("---")
            st.markdown("### 📋 Profile Summary")
            summary_parts = []
            p = st.session_state.patient_profile
            summary_parts.append(f"**Age:** {p.get('age', 'N/A')}")
            summary_parts.append(f"**Gender:** {p.get('gender', 'N/A')}")
            summary_parts.append(f"**Smoking:** {p.get('smoking_status', 'N/A')}")
            if p.get('symptoms'):
                summary_parts.append(f"**Symptoms:** {p.get('symptoms')}")
            st.info("\n\n".join(summary_parts))


def get_model_paths():
    """Get available model paths."""
    models_dir = Path(__file__).parent / "models"
    
    model_paths = {
        "base": models_dir / "ct_clip_v2.pt",
        "vocabfine": models_dir / "ct_vocabfine_v2.pt"
    }
    
    return model_paths


def render_ct_upload_section():
    """Render CT scan upload and analysis section."""
    st.markdown("## 🔬 CT Scan Analysis")
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📤 Upload CT Scan")
    
    with col2:
        model_choice = st.selectbox(
            "AI Model",
            ["Ensemble (Recommended)", "VocabFine (Best AUROC 0.824)", "Base (More Sensitive)"],
            help="Choose the AI model for analysis"
        )
        
        # Map display name to internal key
        model_map = {
            "Ensemble (Recommended)": "ensemble",
            "VocabFine (Best AUROC 0.824)": "vocabfine",
            "Base (More Sensitive)": "base"
        }
        st.session_state.selected_model = model_map[model_choice]
    
    # Model info box
    model_info = {
        "ensemble": "🎯 **Ensemble**: Combines Base (40%) + VocabFine (60%) for robust predictions. Expected AUROC: ~0.87",
        "vocabfine": "🏆 **VocabFine**: Best overall performance. AUROC 0.824. More conservative, excellent for vascular findings.",
        "base": "🔍 **Base**: More sensitive detection. AUROC 0.772. Better for screening, may have more false positives."
    }
    
    st.markdown(f'<div class="model-info">{model_info[st.session_state.selected_model]}</div>', unsafe_allow_html=True)
    
    # File uploader (supports up to 2 GB files)
    uploaded_file = st.file_uploader(
        "Upload CT Volume (NIfTI format) - Supports files up to 2 GB",
        type=["nii", "gz", "nii.gz"],
        help="Upload your CT scan in NIfTI format (.nii or .nii.gz). Large files (500+ MB) are supported."
    )
    
    # Analyze button
    if uploaded_file is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button(" Analyze CT Scan", type="primary", use_container_width=True):
                analyze_ct_scan(uploaded_file)
    
    # Display results if available
    if st.session_state.ct_findings_text:
        st.markdown("---")
        st.markdown("### Clinical Analysis Results")
        
        st.markdown(f'<div class="findings-box">{st.session_state.ct_findings_text}</div>', unsafe_allow_html=True)
        
        # Show detailed predictions in expander
        if st.session_state.ct_predictions:
            with st.expander("🔍 View Detailed Predictions (18 Pathologies)"):
                predictions = st.session_state.ct_predictions
                
                # Sort by probability
                sorted_findings = sorted(
                    predictions.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Display as table
                st.markdown("| Pathology | Probability | Status |")
                st.markdown("|-----------|-------------|---------|")
                
                for pathology, prob in sorted_findings:
                    threshold = PATHOLOGY_SPECIFIC_THRESHOLDS.get(
                        pathology.lower().replace('_', ' '),
                        0.3
                    )
                    status = "✓ Detected" if prob >= threshold else "✗ Not detected"
                    st.markdown(f"| {pathology.replace('_', ' ').title()} | {prob:.3f} | {status} |")
        
        st.markdown("---")
        st.success("✅ CT scan analysis complete! Scroll down to generate personalized recommendations.")


def analyze_ct_scan(uploaded_file):
    """Analyze CT scan using selected model."""
    with st.spinner("⏳ Processing CT scan... This may take 10-20 seconds..."):
        try:
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            st.session_state.ct_file_path = tmp_path
            
            # Get model paths
            model_paths = get_model_paths()
            
            # Check if models exist
            if st.session_state.selected_model == "ensemble":
                if not model_paths["base"].exists() or not model_paths["vocabfine"].exists():
                    st.error("❌ Ensemble requires both Base and VocabFine models. Please download them first.")
                    st.info("See models/DOWNLOAD.md for download instructions")
                    return
            else:
                if not model_paths[st.session_state.selected_model].exists():
                    st.error(f"❌ {st.session_state.selected_model.title()} model not found.")
                    st.info("See models/DOWNLOAD.md for download instructions")
                    return
            
            # Run inference based on selected model
            if st.session_state.selected_model == "ensemble":
                # Ensemble inference
                result = ensemble_inference(
                    volume_path=tmp_path,
                    models={
                        'base': str(model_paths['base']),
                        'vocabfine': str(model_paths['vocabfine'])
                    },
                    weights={'base': 0.4, 'vocabfine': 0.6},
                    threshold=0.3,
                    device='cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu'
                )
                predictions = result['ensemble_predictions']
                model_name = "Ensemble (Base 40% + VocabFine 60%)"
                
            else:
                # Single model inference
                inferencer = CTClipInferenceSingle(
                    model_path=str(model_paths[st.session_state.selected_model]),
                    device='cuda' if os.system('nvidia-smi > /dev/null 2>&1') == 0 else 'cpu',
                    threshold=0.3
                )
                
                result = inferencer.infer_single_volume(
                    volume_path=tmp_path,
                    verbose=False
                )
                predictions = result['predictions']
                model_name = f"CT-CLIP {st.session_state.selected_model.title()}"
            
            # Store predictions
            st.session_state.ct_predictions = predictions
            
            # Generate lung cancer-focused report with explanations
            findings_text = generate_lung_cancer_focused_report(
                predictions=predictions,
                model_name=model_name,
                scan_info={
                    'filename': uploaded_file.name,
                    'analyzed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                include_custom_terms=True  # Include additional lung cancer terms
            )
            
            st.session_state.ct_findings_text = findings_text
            st.session_state.ct_complete = True
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error analyzing CT scan: {str(e)}")
            st.info("ℹ️ Make sure models are downloaded and the file is a valid NIfTI CT volume.")
            import traceback
            with st.expander("� Error Details"):
                st.code(traceback.format_exc())


def render_generate_recommendations():
    """Render the generate recommendations section."""
    st.markdown("#### 📋 Generate Personalized Health Plan")
    
    if not st.session_state.profile_complete:
        st.warning("⚠️ Please complete the patient profile in the sidebar first.")
        return
    
    if not st.session_state.ct_complete:
        st.warning("⚠️ Please upload and analyze a CT scan first.")
        return
    
    if st.button("Generate Personalized Health Plan", type="primary", use_container_width=True):
        with st.spinner("⏳ Analyzing patient data and creating personalized health plan..."):
            try:
                # Create CT results dict from AI findings
                ct_results = {
                    "findings": st.session_state.ct_findings_text,
                    "model_used": st.session_state.selected_model,
                    "scan_date": datetime.now().strftime("%Y-%m-%d"),
                    "predictions": st.session_state.ct_predictions
                }
                
                result = st.session_state.agent.generate_recommendations(
                    patient_profile=st.session_state.patient_profile,
                    ct_results=ct_results
                )
                
                st.session_state.recommendations = result["recommendations"]
                st.session_state.urgency_level = result["urgency_level"]
                
                # Initialize agent state for chat
                st.session_state.agent_state = {
                    "messages": result["messages"],
                    "patient_profile": st.session_state.patient_profile,
                    "ct_results": ct_results,
                    "current_recommendations": result["recommendations"],
                    "urgency_level": result["urgency_level"],
                    "conversation_context": {},
                    "next_action": None
                }
                
                st.success("✅ Personalized health plan generated!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
                st.info("ℹ️ Please check your OpenAI API key in the .env file")


def render_recommendations():
    """Render the recommendations display."""
    if st.session_state.recommendations:
        st.markdown("## 📊 Personalized Health Plan")
        
        # Display urgency level
        urgency = st.session_state.urgency_level
        urgency_class = f"urgency-{urgency.lower()}" if urgency else "urgency-routine"
        
        st.markdown(
            f'<div class="{urgency_class}">⚠️ URGENCY LEVEL: {urgency}</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # Display recommendations
        st.markdown(st.session_state.recommendations)
        
        # Download option
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_content = f"""
CT SCAN HEALTH REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: {st.session_state.selected_model.upper()}

{'='*80}
AI-DETECTED FINDINGS
{'='*80}

{st.session_state.ct_findings_text}

{'='*80}
PERSONALIZED HEALTH PLAN
{'='*80}

{st.session_state.recommendations}

{'='*80}
DISCLAIMER
{'='*80}
This report is generated by AI and should not replace professional medical advice.
Always consult with qualified healthcare providers for medical decisions.
"""
            
            st.download_button(
                label="📥 Download Complete Report",
                data=report_content,
                file_name=f"health_plan_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )


def render_chat_interface():
    """Render the chat interface for follow-up questions."""
    if not st.session_state.recommendations:
        return
    
    st.markdown("## 💬 Ask Follow-up Questions")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_messages):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant", avatar="🏥"):
                    st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about your health plan or CT findings...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Get response from agent
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.agent.chat(
                    user_message=user_input,
                    state=st.session_state.agent_state
                )
                
                # Update agent state
                st.session_state.agent_state = result
                
                # Extract AI response
                ai_response = ""
                for msg in reversed(result["messages"]):
                    if hasattr(msg, 'content') and msg.__class__.__name__ == "AIMessage":
                        ai_response = msg.content
                        break
                
                # Add AI response to chat
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": ai_response
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🫁 Pulmo.ai</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Lung Cancer Screening & Personalized Health Advisory</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    render_ct_upload_section()
    
    st.markdown("---")
    
    render_generate_recommendations()
    
    st.markdown("---")
    
    render_recommendations()
    
    st.markdown("---")
    
    render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; padding: 20px;'>
            <p><strong>⚠️ Medical Disclaimer:</strong> This tool provides AI-generated analysis and recommendations 
            and should not replace professional medical advice. Always consult with qualified healthcare 
            providers for medical decisions.</p>
            <p style='margin-top: 10px;'>Powered by CT-CLIP, LangGraph, LangChain, and OpenAI</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
