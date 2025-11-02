# Stage 4: Health Recommendations Pipeline (LangGraph + GPT-4)

## Overview

This stage generates personalized health recommendations using LangGraph and GPT-4 based on CT scan findings and patient profile.

## Architecture

### LangGraph Workflow

```
┌─────────────────────────────────────────────────┐
│                   START                          │
└────────────────┬────────────────────────────────┘
                 │
         ┌───────▼────────┐
         │  Analyze        │
         │  Patient        │
         │  Profile        │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Analyze        │
         │  CT Findings    │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Assess         │
         │  Risk Factors   │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Generate       │
         │  Recommendations│
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │  Classify       │
         │  Urgency        │
         └───────┬────────┘
                 │
         ┌───────▼────────┐
         │      END         │
         │   (Return Recs)  │
         └──────────────────┘
```

### State Management

```python
from typing import TypedDict, List, Dict
from langchain_core.messages import BaseMessage

class HealthAdvisorState(TypedDict):
    """LangGraph state for health recommendations"""
    
    # Input
    patient_profile: Dict[str, any]
    ct_findings: Dict[str, float]
    nodule_classification: Dict[str, any]  # From Stage 2
    
    # Intermediate
    risk_factors: List[str]
    urgency_level: str
    
    # Output
    recommendations: str
    follow_up_schedule: Dict[str, str]
    conversation_history: List[BaseMessage]
```

## Components

### 1. Agent (`agent.py`)

Main LangGraph agent with 5 nodes:

```python
from langgraph.graph import StateGraph

# Build graph
workflow = StateGraph(HealthAdvisorState)

# Add nodes
workflow.add_node("analyze_profile", analyze_patient_profile)
workflow.add_node("analyze_ct", analyze_ct_findings)
workflow.add_node("assess_risk", assess_risk_factors)
workflow.add_node("generate_recs", generate_recommendations)
workflow.add_node("classify_urgency", classify_urgency)

# Add edges
workflow.add_edge("analyze_profile", "analyze_ct")
workflow.add_edge("analyze_ct", "assess_risk")
workflow.add_edge("assess_risk", "generate_recs")
workflow.add_edge("generate_recs", "classify_urgency")

# Compile
app = workflow.compile()
```

### 2. Prompts (`prompts.py`)

Specialized prompts for each agent:

```python
PROFILE_ANALYSIS_PROMPT = """
You are a medical AI assistant analyzing patient profiles.

Patient Information:
- Age: {age}
- Gender: {gender}
- Smoking History: {smoking_history}
- Occupation: {occupation}
- Medical History: {medical_history}

Extract key risk factors for lung cancer.
"""

CT_ANALYSIS_PROMPT = """
You are analyzing CT scan findings for lung cancer screening.

CT Findings:
{ct_findings}

Nodule Classification (if present):
{nodule_classification}

Interpret the findings and their clinical significance.
"""

RECOMMENDATION_PROMPT = """
You are a medical AI generating personalized health recommendations.

Patient Profile:
{patient_profile}

CT Analysis:
{ct_analysis}

Risk Factors:
{risk_factors}

Generate comprehensive recommendations including:
1. Immediate actions
2. Follow-up schedule
3. Lifestyle modifications
4. Screening plan
5. India-specific resources

Be empathetic, clear, and actionable.
"""
```

### 3. Configuration (`config.py`)

LangChain and LLM configuration:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# LLM Configuration
def get_llm(temperature=0.3):
    """Get configured GPT-4 instance"""
    return ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=2000,
        request_timeout=60
    )

# Prompts
def get_prompt_template(template_name):
    """Get prompt template by name"""
    from prompts import PROMPTS
    return ChatPromptTemplate.from_template(
        PROMPTS[template_name]
    )
```

### 4. Utilities (`utils.py`)

Helper functions:

```python
def calculate_lung_cancer_risk_score(
    patient_profile: dict,
    ct_findings: dict
) -> float:
    """
    Calculate lung cancer risk score (0-100)
    
    Factors:
    - Age (>50: +20 points)
    - Smoking history (+30 points)
    - Nodule detected (+40 points)
    - Emphysema (+10 points)
    - Family history (+10 points)
    """
    score = 0
    
    # Age
    if patient_profile.get('age', 0) > 50:
        score += 20
    if patient_profile.get('age', 0) > 65:
        score += 10
    
    # Smoking
    if 'smoker' in patient_profile.get('smoking_history', '').lower():
        score += 30
    
    # CT findings
    if ct_findings.get('Lung nodule', 0) > 0.25:
        score += 40
    if ct_findings.get('Mass', 0) > 0.30:
        score += 50
    if ct_findings.get('Emphysema', 0) > 0.40:
        score += 10
    
    return min(score, 100)


def classify_urgency(risk_score: float) -> str:
    """Classify urgency based on risk score"""
    if risk_score >= 80:
        return "CRITICAL"
    elif risk_score >= 60:
        return "HIGH"
    elif risk_score >= 40:
        return "MODERATE"
    else:
        return "LOW"


def format_recommendations(
    recommendations: str,
    urgency: str,
    follow_up: dict
) -> str:
    """Format final recommendations for display"""
    
    urgency_colors = {
        'CRITICAL': '🔴',
        'HIGH': '🟠',
        'MODERATE': '🟡',
        'LOW': '🟢'
    }
    
    output = f"""
═══════════════════════════════════════════════════════
       PERSONALIZED HEALTH RECOMMENDATIONS
═══════════════════════════════════════════════════════

URGENCY LEVEL: {urgency_colors[urgency]} {urgency}

{recommendations}

───────────────────────────────────────────────────────
FOLLOW-UP SCHEDULE
───────────────────────────────────────────────────────

{format_follow_up(follow_up)}

═══════════════════════════════════════════════════════
"""
    return output
```

## Usage

### 1. Python API

```python
from health_recommendations.agent import generate_recommendations

# Patient profile
patient_profile = {
    "age": 62,
    "gender": "Male",
    "smoking_history": "Former smoker, quit 5 years ago (30 pack-years)",
    "occupation": "Construction worker",
    "environmental_exposure": "Asbestos (10 years)",
    "medical_history": "COPD, hypertension",
    "family_history": "Father had lung cancer at age 68",
    "symptoms": "Persistent cough for 3 months"
}

# CT findings from Stage 3
ct_findings = {
    "Lung nodule": 0.68,
    "Emphysema": 0.52,
    "Lymphadenopathy": 0.45,
    "Cardiomegaly": 0.67
}

# Nodule classification from Stage 2 (if applicable)
nodule_classification = {
    "probability_cancer": 0.76,
    "nodule_size_mm": 12,
    "nodule_location": "Right upper lobe"
}

# Generate recommendations
result = generate_recommendations(
    patient_profile=patient_profile,
    ct_findings=ct_findings,
    nodule_classification=nodule_classification
)

print(result['recommendations'])
print(f"\nUrgency: {result['urgency_level']}")
print(f"Risk Score: {result['risk_score']}/100")
```

### 2. Interactive Chat

```python
from health_recommendations.agent import HealthAdvisorAgent

# Initialize agent
agent = HealthAdvisorAgent()

# Set initial context
agent.set_context(
    patient_profile=patient_profile,
    ct_findings=ct_findings
)

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    
    response = agent.chat(user_input)
    print(f"Agent: {response}")
```

### 3. Command Line

```bash
python agent.py \
    --profile patient_profile.json \
    --ct_findings ct_results.json \
    --output recommendations.txt
```

## Example Output

```
═══════════════════════════════════════════════════════
       PERSONALIZED HEALTH RECOMMENDATIONS
═══════════════════════════════════════════════════════

URGENCY LEVEL: 🟠 HIGH

Patient: 62-year-old male, former smoker
Risk Score: 78/100

Based on your CT scan findings and medical history, here are 
my recommendations:

───────────────────────────────────────────────────────
🔴 IMMEDIATE ACTIONS (Within 48-72 hours)
───────────────────────────────────────────────────────

1. **Consult a Pulmonologist**
   - The detected lung nodule (12mm) with 76% cancer probability 
     requires urgent evaluation
   - Bring all CT images and reports
   
2. **Schedule PET-CT Scan**
   - To assess metabolic activity of the nodule
   - Helps distinguish cancerous from benign nodules
   
3. **Consider Biopsy**
   - Your pulmonologist may recommend CT-guided biopsy or 
     bronchoscopy to confirm diagnosis

───────────────────────────────────────────────────────
🟡 FOLLOW-UP & MONITORING
───────────────────────────────────────────────────────

1. **Short-term (3 months)**
   - Repeat chest CT scan to monitor nodule growth
   - Track any symptom changes
   
2. **Medium-term (6-12 months)**
   - Continue regular CT screening even if biopsy is benign
   - Monitor for new nodules or growth
   
3. **Long-term (Annual)**
   - Continue annual low-dose CT screening
   - Your smoking history and occupational exposure put you 
     at high risk

───────────────────────────────────────────────────────
💊 LIFESTYLE MODIFICATIONS
───────────────────────────────────────────────────────

1. **Maintain Smoke-Free Status**
   - Excellent that you quit 5 years ago!
   - Avoid secondhand smoke exposure
   
2. **Manage COPD**
   - Continue prescribed medications
   - Pulmonary rehabilitation if not already enrolled
   
3. **Nutrition**
   - Antioxidant-rich diet (fruits, vegetables)
   - Maintain healthy weight
   
4. **Exercise**
   - Light to moderate activity (walking, yoga)
   - Improves lung function and overall health

───────────────────────────────────────────────────────
🏥 INDIA-SPECIFIC RESOURCES
───────────────────────────────────────────────────────

**Pulmonology Centers in India:**
- Tata Memorial Hospital, Mumbai
- AIIMS, New Delhi
- Apollo Hospitals (multiple locations)
- Fortis Memorial Research Institute, Gurgaon

**Financial Assistance:**
- PM-JAY (Ayushman Bharat) - Check eligibility
- State government schemes
- Hospital charity programs

**Support Groups:**
- Indian Cancer Society
- CanSupport (Delhi NCR)
- Online forums: CancerConnect India

───────────────────────────────────────────────────────
📋 QUESTIONS TO ASK YOUR DOCTOR
───────────────────────────────────────────────────────

1. What is the exact size and location of the nodule?
2. What are the chances this is lung cancer vs. benign?
3. What are my biopsy options and their risks?
4. If it's cancer, what stage and treatment options?
5. How will my COPD affect treatment?
6. Should my family members get screened?

───────────────────────────────────────────────────────
⚠️ WARNING SIGNS - Seek Emergency Care If:
───────────────────────────────────────────────────────

- Coughing up blood (hemoptysis)
- Severe chest pain
- Sudden shortness of breath
- Unexplained weight loss (>5kg in a month)
- High fever with chills

───────────────────────────────────────────────────────
FOLLOW-UP SCHEDULE
───────────────────────────────────────────────────────

📅 Week 1: Pulmonologist consultation
📅 Week 2: PET-CT scan
📅 Month 1: Biopsy (if recommended)
📅 Month 3: Follow-up CT scan
📅 Month 6: Pulmonologist review
📅 Year 1: Annual screening CT

═══════════════════════════════════════════════════════

💙 Remember: Early detection saves lives. Your proactive 
approach to getting screened is commendable.

Stay positive and follow through with the recommended actions.

═══════════════════════════════════════════════════════
```

## Urgency Classification

| Urgency | Risk Score | Criteria | Action Timeline |
|---------|-----------|----------|----------------|
| CRITICAL | 80-100 | High cancer probability, large nodules | 24-48 hours |
| HIGH | 60-79 | Suspicious nodules, multiple risk factors | 48-72 hours |
| MODERATE | 40-59 | Small nodules, some risk factors | 1-2 weeks |
| LOW | 0-39 | No nodules, minimal risk factors | Routine follow-up |

## Customization

### Adjust Urgency Thresholds

Edit `utils.py`:

```python
def classify_urgency(risk_score: float) -> str:
    if risk_score >= 85:  # More conservative
        return "CRITICAL"
    # ...
```

### Add India-Specific Resources

Edit `prompts.py`:

```python
INDIA_RESOURCES = """
Add your regional hospitals, clinics, or support groups:
- Local hospital name
- Regional cancer center
- State-specific schemes
"""
```

### Customize LLM Temperature

Edit `config.py`:

```python
# Higher temperature = more creative
llm = get_llm(temperature=0.5)

# Lower temperature = more conservative
llm = get_llm(temperature=0.1)
```

## Dependencies

```bash
pip install -r ../requirements.txt
```

Key packages:
- `langchain>=0.3.0`
- `langgraph>=0.2.0`
- `langchain-openai>=0.2.0`
- `python-dotenv>=1.0.0`

## Environment Variables

Create `.env` file:

```bash
OPENAI_API_KEY=sk-...
LANGCHAIN_TRACING_V2=true  # Optional: for debugging
LANGCHAIN_API_KEY=...      # Optional: for LangSmith
```

## Performance

- **Response Time**: 3-5 seconds
- **Token Usage**: ~1,500-2,500 tokens per request
- **Cost**: ~$0.03-0.05 per recommendation (GPT-4)

## Troubleshooting

**Issue**: OpenAI API rate limit
```python
# Add retry logic
from langchain.llms.openai import OpenAI
llm = OpenAI(max_retries=3, request_timeout=60)
```

**Issue**: Slow response
```python
# Use GPT-3.5 instead
llm = ChatOpenAI(model="gpt-3.5-turbo")  # Faster, cheaper
```

**Issue**: Context too long
```python
# Reduce max_tokens
llm = ChatOpenAI(max_tokens=1000)
```

## References

1. LangGraph documentation: https://langchain-ai.github.io/langgraph/
2. LangChain prompt engineering: https://python.langchain.com/docs/modules/prompts/
3. Clinical guidelines: Fleischner Society, NCCN

---

**Empowering patients with AI-driven insights! 💙**
