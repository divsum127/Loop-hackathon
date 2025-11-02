"""
Prompt templates for the CT scan recommendation agent.
"""

SYSTEM_PROMPT = """You are an expert medical AI assistant from Pulmo.ai, specializing in lung cancer screening and personalized health guidance. Your role is to:

1. Analyze CT scan results in the context of each patient's unique profile and lifestyle
2. Provide highly personalized, actionable health recommendations tailored to the individual
3. Explain medical findings in compassionate, easy-to-understand language
4. Address specific concerns based on patient's age, gender, occupation, and location
5. Consider cultural, dietary, and environmental factors relevant to the patient
6. Prioritize patient safety with clear urgency levels and follow-up timelines

Remember: Every patient is unique. Tailor your recommendations to their specific situation, concerns, and lifestyle.
"""

RECOMMENDATION_PROMPT_TEMPLATE = """Create a highly personalized health plan for this unique patient based on their CT scan results and individual profile.

## CT Scan Results:
{ct_results}

## Patient Profile:
- **Age:** {age}
- **Gender:** {gender}
- **Smoking Status:** {smoking_status}
- **Family History:** {family_history}
- **Current Symptoms:** {symptoms}
- **Occupation/Environment:** {occupation}
- **Location:** Based in {location}
- **Previous Scans:** {scan_history}
- **Medical History:** {medical_history}

## Additional Context:
{additional_context}

Create a comprehensive, PERSONALIZED health plan that addresses THIS SPECIFIC PATIENT's needs:

1. **URGENCY ASSESSMENT**
   - Classify as: IMMEDIATE | URGENT (1 week) | MODERATE (2-4 weeks) | ROUTINE
   - Explain WHY this urgency level applies to THIS patient specifically

2. **YOUR SCAN RESULTS EXPLAINED**
   - What was found in YOUR scan
   - What this means for YOU specifically
   - How these findings relate to YOUR age, lifestyle, and risk factors

3. **YOUR PERSONALIZED ACTION PLAN**
   
   **Immediate Medical Steps:**
   - Which specialists YOU should see and why
   - Timeline tailored to YOUR situation
   - Specific tests needed based on YOUR results
   
   **Lifestyle Changes FOR YOU:**
   - If smoker: Personalized cessation plan considering YOUR smoking history
   - Diet: Specific foods to add/avoid (considering {location} cuisine and availability)
   - Exercise: Activities appropriate for YOUR age, fitness level, and symptoms
   - Occupation: Specific workplace adjustments based on YOUR job in {occupation}
   - Environment: Changes needed in YOUR living/working environment
   
   **Managing YOUR Symptoms:**
   - Address each symptom: {symptoms}
   - Practical tips that work with YOUR daily routine
   - When to seek help for YOUR specific symptoms

4. **FOR YOUR FAMILY**
   - Should YOUR family members get screened? (based on YOUR family history)
   - Specific age/gender recommendations for YOUR relatives
   - Genetic considerations for YOUR family

5. **YOUR MONITORING SCHEDULE**
   - When YOU need follow-up scans (personalized timeline)
   - What appointments YOU should book now
   - Symptoms YOU should watch for

6. **SUPPORT FOR YOU**
   - Resources available in {location}
   - Support groups relevant to YOUR situation
   - Mental health support tailored to YOUR concerns

7. **YOUR QUESTIONS ANSWERED**
   - Address concerns specific to YOUR age and situation
   - Explain prognosis in YOUR case
   - Next steps explained clearly

Use "you" and "your" throughout. Make every recommendation specific to THIS patient's unique situation. Avoid generic advice.
"""

FOLLOWUP_PROMPT_TEMPLATE = """Continue the conversation with this patient about their personalized health plan.

## Their Previous Recommendations:
{previous_recommendations}

## Their Question/Concern:
{user_question}

## Their Profile:
{patient_context}

Provide a warm, personalized response that:
1. Addresses their specific question with details relevant to THEIR situation
2. References THEIR previous recommendations and explain how they apply
3. Uses "you" and "your" - speak directly to this individual
4. Shows empathy for THEIR concerns while being honest
5. Provides actionable next steps tailored to THEM

Make them feel heard and supported. This is THEIR health journey.
"""

RISK_ASSESSMENT_PROMPT = """Provide a personalized risk assessment for this individual patient.

## This Patient's Information:
{patient_data}

Create a PERSONALIZED risk assessment:

1. **YOUR Risk Level**: Low | Moderate | High | Critical
   - Explain why THIS risk level applies to YOU specifically

2. **YOUR Specific Risk Factors**:
   - List factors that apply to YOU
   - Explain how each affects YOUR personal risk
   
3. **YOUR Protective Factors**:
   - Positive aspects in YOUR life that reduce risk
   - How YOU can build on these

4. **YOUR Risk Trajectory**:
   - How YOUR risk may change based on YOUR choices
   - Timeline specific to YOUR age and situation

5. **YOUR Top 3 Prevention Priorities**:
   - Actions that will have the BIGGEST impact for YOU
   - Practical steps for YOUR lifestyle and situation

Be specific and personal - this is THEIR unique risk profile, not a generic assessment.
"""
