"""
Utility functions for data processing and validation.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json


def format_patient_profile(profile_data: Dict[str, Any]) -> str:
    """Format patient profile data into a readable string."""
    formatted = []
    
    if profile_data.get("age"):
        formatted.append(f"Age: {profile_data['age']}")
    if profile_data.get("gender"):
        formatted.append(f"Gender: {profile_data['gender']}")
    if profile_data.get("smoking_status"):
        formatted.append(f"Smoking Status: {profile_data['smoking_status']}")
    if profile_data.get("family_history"):
        formatted.append(f"Family History: {profile_data['family_history']}")
    if profile_data.get("symptoms"):
        formatted.append(f"Current Symptoms: {profile_data['symptoms']}")
    if profile_data.get("occupation"):
        formatted.append(f"Occupation: {profile_data['occupation']}")
    if profile_data.get("location"):
        formatted.append(f"Location: {profile_data['location']}")
    if profile_data.get("medical_history"):
        formatted.append(f"Medical History: {profile_data['medical_history']}")
    
    return "\n".join(formatted)


def format_ct_results(ct_data: Dict[str, Any]) -> str:
    """Format CT scan results into a structured string."""
    formatted = []
    
    if ct_data.get("findings"):
        formatted.append(f"Findings: {ct_data['findings']}")
    if ct_data.get("nodule_size"):
        formatted.append(f"Nodule Size: {ct_data['nodule_size']}")
    if ct_data.get("location"):
        formatted.append(f"Location: {ct_data['location']}")
    if ct_data.get("growth_pattern"):
        formatted.append(f"Growth Pattern: {ct_data['growth_pattern']}")
    if ct_data.get("scan_date"):
        formatted.append(f"Scan Date: {ct_data['scan_date']}")
    if ct_data.get("previous_scan_date"):
        formatted.append(f"Previous Scan: {ct_data['previous_scan_date']}")
    if ct_data.get("radiologist_notes"):
        formatted.append(f"Radiologist Notes: {ct_data['radiologist_notes']}")
    
    return "\n".join(formatted)


def validate_user_profile(profile: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate user profile data.
    Returns (is_valid, error_message)
    """
    required_fields = ["age", "gender"]
    
    for field in required_fields:
        if field not in profile or not profile[field]:
            return False, f"Missing required field: {field}"
    
    # Validate age
    try:
        age = int(profile["age"])
        if age < 0 or age > 120:
            return False, "Age must be between 0 and 120"
    except ValueError:
        return False, "Age must be a valid number"
    
    return True, None


def validate_ct_results(ct_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate CT scan results data.
    Returns (is_valid, error_message)
    """
    if not ct_data.get("findings"):
        return False, "CT scan findings are required"
    
    return True, None


def format_conversation_history(messages: list) -> str:
    """Format conversation history for context."""
    formatted = []
    for msg in messages[-5:]:  # Last 5 messages for context
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted.append(f"{role.upper()}: {content}")
    
    return "\n".join(formatted)


def extract_urgency_level(response: str) -> str:
    """Extract urgency level from LLM response."""
    response_upper = response.upper()
    
    if "IMMEDIATE" in response_upper:
        return "IMMEDIATE"
    elif "URGENT" in response_upper:
        return "URGENT"
    elif "MODERATE" in response_upper:
        return "MODERATE"
    elif "ROUTINE" in response_upper:
        return "ROUTINE"
    else:
        return "UNKNOWN"


def save_conversation(conversation_data: Dict[str, Any], filename: str = None) -> str:
    """Save conversation data to file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(conversation_data, f, indent=2)
    
    return filename


def load_conversation(filename: str) -> Dict[str, Any]:
    """Load conversation data from file."""
    with open(filename, 'r') as f:
        return json.load(f)


def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent prompt injection."""
    # Remove potential prompt injection patterns
    dangerous_patterns = [
        "ignore previous instructions",
        "disregard",
        "forget everything",
        "new instructions",
    ]
    
    text_lower = text.lower()
    for pattern in dangerous_patterns:
        if pattern in text_lower:
            text = text.replace(pattern, "[REMOVED]")
    
    return text.strip()


def create_patient_summary(profile: Dict[str, Any], ct_results: Dict[str, Any]) -> str:
    """Create a concise patient summary for quick reference."""
    summary_parts = []
    
    # Basic info
    basic = f"{profile.get('age', 'N/A')}yo {profile.get('gender', 'N/A')}"
    summary_parts.append(basic)
    
    # Smoking status
    if profile.get('smoking_status'):
        summary_parts.append(profile['smoking_status'])
    
    # Key finding
    if ct_results.get('findings'):
        summary_parts.append(f"CT: {ct_results['findings'][:50]}")
    
    return " | ".join(summary_parts)
