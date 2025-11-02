"""
LangGraph Agent for CT Scan Recommendation System.
Uses StateGraph for managing conversation flow and state.
"""

from typing import TypedDict, Annotated, Sequence, Optional, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import operator
from .prompts import (
    SYSTEM_PROMPT,
    RECOMMENDATION_PROMPT_TEMPLATE,
    FOLLOWUP_PROMPT_TEMPLATE,
    RISK_ASSESSMENT_PROMPT
)
from .utils import (
    format_patient_profile,
    format_ct_results,
    sanitize_input,
    extract_urgency_level
)


# Define the state schema
class AgentState(TypedDict):
    """State schema for the recommendation agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    patient_profile: Dict[str, Any]
    ct_results: Dict[str, Any]
    current_recommendations: Optional[str]
    urgency_level: Optional[str]
    conversation_context: Dict[str, Any]
    next_action: Optional[str]


class CTRecommendationAgent:
    """
    LangGraph-based agent for personalized CT scan recommendations.
    Uses latest LangChain patterns with llm.invoke().
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        api_key: Optional[str] = None
    ):
        """Initialize the agent with LLM configuration."""
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )
        
        # Build the state graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("validate_input", self._validate_input_node)
        workflow.add_node("generate_recommendations", self._generate_recommendations_node)
        workflow.add_node("handle_followup", self._handle_followup_node)
        workflow.add_node("assess_risk", self._assess_risk_node)
        
        # Define entry point
        workflow.set_entry_point("validate_input")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "validate_input",
            self._route_after_validation,
            {
                "generate_recommendations": "generate_recommendations",
                "handle_followup": "handle_followup",
                "assess_risk": "assess_risk",
                "end": END
            }
        )
        
        # All nodes lead to END after processing
        workflow.add_edge("generate_recommendations", END)
        workflow.add_edge("handle_followup", END)
        workflow.add_edge("assess_risk", END)
        
        return workflow
    
    def _validate_input_node(self, state: AgentState) -> AgentState:
        """Validate and sanitize input data."""
        # Sanitize any text inputs
        if state.get("messages"):
            last_message = state["messages"][-1]
            if isinstance(last_message, HumanMessage):
                sanitized_content = sanitize_input(last_message.content)
                state["messages"][-1] = HumanMessage(content=sanitized_content)
        
        # Determine next action based on state
        has_profile = bool(state.get("patient_profile"))
        has_ct = bool(state.get("ct_results"))
        has_recommendations = bool(state.get("current_recommendations"))
        
        if has_profile and has_ct and not has_recommendations:
            state["next_action"] = "generate_recommendations"
        elif has_recommendations and len(state.get("messages", [])) > 0:
            state["next_action"] = "handle_followup"
        elif state.get("conversation_context", {}).get("request_risk_assessment"):
            state["next_action"] = "assess_risk"
        else:
            state["next_action"] = "end"
        
        return state
    
    def _route_after_validation(self, state: AgentState) -> str:
        """Route to appropriate node after validation."""
        return state.get("next_action", "end")
    
    def _generate_recommendations_node(self, state: AgentState) -> AgentState:
        """Generate initial personalized recommendations."""
        patient_profile = state.get("patient_profile", {})
        ct_results = state.get("ct_results", {})
        
        # Format the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", RECOMMENDATION_PROMPT_TEMPLATE)
        ])
        
        # Prepare variables
        formatted_prompt = prompt.format_messages(
            ct_results=format_ct_results(ct_results),
            age=patient_profile.get("age", "Not provided"),
            gender=patient_profile.get("gender", "Not provided"),
            smoking_status=patient_profile.get("smoking_status", "Not provided"),
            family_history=patient_profile.get("family_history", "None reported"),
            symptoms=patient_profile.get("symptoms", "None reported"),
            occupation=patient_profile.get("occupation", "Not provided"),
            location=patient_profile.get("location", "Not provided"),
            scan_history=patient_profile.get("scan_history", "No previous scans"),
            medical_history=patient_profile.get("medical_history", "None reported"),
            additional_context=patient_profile.get("additional_context", "None")
        )
        
        # Invoke LLM (latest pattern)
        response = self.llm.invoke(formatted_prompt)
        
        # Extract recommendations
        recommendations = response.content
        
        # Extract urgency level
        urgency = extract_urgency_level(recommendations)
        
        # Update state
        state["current_recommendations"] = recommendations
        state["urgency_level"] = urgency
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=recommendations)
        ]
        
        return state
    
    def _handle_followup_node(self, state: AgentState) -> AgentState:
        """Handle follow-up questions from the user."""
        messages = state.get("messages", [])
        
        if not messages:
            return state
        
        # Get the last user message
        last_user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break
        
        if not last_user_message:
            return state
        
        # Create context
        patient_context = format_patient_profile(state.get("patient_profile", {}))
        previous_recommendations = state.get("current_recommendations", "No previous recommendations")
        
        # Format follow-up prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", FOLLOWUP_PROMPT_TEMPLATE),
            MessagesPlaceholder(variable_name="history")
        ])
        
        # Get recent conversation history (exclude current message)
        history = messages[:-1][-10:]  # Last 10 messages for context
        
        formatted_prompt = prompt.format_messages(
            previous_recommendations=previous_recommendations,
            user_question=last_user_message,
            patient_context=patient_context,
            history=history
        )
        
        # Invoke LLM
        response = self.llm.invoke(formatted_prompt)
        
        # Add response to messages
        state["messages"] = state["messages"] + [AIMessage(content=response.content)]
        
        return state
    
    def _assess_risk_node(self, state: AgentState) -> AgentState:
        """Perform comprehensive risk assessment."""
        patient_profile = state.get("patient_profile", {})
        ct_results = state.get("ct_results", {})
        
        # Combine all patient data
        patient_data = {
            "profile": patient_profile,
            "ct_results": ct_results
        }
        
        # Format prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", RISK_ASSESSMENT_PROMPT)
        ])
        
        formatted_prompt = prompt.format_messages(
            patient_data=f"Profile:\n{format_patient_profile(patient_profile)}\n\nCT Results:\n{format_ct_results(ct_results)}"
        )
        
        # Invoke LLM
        response = self.llm.invoke(formatted_prompt)
        
        # Add to messages
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=response.content)
        ]
        
        return state
    
    def generate_recommendations(
        self,
        patient_profile: Dict[str, Any],
        ct_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate initial recommendations for a patient.
        
        Args:
            patient_profile: Patient demographic and history data
            ct_results: CT scan findings and details
            
        Returns:
            Dictionary with recommendations and metadata
        """
        initial_state = AgentState(
            messages=[],
            patient_profile=patient_profile,
            ct_results=ct_results,
            current_recommendations=None,
            urgency_level=None,
            conversation_context={},
            next_action=None
        )
        
        result = self.app.invoke(initial_state)
        
        return {
            "recommendations": result["current_recommendations"],
            "urgency_level": result["urgency_level"],
            "messages": result["messages"]
        }
    
    def chat(
        self,
        user_message: str,
        state: AgentState
    ) -> Dict[str, Any]:
        """
        Handle a chat message in the context of existing conversation.
        
        Args:
            user_message: User's question or message
            state: Current conversation state
            
        Returns:
            Updated state with new messages
        """
        # Add user message to state
        state["messages"] = state.get("messages", []) + [
            HumanMessage(content=user_message)
        ]
        
        # Process through graph
        result = self.app.invoke(state)
        
        return result
    
    def assess_risk(
        self,
        patient_profile: Dict[str, Any],
        ct_results: Dict[str, Any]
    ) -> str:
        """
        Perform standalone risk assessment.
        
        Args:
            patient_profile: Patient data
            ct_results: CT scan results
            
        Returns:
            Risk assessment text
        """
        initial_state = AgentState(
            messages=[],
            patient_profile=patient_profile,
            ct_results=ct_results,
            current_recommendations=None,
            urgency_level=None,
            conversation_context={"request_risk_assessment": True},
            next_action=None
        )
        
        result = self.app.invoke(initial_state)
        
        # Return the last AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                return msg.content
        
        return "Unable to generate risk assessment."


# Factory function for easy initialization
def create_agent(model_name: str = "gpt-4o", temperature: float = 0.7, api_key: Optional[str] = None) -> CTRecommendationAgent:
    """
    Create and return a configured CTRecommendationAgent.
    
    Args:
        model_name: OpenAI model name
        temperature: LLM temperature (0-1)
        api_key: OpenAI API key (optional if set in environment)
        
    Returns:
        Configured agent instance
    """
    return CTRecommendationAgent(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key
    )
