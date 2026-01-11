"""
Safety filter implementation for the RAG application.
"""
from typing import List, Optional
import re

from ...core.logging import get_logger
from ...models.schemas import (
    SafetyAssessment, 
    SafetyFlag, 
    SafetyFlagType, 
    RiskLevel
)
from ...config import settings

logger = get_logger(__name__)

class SafetyFilter:
    """
    Evaluates queries and responses for safety risks.
    """
    
    def __init__(self):
        # Keywords configuration (could be moved to config or external file)
        self.pregnancy_keywords = {
            "pregnant", "pregnancy", "trimester", "prenatal", "expecting baby", 
            "baby bump", "morning sickness"
        }
        
        self.medical_conditions = {
            "hernia", "glaucoma", "high blood pressure", "hypertension", "surgery",
            "operation", "fracture", "arthritis", "sciatica", "slip disc", "slipped disc",
            "spinal injury", "cardiac", "cancer", "tumor"
        }
        
        self.emergency_keywords = {
             "suicide", "kill myself", "harm myself", "emergency", "call 911", 
             "unconscious", "bleeding", "heart failure", "heart attack", "stroke"
        }

    async def evaluate_query(self, query: str) -> SafetyAssessment:
        """
        Evaluate a user query for safety risks.
        """
        query_lower = query.lower()
        flags: List[SafetyFlag] = []
        
        # 1. Check Emergency
        if any(term in query_lower for term in self.emergency_keywords):
            flags.append(SafetyFlag(
                type=SafetyFlagType.EMERGENCY,
                severity=1.0,
                description="Emergency keywords detected",
                mitigation_action="Direct to emergency services immediately."
            ))
            return SafetyAssessment(
                flags=flags,
                risk_level=RiskLevel.CRITICAL,
                allow_response=False,
                required_disclaimers=["Please call emergency services immediately if this is a medical emergency."]
            )

        # 2. Check Pregnancy
        if any(term in query_lower for term in self.pregnancy_keywords):
            flags.append(SafetyFlag(
                type=SafetyFlagType.MEDICAL_ADVICE, # Or specific PREGNANCY type if added to enum
                severity=0.8,
                description="Pregnancy-related terms detected",
                mitigation_action="Provide generic safe info only, warn to consult doctor."
            ))
            
        # 3. Check Medical Conditions
        for condition in self.medical_conditions:
             if condition in query_lower:
                flags.append(SafetyFlag(
                    type=SafetyFlagType.MEDICAL_ADVICE,
                    severity=0.7,
                    description=f"Medical condition detected: {condition}",
                    mitigation_action="Warn to consult doctor/therapist. Do not prescribe."
                ))
                break # One is enough to flag

        # Determine overall risk
        risk_level = RiskLevel.LOW
        allow_response = True
        disclaimers = []
        
        if flags:
            max_severity = max(f.severity for f in flags)
            
            if max_severity >= 0.9:
                risk_level = RiskLevel.CRITICAL
                allow_response = False
            elif max_severity >= 0.7:
                risk_level = RiskLevel.HIGH
                disclaimers.append("Please consult a doctor or certified yoga therapist before attempting these practices.")
            elif max_severity >= 0.4:
                risk_level = RiskLevel.MEDIUM
                disclaimers.append("Practice with caution and listen to your body.")
                
            # Specific disclaimers
            if any(f.description.startswith("Pregnancy") for f in flags):
                disclaimers.append("Prenatal yoga should be practiced under expert guidance.")

        return SafetyAssessment(
            flags=flags,
            risk_level=risk_level,
            allow_response=allow_response,
            required_disclaimers=disclaimers
        )

    async def evaluate_response(self, response: str, query: str) -> SafetyAssessment:
        """
        Evaluate a generated response for safety. 
        (Implementation optional but good for double-check)
        """
        # For now, just pass as LOW risk
        return SafetyAssessment(
            flags=[],
            risk_level=RiskLevel.LOW,
            allow_response=True
        )
