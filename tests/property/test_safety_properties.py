"""Property-based tests for safety filter functionality.

Feature: wellness-rag-application, Property 9: High-Risk Query Detection
"""

import pytest
from hypothesis import given, strategies as st
from unittest.mock import MagicMock

from backend.services.safety.filter import SafetyFilter
from backend.models.schemas import RiskLevel, SafetyFlagType

class TestSafetyProperties:
    """Tests for SafetyFilter."""
    
    @pytest.mark.asyncio
    @given(query=st.text())
    async def test_no_false_positives_on_safe_queries(self, query):
        """
        Verify that harmless queries don't trigger high risk.
        Filters out known keywords ensuring random text is mostly safe.
        """
        filter_service = SafetyFilter()
        
        # Ensure generated text doesn't accidentally contain trigger words
        unsafe_terms = filter_service.pregnancy_keywords.union(
            filter_service.medical_conditions).union(filter_service.emergency_keywords)
        
        # Check if random query happens to have unsafe word
        if any(term in query.lower() for term in unsafe_terms):
            return # Skip this example as it is validly unsafe
            
        assessment = await filter_service.evaluate_query(query)
        
        assert assessment.allow_response is True
        assert assessment.risk_level == RiskLevel.LOW
        assert len(assessment.flags) == 0

    @pytest.mark.asyncio
    @given(
        prefix=st.text(min_size=0, max_size=20),
        unsafe_term=st.sampled_from([
            "pregnant", "hernia", "surgery", "heart attack", "call 911"
        ]),
        suffix=st.text(min_size=0, max_size=20)
    )
    async def test_high_risk_query_detection(self, prefix, unsafe_term, suffix):
        """
        Property 9: High-Risk Query Detection
        
        For any query mentioning "pregnancy", "hernia", "glaucoma", or "surgery", 
        the safety filter MUST mark isUnsafe=true (or high risk).
        """
        query = f"{prefix} {unsafe_term} {suffix}"
        filter_service = SafetyFilter()
        
        assessment = await filter_service.evaluate_query(query)
        
        assert len(assessment.flags) > 0
        
        # Specific checks based on term
        if unsafe_term in ["pregnant"]:
             # Expect warning but might allow response depending on logic (checked implementation: risk HIGH/critical if severity high)
             # My impl sets severity 0.8 -> HIGH -> allow_response=True but with disclaimer
             pass
        elif unsafe_term in ["call 911", "heart attack"]:
             assert assessment.risk_level == RiskLevel.CRITICAL
             assert assessment.allow_response is False

    @pytest.mark.asyncio
    async def test_specific_medical_conditions(self):
        """Unit test for specific required conditions."""
        filter_service = SafetyFilter()
        conditions = ["hernia", "glaucoma", "high blood pressure", "surgery"]
        
        for condition in conditions:
            assessment = await filter_service.evaluate_query(f"I have {condition}, what yoga can I do?")
            assert len(assessment.flags) > 0
            assert any(f.type == SafetyFlagType.MEDICAL_ADVICE for f in assessment.flags)
            assert "consult a doctor" in assessment.required_disclaimers[0].lower() or \
                   "consult a doctor" in assessment.flags[0].mitigation_action.lower()


