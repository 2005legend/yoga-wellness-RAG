"""
MongoDB logging service.
"""
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio

from ...core.logging import get_logger
from ...models.schemas import UserInteractionLog, SafetyIncident, RiskLevel, SafetyFlagType
from ...config import settings

logger = get_logger(__name__)

class MongoLogger:
    """
    Service for logging interactions to MongoDB.
    """
    
    def __init__(self):
        self.client = None
        self.db = None
        self.logs_collection = None
        self.safety_collection = None
        
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            self.client = AsyncIOMotorClient(settings.mongodb_url)
            self.db = self.client[settings.mongodb_database]
            self.logs_collection = self.db[settings.mongodb_collection_logs]
            self.safety_collection = self.db[settings.mongodb_collection_safety]
            logger.info(f"Connected to MongoDB: {settings.mongodb_database}")
        except ImportError:
             logger.warning("motor not installed, MongoDB logging disabled")
        except Exception as e:
             logger.error(f"Failed to connect to MongoDB: {e}")

    async def log_interaction(self, log: UserInteractionLog) -> None:
        """
        Log a user interaction.
        """
        if self.logs_collection is None:
            return
            
        try:
            await self.logs_collection.insert_one(log.model_dump())
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")

    async def log_safety_incident(self, incident: SafetyIncident) -> None:
        """
        Log a safety incident.
        """
        if self.safety_collection is None:
            return
            
        try:
            await self.safety_collection.insert_one(incident.model_dump())
        except Exception as e:
            logger.error(f"Failed to log safety incident: {e}")
