// MongoDB initialization script for wellness RAG application

// Switch to the wellness_rag database
db = db.getSiblingDB('wellness_rag');

// Create collections with proper indexes
db.createCollection('interaction_logs');
db.createCollection('safety_incidents');
db.createCollection('knowledge_documents');

// Create indexes for interaction_logs collection
db.interaction_logs.createIndex({ "session_id": 1 });
db.interaction_logs.createIndex({ "timestamp": 1 });
db.interaction_logs.createIndex({ "query.original": "text" });
db.interaction_logs.createIndex({ "safety.riskLevel": 1 });

// Create indexes for safety_incidents collection
db.safety_incidents.createIndex({ "session_id": 1 });
db.safety_incidents.createIndex({ "timestamp": 1 });
db.safety_incidents.createIndex({ "incident_type": 1 });
db.safety_incidents.createIndex({ "severity": 1 });
db.safety_incidents.createIndex({ "resolved": 1 });

// Create indexes for knowledge_documents collection
db.knowledge_documents.createIndex({ "category": 1 });
db.knowledge_documents.createIndex({ "source": 1 });
db.knowledge_documents.createIndex({ "last_updated": 1 });
db.knowledge_documents.createIndex({ "title": "text", "content": "text" });

// Create a user for the application
db.createUser({
  user: "wellness_app",
  pwd: "wellness_password",
  roles: [
    {
      role: "readWrite",
      db: "wellness_rag"
    }
  ]
});

print("MongoDB initialization completed for wellness RAG application");