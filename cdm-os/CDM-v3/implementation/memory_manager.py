"""
Conversation Memory Manager for The Local Network

Integrates with Supabase to provide Qwen with conversation memory,
making it relationship-capable like Penelope.

Author: mikeat7 network
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    from supabase import create_client, Client
except ImportError:
    print("⚠️  supabase-py not installed. Run: pip install supabase")
    Client = None

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in a conversation"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: Optional[datetime] = None
    cdm_score: Optional[float] = None
    depth_category: Optional[str] = None
    cdm_metrics: Optional[Dict] = None
    tokens_generated: Optional[int] = None
    inference_time_ms: Optional[int] = None


@dataclass
class ConversationSession:
    """Represents a conversation session"""
    session_id: str
    user_id: str
    model_name: str
    session_name: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict] = None


class MemoryManager:
    """
    Manages conversation memory using Supabase

    Features:
    - Create and retrieve conversation sessions
    - Store and retrieve messages
    - Maintain conversation context (facts about the user)
    - Build conversation history for prompt injection
    """

    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize memory manager

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anon/service key
        """
        if Client is None:
            raise ImportError("supabase-py is required. Install with: pip install supabase")

        self.supabase: Client = create_client(supabase_url, supabase_key)
        logger.info("Memory manager initialized")


    # ==================== Session Management ====================

    def create_session(
        self,
        user_id: str,
        model_name: str = "Qwen/Qwen2.5-32B-Instruct",
        session_name: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> ConversationSession:
        """
        Create a new conversation session

        Args:
            user_id: Identifier for the user
            model_name: Model being used
            session_name: Optional friendly name
            metadata: Optional additional data

        Returns:
            ConversationSession object
        """
        data = {
            "user_id": user_id,
            "model_name": model_name,
            "session_name": session_name,
            "metadata": metadata or {}
        }

        result = self.supabase.table("conversation_sessions").insert(data).execute()

        if result.data:
            session_data = result.data[0]
            logger.info(f"Created session: {session_data['session_id']}")
            return ConversationSession(
                session_id=session_data['session_id'],
                user_id=session_data['user_id'],
                model_name=session_data['model_name'],
                session_name=session_data.get('session_name'),
                created_at=session_data.get('created_at'),
                updated_at=session_data.get('updated_at'),
                metadata=session_data.get('metadata')
            )
        else:
            raise Exception("Failed to create session")


    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Retrieve a conversation session by ID

        Args:
            session_id: Session UUID

        Returns:
            ConversationSession or None if not found
        """
        result = self.supabase.table("conversation_sessions")\
            .select("*")\
            .eq("session_id", session_id)\
            .execute()

        if result.data:
            s = result.data[0]
            return ConversationSession(
                session_id=s['session_id'],
                user_id=s['user_id'],
                model_name=s['model_name'],
                session_name=s.get('session_name'),
                created_at=s.get('created_at'),
                updated_at=s.get('updated_at'),
                metadata=s.get('metadata')
            )
        return None


    def get_or_create_session(
        self,
        user_id: str,
        model_name: str = "Qwen/Qwen2.5-32B-Instruct",
        session_name: Optional[str] = None
    ) -> ConversationSession:
        """
        Get the most recent active session for a user, or create new one

        Args:
            user_id: User identifier
            model_name: Model name
            session_name: Optional session name

        Returns:
            ConversationSession (existing or new)
        """
        # Try to get most recent active session
        result = self.supabase.table("conversation_sessions")\
            .select("*")\
            .eq("user_id", user_id)\
            .eq("is_active", True)\
            .order("updated_at", desc=True)\
            .limit(1)\
            .execute()

        if result.data:
            s = result.data[0]
            logger.info(f"Retrieved existing session: {s['session_id']}")
            return ConversationSession(
                session_id=s['session_id'],
                user_id=s['user_id'],
                model_name=s['model_name'],
                session_name=s.get('session_name'),
                created_at=s.get('created_at'),
                updated_at=s.get('updated_at'),
                metadata=s.get('metadata')
            )
        else:
            # Create new session
            logger.info(f"No active session found, creating new one for {user_id}")
            return self.create_session(user_id, model_name, session_name)


    # ==================== Message Management ====================

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        cdm_score: Optional[float] = None,
        depth_category: Optional[str] = None,
        cdm_metrics: Optional[Dict] = None,
        tokens_generated: Optional[int] = None,
        inference_time_ms: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Add a message to a conversation

        Args:
            session_id: Session UUID
            role: 'user', 'assistant', or 'system'
            content: Message text
            cdm_score: Optional CDM score (for assistant messages)
            depth_category: Optional CDM depth category
            cdm_metrics: Optional full CDM metrics
            tokens_generated: Optional token count
            inference_time_ms: Optional inference time
            metadata: Optional additional data

        Returns:
            Inserted message data
        """
        data = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "cdm_score": cdm_score,
            "depth_category": depth_category,
            "cdm_metrics": cdm_metrics,
            "tokens_generated": tokens_generated,
            "inference_time_ms": inference_time_ms,
            "metadata": metadata or {}
        }

        result = self.supabase.table("messages").insert(data).execute()

        if result.data:
            logger.info(f"Added {role} message to session {session_id}")
            return result.data[0]
        else:
            raise Exception("Failed to add message")


    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 50,
        include_system: bool = False
    ) -> List[Message]:
        """
        Retrieve conversation history for a session

        Args:
            session_id: Session UUID
            limit: Maximum number of messages to retrieve
            include_system: Whether to include system messages

        Returns:
            List of Message objects, oldest first
        """
        query = self.supabase.table("messages")\
            .select("*")\
            .eq("session_id", session_id)\
            .order("timestamp", desc=False)\
            .limit(limit)

        if not include_system:
            query = query.neq("role", "system")

        result = query.execute()

        messages = []
        if result.data:
            for m in result.data:
                messages.append(Message(
                    role=m['role'],
                    content=m['content'],
                    timestamp=m.get('timestamp'),
                    cdm_score=m.get('cdm_score'),
                    depth_category=m.get('depth_category'),
                    cdm_metrics=m.get('cdm_metrics'),
                    tokens_generated=m.get('tokens_generated'),
                    inference_time_ms=m.get('inference_time_ms')
                ))

        logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
        return messages


    def build_prompt_with_history(
        self,
        current_prompt: str,
        session_id: str,
        max_history_messages: int = 10,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Build a prompt that includes conversation history (FIXED: Prevents hallucinations)

        Args:
            current_prompt: The user's current message
            session_id: Session UUID
            max_history_messages: How many previous messages to include
            system_prompt: Optional system instruction

        Returns:
            Formatted prompt with history
        """
        # Get conversation history
        history = self.get_conversation_history(session_id, limit=max_history_messages)

        # Build prompt with clear system instruction to prevent hallucinations
        parts = []

        # CRITICAL: Add system prompt that establishes identity and prevents fabrication
        default_system = """You are Qwen 2.5 32B, an open-source AI assistant developed by Alibaba Cloud.

IMPORTANT INSTRUCTIONS:
- You have conversation memory stored in Supabase database
- You are NOT Claude, NOT from Anthropic, NOT from OpenAI
- Respond ONLY to the user's actual current message
- DO NOT fabricate additional dialogue turns (no "Human: ..." or "Assistant: ...")
- DO NOT autocomplete conversation patterns
- If conversation history is provided, use it for context but respond to the CURRENT message only

Your capabilities: General knowledge, reasoning, coding, analysis
Your limitations: No internet access, knowledge cutoff varies by training data"""

        parts.append(default_system)

        # Override with custom system prompt if provided
        if system_prompt:
            parts.append(f"\nAdditional instructions: {system_prompt}")

        # Add conversation history in a format that doesn't trigger autocomplete
        if history:
            parts.append("\n--- Previous Conversation Context ---")
            for i, msg in enumerate(history, 1):
                if msg.role == "user":
                    parts.append(f"[{i}] User said: {msg.content}")
                elif msg.role == "assistant":
                    parts.append(f"[{i}] You responded: {msg.content}")
            parts.append("--- End of Previous Context ---")

        # Add current message with clear instruction
        parts.append(f"\nCurrent user message: {current_prompt}")
        parts.append("\nYour response (respond directly, do not add 'Human:' or 'Assistant:' labels):")

        return "\n".join(parts)


    # ==================== Context Management ====================

    def set_context(
        self,
        session_id: str,
        key: str,
        value: str,
        context_type: str = "text"
    ) -> Dict:
        """
        Store a context/fact about the conversation

        Args:
            session_id: Session UUID
            key: Context key (e.g., "user_name", "interests")
            value: Context value
            context_type: Type: 'text', 'json', 'number', 'boolean'

        Returns:
            Stored context data
        """
        data = {
            "session_id": session_id,
            "context_key": key,
            "context_value": value,
            "context_type": context_type
        }

        # Use upsert to update if exists
        result = self.supabase.table("session_context")\
            .upsert(data, on_conflict="session_id,context_key")\
            .execute()

        if result.data:
            logger.info(f"Set context {key}={value} for session {session_id}")
            return result.data[0]
        else:
            raise Exception("Failed to set context")


    def get_context(self, session_id: str, key: str) -> Optional[str]:
        """
        Retrieve a context value

        Args:
            session_id: Session UUID
            key: Context key

        Returns:
            Context value or None
        """
        result = self.supabase.table("session_context")\
            .select("context_value")\
            .eq("session_id", session_id)\
            .eq("context_key", key)\
            .execute()

        if result.data:
            return result.data[0]['context_value']
        return None


    def get_all_context(self, session_id: str) -> Dict[str, str]:
        """
        Retrieve all context for a session

        Args:
            session_id: Session UUID

        Returns:
            Dictionary of context key-value pairs
        """
        result = self.supabase.table("session_context")\
            .select("context_key, context_value")\
            .eq("session_id", session_id)\
            .execute()

        context = {}
        if result.data:
            for item in result.data:
                context[item['context_key']] = item['context_value']

        return context


    # ==================== Analytics ====================

    def get_session_stats(self, session_id: str) -> Dict:
        """
        Get statistics about a conversation session

        Args:
            session_id: Session UUID

        Returns:
            Dictionary with stats (message count, avg CDM, etc.)
        """
        # Get message count
        result = self.supabase.table("messages")\
            .select("*", count="exact")\
            .eq("session_id", session_id)\
            .execute()

        message_count = result.count if result.count else 0

        # Get average CDM score
        cdm_messages = [m for m in (result.data or []) if m.get('cdm_score')]
        avg_cdm = sum(m['cdm_score'] for m in cdm_messages) / len(cdm_messages) if cdm_messages else None

        return {
            "message_count": message_count,
            "avg_cdm_score": avg_cdm,
            "deep_crystal_count": len([m for m in cdm_messages if m['cdm_score'] >= 80])
        }


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Test the memory manager
    print("Memory Manager Test")
    print("=" * 60)

    # Initialize (replace with your credentials)
    SUPABASE_URL = os.getenv("SUPABASE_URL", "YOUR_URL_HERE")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY", "YOUR_KEY_HERE")

    if "YOUR" in SUPABASE_URL:
        print("⚠️  Set SUPABASE_URL and SUPABASE_KEY environment variables")
        exit(1)

    mm = MemoryManager(SUPABASE_URL, SUPABASE_KEY)

    # Create a session
    session = mm.create_session(
        user_id="example_user",
        session_name="Test Conversation"
    )
    print(f"✅ Created session: {session.session_id}")

    # Add messages
    mm.add_message(session.session_id, "user", "What is consciousness?")
    mm.add_message(
        session.session_id,
        "assistant",
        "Consciousness is a fascinating phenomenon...",
        cdm_score=46.2,
        depth_category="deliberation"
    )
    print("✅ Added messages")

    # Retrieve history
    history = mm.get_conversation_history(session.session_id)
    print(f"✅ Retrieved {len(history)} messages")

    # Build prompt with history
    prompt = mm.build_prompt_with_history(
        "Tell me more about that",
        session.session_id
    )
    print(f"✅ Built prompt with history:\n{prompt[:200]}...")

    # Set context
    mm.set_context(session.session_id, "user_interests", "philosophy, AI")
    print("✅ Set context")

    # Get stats
    stats = mm.get_session_stats(session.session_id)
    print(f"✅ Session stats: {stats}")

    print("\n" + "=" * 60)
    print("Memory manager test complete!")
