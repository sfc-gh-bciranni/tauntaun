import requests
import yaml
import pandas as pd

from snowflake.snowpark import Session
from typing import Dict, Any, List, Optional
from uuid import uuid4
from datetime import datetime

class MessageStore:
    """Abstract base class for message storage.
    
    This defines the interface for storing and retrieving messages and agent state.
    Implementations could use different backends (e.g., SQLite, Redis, in-memory dict).
    """
    def get_messages(self) -> List[Dict[str, Any]]:
        raise NotImplementedError()
    
    def add_message(self, message: Dict[str, Any]) -> None:
        raise NotImplementedError()
    
    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        raise NotImplementedError()
    
    def update_agent_state(self, agent_id: str, updates: Dict[str, Any]) -> None:
        raise NotImplementedError()
    
    def clear_messages(self, agent_id: Optional[str] = None) -> None:
        raise NotImplementedError()
    
    def clear_agent_state(self, agent_id: str) -> None:
        raise NotImplementedError()


class DictMessageStore(MessageStore):
    """Simple in-memory implementation of MessageStore using dictionaries."""
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.agent_states: Dict[str, Dict[str, Any]] = {}
    
    def get_messages(self) -> List[Dict[str, Any]]:
        return self.messages
    
    def add_message(self, message: Dict[str, Any]) -> None:
        self.messages.append(message)
    
    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = {}
        return self.agent_states[agent_id]
    
    def update_agent_state(self, agent_id: str, updates: Dict[str, Any]) -> None:
        if agent_id not in self.agent_states:
            self.agent_states[agent_id] = {}
        self.agent_states[agent_id].update(updates)
    
    def clear_messages(self, agent_id: Optional[str] = None) -> None:
        if agent_id is None:
            self.messages = []
        else:
            self.messages = [msg for msg in self.messages if msg.get("agent_id") != agent_id]
    
    def clear_agent_state(self, agent_id: str) -> None:
        self.agent_states[agent_id] = {}


class Agent:
    """Base class for all agents in the system.

    This class defines the core interface that all agents must implement,
    as well as common functionality for message handling and state management.

    Args:
        message_store (MessageStore): The shared message store.
        agent_type (str): Type of agent (e.g., "analyst", "chart", "rag").
        name (str): Human-readable name for this agent instance.
    """
    def __init__(self,
                 message_store: MessageStore,
                 agent_type: str,
                 name: str) -> None:
        self.agent_id = str(uuid4())
        self.agent_type = agent_type
        self.name = name
        self.message_store = message_store

    def get_messages(self, filter_by_agent: bool = False) -> List[Dict[str, Any]]:
        """Get messages from the shared message store.

        Args:
            filter_by_agent (bool): If True, only return messages from this agent.

        Returns:
            List[Dict[str, Any]]: List of message dictionaries.
        """
        messages = self.message_store.get_messages()
        if not filter_by_agent:
            return messages
        return [msg for msg in messages if msg.get("agent_id") == self.agent_id]

    def add_message(self, role: str, content: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the shared message store.

        Args:
            role (str): The role of the message sender (e.g., "user" or "assistant").
            content (List[Dict[str, Any]]): The content of the message.
            metadata (Optional[Dict[str, Any]]): Additional metadata to store with the message.
        """
        message = {
            "role": role,
            "content": content,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "agent_name": self.name,
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            message.update(metadata)
        self.message_store.add_message(message)

    def get_state(self) -> Dict[str, Any]:
        """Get this agent's private state.

        Returns:
            Dict[str, Any]: The agent's current state.
        """
        return self.message_store.get_agent_state(self.agent_id)

    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update this agent's private state.

        Args:
            updates (Dict[str, Any]): The state updates to apply.
        """
        self.message_store.update_agent_state(self.agent_id, updates)

    def clear_state(self) -> None:
        """Clear this agent's private state."""
        self.message_store.clear_agent_state(self.agent_id)

    def clear_history(self, clear_all_agents: bool = False) -> None:
        """Clear message history.

        Args:
            clear_all_agents (bool): If True, clear shared message history for all agents.
        """
        self.message_store.clear_messages(None if clear_all_agents else self.agent_id)

    def can_handle_message(self, message: Dict[str, Any]) -> bool:
        """Determine if this agent can handle a given message.

        This method should be implemented by subclasses to determine
        if they can process a particular type of message or request.

        Args:
            message (Dict[str, Any]): The message to evaluate.

        Returns:
            bool: True if this agent can handle the message, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement can_handle_message")

    def call(self, prompt: str, message_history: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Dict[str, Any]:
        """Make the actual API/service call to process the message.

        This method should be implemented by subclasses to define
        how they interact with their underlying service/API.

        Args:
            prompt (str): The prompt/message to process.
            message_history (Optional[List[Dict[str, Any]]]): Previous messages for context.
            **kwargs: Additional arguments specific to the agent type.

        Returns:
            Dict[str, Any]: The service/API response.
        """
        raise NotImplementedError("Subclasses must implement call")

    def process_message(self, 
                       prompt: str, 
                       include_history: bool = True,
                       **kwargs) -> Dict[str, Any]:
        """Process a message and update the state.

        This method handles the standard flow of:
        1. Adding user message to history
        2. Getting relevant message history
        3. Making the service/API call
        4. Adding response to history
        5. Updating agent state

        Args:
            prompt (str): The prompt/message to process.
            include_history (bool): Whether to include message history in processing.
            **kwargs: Additional arguments passed to call().

        Returns:
            Dict[str, Any]: The processing results.
        """
        # Add user message to history
        self.add_message(
            role="user",
            content=[{"type": "text", "text": prompt}]
        )

        # Get message history for context if needed
        message_history = self.get_messages(filter_by_agent=not include_history)[:-1]

        # Make the service/API call
        response = self.call(prompt=prompt, message_history=message_history, **kwargs)
        
        # Add response to history
        self.add_message(
            role="assistant",
            content=response.get("content", []),
            metadata=response.get("metadata", {})
        )

        return response


class Analyst(Agent):
    """A class for interacting with the Cortex Analyst API.

    Args:
        message_store (MessageStore): The shared message store.
        session (Session): The Snowflake session object.
        semantic_model_full_path (str): The full path to the semantic model file.
        name (str): Human-readable name for this analyst instance.
    """
    def __init__(self,
                 message_store: MessageStore,
                 session: Session,
                 semantic_model_full_path: str,
                 name: str = "Default Analyst") -> None:
        super().__init__(message_store=message_store,
                        agent_type="analyst",
                        name=name)
        self.session = session
        self.semantic_model_full_path = semantic_model_full_path
        self.base_url = f"https://{session.connection.host}/api/v2/cortex/analyst/message"

    def can_handle_message(self, message: Dict[str, Any]) -> bool:
        """Determine if this analyst can handle a given message.

        Currently handles all text-based queries that might need data analysis.
        Could be made more sophisticated with message content analysis.

        Args:
            message (Dict[str, Any]): The message to evaluate.

        Returns:
            bool: True if this analyst can handle the message, False otherwise.
        """
        # For now, assume we can handle any text-based query
        # Could be enhanced with more sophisticated content analysis
        return True

    def process_message(self, 
                       prompt: str, 
                       include_history: bool = True,
                       **kwargs) -> Dict[str, Any]:
        """Process a message through the Cortex Analyst API.

        Args:
            prompt (str): The prompt to send to the API.
            include_history (bool): Whether to include message history.
            **kwargs: Additional arguments (unused).

        Returns:
            Dict[str, Any]: The API response.
        """
        # Add user message to history
        self.add_message(
            role="user",
            content=[{"type": "text", "text": prompt}]
        )

        # Get message history for context if needed
        message_history = self.get_messages(filter_by_agent=not include_history)[:-1]

        # Call API
        response = self.call(prompt=prompt, message_history=message_history)
        
        # Add assistant response to history
        self.add_message(
            role="analyst",
            content=response["message"]["content"],
            metadata={"request_id": response["request_id"]}
        )

        # Store any suggestions in agent state
        for item in response["message"]["content"]:
            if item["type"] == "text":
                self.update_state({
                    "text": item["text"]
                })
            elif item["type"] == "sql":
                self.update_state({
                    "sql": item["statement"]
                })
            elif item["type"] == "suggestions":
                self.update_state({
                    "suggestions": item["suggestions"],
                    "active_suggestion": None
                })
            else:
                raise ValueError(f"Unknown item type: {item['type']}")

        return response

    def get_semantic_model(self) -> Dict[str, Any]:
        """Get the semantic model file from the Snowflake stage."""
        semantic_model_file = self.session.file.get_stream(self.semantic_model_full_path)
        return yaml.safe_load(semantic_model_file)

    def call(self, prompt: str, message_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Call the Cortex Analyst API."""
        jwt = self.session.connection.rest.token
        this_message = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        if message_history is None:
            messages = this_message
        else:
            messages = message_history + this_message

        request_body = {
            "messages": messages,
            "semantic_model_file": f"@{self.semantic_model_full_path}",
        }
        resp = requests.post(
            url=self.base_url,
            json=request_body,
            headers={
                "Authorization": f'Snowflake Token="{jwt}"',
                "Content-Type": "application/json",
            },
        )
        resp_time = datetime.now()
        request_id = resp.headers.get("X-Snowflake-Request-Id")
        if resp.status_code < 400:
            return {**resp.json(), "request_id": request_id, "resp_time": resp_time}
        else:
            raise Exception(
                f"Failed request (id: {request_id}) with status {resp.status_code}: {resp.content}"
            )

    def get_suggestions(self) -> List[str]:
        """Get current suggestions from agent state."""
        return self.get_state().get("suggestions", [])

    def get_active_suggestion(self) -> Optional[str]:
        """Get the currently active suggestion from agent state."""
        return self.get_state().get("active_suggestion")

    def set_active_suggestion(self, suggestion: Optional[str]) -> None:
        """Set the active suggestion in agent state."""
        self.update_state({"active_suggestion": suggestion})

    def get_results(self) -> pd.DataFrame:
        """Run the SQL statement and return the result."""
        sql = self.get_state().get("sql")
        if sql is None:
            raise ValueError("No SQL statement to run")
        return self.session.sql(sql).to_pandas()
