# Tauntaun (Work in Progress)

Tauntauns (Star Wars reference) are a creature native to the planet Hoth, known for their ability to withstand extreme cold (Snow) - They often traveled in herds and were used as transports by the Rebel Alliance. There are some fun Snowflake Agents parallels there.

The goal of Tauntaun is to make it easy to use Snowflake Agents like Cortex Analyst in just a few lines of code. Don't worry about message history, state management, or the underlying API.

## High Level Usage

```python
from tauntaun.agents import Analyst, DictMessageStore

session = ... # create a snowflake session

# create a message store
messages = DictMessageStore()

# create an analyst agent
analyst = Analyst(session=session, message_store=messages, semantic_model_full_path="...")

# ask a question
analyst.process_message(prompt="What is the revenue for the year 2024?")

# get the query results
results = analyst.get_results()

# ask a follow up question - automatically uses the previous analyst response
analyst.process_message(prompt="Can you break it out by region?")

# get the query results
results = analyst.get_results()
```



## Overview

Tauntaun provides a flexible framework for building and managing conversational agents that can interact with Snowflake services. The core features include:

- Abstract `Agent` base class for building different types of agents
- Flexible message and state management through the `MessageStore` interface
- Built-in support for Snowflake's Cortex Analyst
- Shared message history between agents
- Private state management per agent

## Installation

```bash
pip install snowflake-snowpark-python
# Additional dependencies based on your needs (e.g., streamlit, pandas)
```

## Quick Start

Here's a simple example using the Cortex Analyst agent (from `tauntaun_demo.ipynb`):

```python
from snowflake.snowpark import Session
from tauntaun.agents import Analyst, DictMessageStore

# Create a Snowflake session
session = Session.builder.config("connection_name", "<YOUR_CONNECTION_NAME>").create()

# Create a message store for managing conversation history and state
messages = DictMessageStore()

# Initialize an analyst agent
analyst = Analyst(
    session=session,
    message_store=messages,
    semantic_model_full_path="cortex_analyst_demo.revenue_timeseries.raw_data/revenue_timeseries.yaml",
)

# Ask a question
analyst.process_message(prompt="What is the revenue for the year 2024?")

# Get the agent's state (includes generated SQL, text response, etc.)
print(analyst.get_state())

# Get the query results as a pandas DataFrame
results = analyst.get_results()
print(results)
```

## Core Components

### MessageStore

The `MessageStore` interface defines how messages and agent state are stored and retrieved. The framework includes a simple in-memory implementation (`DictMessageStore`), but you can implement your own storage backends (e.g., SQLite, Redis).

### Agent

The base `Agent` class provides:
- Message history management
- State management
- Abstract methods for message handling and API calls

### Analyst

The `Analyst` class implements the Cortex Analyst API integration, providing:
- Natural language to SQL translation
- Query execution
- Suggestion handling
- State management for SQL queries and results

## Example Usage

Check out `tauntaun_demo.ipynb` for a complete example showing:
1. Basic setup and initialization
2. Asking questions and getting responses
3. Accessing generated SQL
4. Getting query results
5. Working with suggestions
