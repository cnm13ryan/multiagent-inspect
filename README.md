# multiagent-inspect

> A lightweight, powerful multi-agent extension for the `inspect-ai` framework, enabling agents to delegate tasks to specialized sub-agents.

  

**multiagent-inspect** solves the challenge of creating complex AI evaluation scenarios. It allows a primary agent to manage and delegate tasks to a team of sub-agents, each with its own unique tools, model, and purpose. This enables you to simulate sophisticated workflows and assess an AI's ability to coordinate, delegate, and synthesize information from multiple sources.

-----

## What Problem Does This Solve?

Evaluating an AI's ability to act as a manager or orchestrator is difficult. Most evaluation frameworks treat the AI as a single entity. This library provides the essential tooling to create multi-agent systems within `inspect-ai`, allowing a primary agent to control a team of specialized sub-agents.

**It is perfect for:** AI researchers and developers who need to test complex reasoning, delegation, and task decomposition in multi-agent environments.

-----

<a id="quick-start"></a>

## 🚀 Quick Start

Get up and running with two commands.

```bash
# 1. Install the package and its peer dependencies
pip install multiagent-inspect openai

# 2. Run a basic evaluation (see example below)
python your_evaluation_script.py
```

**No API keys required** for the library itself—just for your chosen model provider (like OpenAI).

-----

## 📋 Table of Contents

  - [Quick Start](#quick-start)
  - [Key Features](#key-features)
  - [Core Tools](#core-tools)
  - [Usage Example](#usage-example)
  - [Status & Roadmap](#status-roadmap)
  - [Contributing](#contributing)
  - [License](#license)

-----

<a id="key-features"></a>

## ✨ Key Features

  - **🎯 Modular and Scalable**: Define any number of sub-agents, each with distinct models, system prompts, and tools.
  - **⚙️ Automatic Tool Generation**: `init_sub_agents` seamlessly integrates sub-agent management tools into the main agent's solver.
  - **💬 Conversation Management**: Automatically handles message history and context trimming between the main agent and sub-agents to prevent token overflow.
  - **🛡️ Robust Control Flow**: Sub-agents operate within a maximum step limit and use an internal `_end_run` tool to signal task completion, preventing infinite loops.
  - **✅ Fully Tested**: The repository includes a comprehensive test suite to ensure reliability and consistent behavior.

-----

<a id="core-tools"></a>

## 🛠️ Core Tools

The `init_sub_agents` function automatically equips the main agent with three core tools for managing its sub-agents.

| Tool                      | Purpose                                                                                | Returns                                        |
| ------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `sub_agent_specs()`       | Lists all available sub-agents and their capabilities (description, tools, etc.).      | A formatted string describing each sub-agent.  |
| `run_sub_agent()`         | Assigns a new, complete task to a specific sub-agent.                                  | A confirmation that the sub-agent has run.     |
| `chat_with_sub_agent()`   | Asks a follow-up question to a sub-agent about its previous run.                       | The sub-agent's direct textual response.       |

The main agent uses these tools to understand its team, select the right sub-agent, delegate the task, and retrieve the outcome.

-----

<a id="usage-example"></a>

## Usage Example

Here is a simple evaluation where a main agent must delegate a task to the correct sub-agent.

```python
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import basic_agent
from inspect_ai.tool import tool
from multiagent_inspect import SubAgentConfig, init_sub_agents

# Define a simple tool for one of our sub-agents
@tool
def weather_tool():
    """Returns the weather in London."""
    async def execute():
        return "The weather in London is sunny."
    return execute

# 1. Configure your sub-agents
sub_agent_weather = SubAgentConfig(
    public_description="An agent that can check the weather.",
    tools=[weather_tool],
    model="openai/gpt-4o-mini"
)

sub_agent_chat = SubAgentConfig(
    public_description="A general-purpose conversational agent.",
    model="openai/gpt-4o-mini"
)

# 2. Initialize the main agent with the sub-agent configurations
main_agent = basic_agent(
    init=init_sub_agents([sub_agent_weather, sub_agent_chat])
)

# 3. Create an evaluation task
task = Task(
    dataset=[
        Sample(input="Use the appropriate tool to find the weather in London and tell me what it is.")
    ],
    solver=main_agent,
)

# 4. Run the evaluation
if __name__ == "__main__":
    eval(task, model="openai/gpt-4o")
```

```
from inspect_ai.solver import basic_agent
from multiagent_inspect import SubAgentConfig, init_sub_agents
from my_inspect_tools import tool1, tool2, tool3, tool4

sub_agent_1 = SubAgentConfig(tools=[tool1, tool2], max_steps=5)
sub_agent_2 = SubAgentConfig(tools=[tool3], model="openai/gpt-4o")

main_agent=basic_agent(
    init=init_sub_agents([sub_agent_1, sub_agent_2]),
    tools=[tool4],
)
```

-----

<a id="status-roadmap"></a>

## 🎯 Status & Roadmap

The project is currently active and should be considered **beta**: the core functionality is stable, but APIs may evolve based on user feedback.

### ✅ v0.1 (Production Ready)

  - [x] Core `init_sub_agents` initialization function.
  - [x] Automatic generation of `sub_agent_specs`, `run_sub_agent`, and `chat_with_sub_agent` tools.
  - [x] Per-sub-agent context and history management.
  - [x] Comprehensive test suite is passing.
  - [x] Core functionality is stable and published to PyPI.

### 🔄 v0.2 (Coming Soon)

  - [ ] More sophisticated sub-agent state management.
  - [ ] Tools for direct agent-to-agent communication.
  - [ ] Helper functions for common multi-agent patterns (e.g., broadcast, map-reduce).
  - [ ] Export and logging of inter-agent conversations.

-----

<a id="contributing"></a>

## 🤝 Contributing

Contributions are welcome! This project is intended to be a community-driven extension to the `inspect-ai` ecosystem. If you have ideas for features, improvements, or have found a bug, please feel free to open an issue or submit a pull request on our [GitHub repository](https://github.com/AndonLabs/multiagent-inspect).

-----

<a id="license"></a>

## 📝 License

This project is licensed under the **Apache License, Version 2.0**. You can find the full license text in the `LICENSE` file.
