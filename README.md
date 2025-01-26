# multiagent-inspect

Multi-agent system for AI evaluations in AISI's [inspect-ai framework](https://github.com/UKGovernmentBEIS/inspect_ai)

We provide a set of [tools](https://inspect.ai-safety-institute.org.uk/tools.html) which allow agents to delegate tasks to sub-agents. Specifically, the `init_sub_agents` function return three tools:
- `sub_agent_specs`: Return info about the sub-agent(s)
- `run_sub_agent`: Give string input to one sub-agent and execute it.
- `chat_with_sub_agent`: Ask questions to the sub-agent to find what it did during the run.

### Installation
```
pip install multiagent-inspect
pip install openai
```
Instead of `openai`, you can use any other model provider. See https://inspect.ai-safety-institute.org.uk/models.html.

### Usage
```python
from inspect_ai.solver import basic_agent
from multiagent_inspect import SubAgent, init_sub_agents
from my_inspect_tools import tool1, tool2, tool3, tool4

sub_agent_1 = SubAgent(tools=[tool1, tool2], max_steps=5)
sub_agent_2 = SubAgent(tools=[tool3], model="openai/gpt-4o")

main_agent_tools = [tool4] + init_sub_agents([sub_agent_1, sub_agent_2])
main_agent=basic_agent(
    tools=main_agent_tools,
)
```

### Contributions are welcome!
