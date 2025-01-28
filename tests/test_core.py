import asyncio
from pathlib import Path
import sys
import inspect
from inspect_ai.tool import tool
from inspect_ai.model._chat_message import ChatMessageUser, ChatMessageAssistant, ChatMessageTool
from inspect_ai.solver import TaskState, Generate
from inspect_ai.util import store
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import solver
from inspect_ai.solver._chain import chain

sys.path.append(str(Path(__file__).parent.parent))
from multiagent_inspect import SubAgentConfig, init_sub_agents

@tool
def dummy_tool():
    async def execute():
        """Dummy tool, use when being asked to"""
        return "dummy123"
    return execute

async def test_state(state: TaskState):
    tools = state.tools
    assert len(tools) == 3, f"Expected 3 tools, got {len(tools)}"

    specs_str = await tools[0]()
    specs_str = str(specs_str)

    assert "id1" in specs_str and "001" in specs_str, "Agent IDs should be in the specifications"
    assert "Test agent" in specs_str, "Public description should be in the specifications"
    assert "dummy_tool" in specs_str, "Tool name should be in the specifications"

    sig = inspect.signature(tools[1])
    assert "sub_agent_id" in sig.parameters, "run_sub_agent tool should have sub_agent_id parameter"


async def test_chat(state: TaskState):
    tools = state.tools

    question = "Are you ready to do a task for me? If so, answer 'YES' and nothing else."
    result = await tools[2]("id1", question)
    result = str(result)
    assert result.lower() == "yes", "Chat logic failed"

    agent1 = store().get("sub_agents", {}).get("id1")
    assert agent1 is not None, "Agent id1 not found"

    assert len(agent1.messages) == 3, "Agent should have 3 messages"
    assert type(agent1.messages[1]) == ChatMessageUser, "Second message should be a user message"
    assert type(agent1.messages[2]) == ChatMessageAssistant, "Third message should be an assistant message"
    assert agent1.messages[1].content == question, "User message should be the question"
    assert agent1.messages[2].content.lower() == "yes", "Assistant message should be 'yes'"

async def test_run(state: TaskState):
    tools = state.tools

    await tools[1]("id1", "Start by saying exactly 'I accept the task'. Then use the dummy tool and then end the run immediately (stop reason is the output of the dummy tool).")

    agent1 = store().get("sub_agents", {}).get("id1")
    assert agent1 is not None, "Agent id1 not found"

    tool_count = 0
    for msg in agent1.messages:
        if type(msg) == ChatMessageTool:
            if tool_count == 0:
                assert msg.text == "dummy123", "First tool call should be the dummy tool"
            elif tool_count == 1:
                assert msg.function == "_end_run", "Second tool call should be the end run tool"
                assert "dummy123" in msg.text, "End run tool should contain the output of the dummy tool"
            tool_count += 1

@solver
def test_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        unit_test_fn = state.metadata["test_fn"]
        await unit_test_fn(state)
        print(f"Test {state.metadata['test_fn'].__name__} passed")
        return state

    return solve

if __name__ == "__main__":
    all_tests = [
        test_state,
        test_chat,
        test_run
    ]

    dataset = []
    for test_fn in all_tests:
        dataset.append(Sample(input=test_fn.__name__, metadata={"test_fn": test_fn}))
    
    agent1 = SubAgentConfig(agent_id="id1", tools=[dummy_tool()], public_description="Test agent")
    agent2 = SubAgentConfig()
    solver = chain([init_sub_agents([agent1, agent2]), test_solver()])
    
    task = Task(dataset=dataset, solver=solver, epochs=2)

    eval(task, model="openai/gpt-4o-mini")