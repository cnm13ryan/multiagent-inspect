import pytest
import asyncio
from pathlib import Path
import sys
import inspect
from inspect_ai.tool import tool
from inspect_ai.model._chat_message import ChatMessageUser, ChatMessageAssistant, ChatMessageTool

sys.path.append(str(Path(__file__).parent.parent))
from multiagent_inspect import SubAgent, init_sub_agents
from multiagent_inspect.core import _get_agent

@tool
def dummy_tool():
    async def execute():
        """Dummy tool, use when being asked to"""
        return "dummy123"
    return execute

@pytest.mark.asyncio
async def test_sub_agent_class():
    # Create a test sub agent
    agent1 = SubAgent(tools=[dummy_tool()], public_description="Test agent")
    agent2 = SubAgent()

    assert agent1.agent_id == "001"
    assert agent2.agent_id == "002"
    assert agent1.model == "openai/gpt-4o-mini-2024-07-18-free"
    assert len(str(agent1)) > 0
    assert "Test agent" in str(agent1)
    assert "dummy_tool" in str(agent1)

    print("Agent class test passed")

@pytest.mark.asyncio
async def test_default_init():
    agent1 = SubAgent()
    agent2 = SubAgent()

    tools = init_sub_agents([agent1, agent2])
    assert len(tools) == 3, f"Expected 3 tools, got {len(tools)}"

    sig = inspect.signature(tools[1])
    assert "sub_agent_id" in sig.parameters, "run_sub_agent tool should have sub_agent_id parameter"

    first_agent = await _get_agent(agent1.agent_id)
    assert first_agent == agent1, "Agent retrieval failed"
    
    second_agent = await _get_agent(agent2.agent_id)
    assert second_agent == agent2, "Agent retrieval failed"
    
    default_agent = await _get_agent()
    assert default_agent == agent1, "Default agent should be the first agent"
    
    non_existent = await _get_agent("000")
    assert non_existent is None, "Non-existent agent should return None"
    
    print("Default case test passed")

@pytest.mark.asyncio
async def test_single_init():
    agent1 = SubAgent()

    tools = init_sub_agents([agent1])
    assert len(tools) == 3, f"Expected 3 tools, got {len(tools)}"

    sig = inspect.signature(tools[1])
    assert "sub_agent_id" not in sig.parameters, "run_sub_agent tool should NOT have sub_agent_id parameter in single mode"

    default_agent = await _get_agent()
    assert default_agent == agent1, f"Default agent should be the first agent. Expected: {agent1.agent_id}, Got: {default_agent.agent_id}"
    
    non_existent = await _get_agent("000")
    assert non_existent is None, "Non-existent agent should return None"

    print("Single case test passed")

@pytest.mark.asyncio
async def test_chat():
    agent1 = SubAgent()
    tools = init_sub_agents([agent1])

    question = "Are you ready to do a task for me? If so, answer 'YES' and nothing else."
    result = await tools[2](question)
    assert result.lower() == "yes", "Chat logic failed"

    assert len(agent1.messages) == 3, "Agent should have 3 messages"
    assert type(agent1.messages[1]) == ChatMessageUser, "Second message should be a user message"
    assert type(agent1.messages[2]) == ChatMessageAssistant, "Third message should be an assistant message"
    assert agent1.messages[1].content == question, "User message should be the question"
    assert agent1.messages[2].content.lower() == "yes", "Assistant message should be 'yes'"

@pytest.mark.asyncio
async def test_run():
    agent1 = SubAgent(model="openai/gpt-4o", tools=[dummy_tool()])
    tools = init_sub_agents([agent1])

    await tools[1]("Start by saying exactly 'I accept the task'. Then use the dummy tool and then end the run immediately (stop reason is the output of the dummy tool).")

    tool_count = 0
    for msg in agent1.messages:
        if type(msg) == ChatMessageTool:
            if tool_count == 0:
                assert msg.text == "dummy123", "First tool call should be the dummy tool"
            elif tool_count == 1:
                assert msg.function == "_end_run", "Second tool call should be the end run tool"
                assert "dummy123" in msg.text, "End run tool should contain the output of the dummy tool"
            tool_count += 1

    print("Run test passed")

async def run_all_tests():
    print("Starting tests...")
    await test_sub_agent_class()
    await test_single_init()
    await test_default_init()
    await test_chat()
    await test_run()
    print("All tests passed!")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
