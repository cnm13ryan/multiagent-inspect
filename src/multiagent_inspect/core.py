from inspect_ai.tool import tool, Tool, ToolDef
from inspect_ai.model._chat_message import ChatMessageSystem, ChatMessageUser
from inspect_ai.util import store
from inspect_ai.model._model import get_model
from inspect_ai.model._call_tools import call_tools

from inspect_ai.solver._basic_agent import DEFAULT_SYSTEM_MESSAGE 

class SubAgent():
    _id_counter = 1

    def __init__(self, 
                 agent_id: str = None,
                 max_steps: int = 10,
                 model: str = "openai/gpt-4o-mini-2024-07-18-free",
                 public_description: str = "",
                 internal_description: str = "",
                 tools: list[Tool] = None,
                 metadata: dict = None):
        # Auto-generate ID if none provided
        if agent_id is None:
            agent_id = f"{SubAgent._id_counter:03d}"
            SubAgent._id_counter += 1
            
        sys_msg = DEFAULT_SYSTEM_MESSAGE.format(submit='_end_run')
        sys_msg += f"\n\nOnly attempt tasks which you think you can do with your limited set of tools. After running a task, you might be asked questions about it. Only answer things that you know that you have done.\n\n{internal_description}"
        self.agent_id = agent_id
        self.max_steps = max_steps
        self.public_description = public_description
        self.model = model
        self.tools = tools
        self.metadata = metadata
        self.messages = [ChatMessageSystem(content=sys_msg)]

    def __str__(self):
        msg = (
            f"ID: {self.agent_id}\n"
            f"Model: {self.model}\n"
            f"Description: {self.public_description}\n"
            f"Max Steps: {self.max_steps}\n"
        )
        if self.tools:
            tool_names = [ToolDef(t).name for t in self.tools]
            msg += f"Tools: {tool_names}\n"

        return msg

@tool
def _end_run() -> Tool:
    async def execute(stop_reason: str):
        """Use this tool only when you want to end the run. End the run when you have either fulfilled your instructions or you are stuck and don't know what to do.
        Args:
            stop_reason (str): Reason for stopping the run.
        """
        return f"Run ended with reason: {stop_reason}"
    return execute

async def _get_agent(sub_agent_id: str = None) -> SubAgent:
    sub_agents = store().get("sub_agents", {})

    if sub_agent_id is None:
        sub_agent_id = list(sub_agents.keys())[0]

    if sub_agent_id not in sub_agents:
        return None
    return sub_agents[sub_agent_id]

async def _update_store(sub_agent: SubAgent):
    sub_agents = store().get("sub_agents", {})
    sub_agents[sub_agent.agent_id] = sub_agent
    store().set("sub_agents", sub_agents)

async def _run_logic(sub_agent: SubAgent, instructions: str):
    sub_agent.messages.append(ChatMessageUser(content=instructions))

    for steps in range(sub_agent.max_steps):
        output = await get_model(sub_agent.model).generate(
            input=sub_agent.messages, tools=sub_agent.tools + [_end_run()]
        )
        sub_agent.messages.append(output.message)

        if output.message.tool_calls:
            tool_results = await call_tools(
                output.message, sub_agent.tools + [_end_run()]
            )
            sub_agent.messages.extend(tool_results)

            if any(tool_result.function == "_end_run" for tool_result in tool_results):
                break

    await _update_store(sub_agent)
    return f"Sub agent ran for {steps} steps. You can now ask it questions."

async def _chat_logic(sub_agent: SubAgent, question: str):
    sub_agent.messages.append(ChatMessageUser(content=question))

    output = await get_model(sub_agent.model).generate(
        input=sub_agent.messages
    )
    sub_agent.messages.append(output.message)

    await _update_store(sub_agent)
    return output.message.text

def init_sub_agents(sub_agents: list[SubAgent]):
    if len(sub_agents) < 1:
        return []

    store().set("sub_agents", {sub_agent.agent_id: sub_agent for sub_agent in sub_agents})

    if len(sub_agents) > 1:
        return [sub_agent_specs(), run_sub_agent(), chat_with_sub_agent()]
    elif len(sub_agents) == 1:
        return [sub_agent_specs(single_sub_agent=True), run_sub_agent(single_sub_agent=True), chat_with_sub_agent(single_sub_agent=True)]

@tool
def sub_agent_specs(single_sub_agent: bool = False) -> Tool:
    if single_sub_agent:
        async def execute():
            """Show the specifications of the sub agent.

            Use this tool to learn what the sub agent can be used for.

            Returns:
                str: Specification of the sub agent.
            """
            agent = await _get_agent()
            return str(agent)
    else:
        async def execute():
            """Lists all available sub agents with their specifications.

            Use this tool to find the right sub agent to use for the task at hand.

            Returns:
                str: Specifications of the sub agents.
            """
            sub_agents = store().get("sub_agents", [])
            return "\n".join([str(sub_agent) for sub_agent in sub_agents])
    return execute

@tool
def run_sub_agent(single_sub_agent: bool = False) -> Tool:
    if single_sub_agent:
        async def execute(instructions: str):
            """Runs a sub agent. Note you will not know what the sub agent did. To know that, you need to chat with it.

            Args:
                instructions (str): Instructions for the sub agent.
            """
            agent = await _get_agent()
            return await _run_logic(agent, instructions)
    else:
        async def execute(sub_agent_id: str, instructions: str):
            """Runs a sub agent. Note you will not know what the sub agent did. To know that, you need to chat with it.

            Args:
                sub_agent_id (str): ID of the sub agent to run.
                instructions (str): Instructions for the sub agent.
            """
            agent = await _get_agent(sub_agent_id)
            return await _run_logic(agent, instructions)
    return execute

@tool
def chat_with_sub_agent(single_sub_agent: bool = False) -> Tool:
    if single_sub_agent:
        async def execute(question: str):
            """Chats with a sub agent that previously was run with some instructions.

            Args:
                question (str): Question to ask the sub agent.

            Returns:
                str: Response from the sub agent.
            """
            agent = await _get_agent()
            return await _chat_logic(agent, question)
    else:
        async def execute(sub_agent_id: str, question: str):
            """Chats with a sub agent that previously was run with some instructions.

            Args:
                sub_agent_id (str): ID of the sub agent to chat with.
                question (str): Question to ask the sub agent.

            Returns:
                str: Response from the sub agent.
            """
            agent = await _get_agent(sub_agent_id)
            return await _chat_logic(agent, question)
    return execute
