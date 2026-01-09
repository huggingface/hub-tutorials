# requirements = ["python-dotenv", "openai-agents[litellm]"]

from agents import function_tool, RunContextWrapper, Agent
from agents.extensions.models.litellm_model import LitellmModel
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

from pydantic import BaseModel

import os 
import getpass
from dotenv import load_dotenv

load_dotenv("../../.env")

if "OPENAI_API_KEY" not in os.environ or os.environ["OPENAI_API_KEY"] == "":
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

if "HF_TOKEN" not in os.environ or os.environ["HF_TOKEN"] == "":
    os.environ["HF_TOKEN"] = getpass.getpass("Enter your HuggingFace token: ")

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings:.*",
)

### CONTEXT
class AirlineAgentContext(BaseModel):
    passenger_name: str | None = None
    confirmation_number: str | None = None
    seat_number: str | None = None
    flight_number: str | None = None

### TOOLS
@function_tool(
    name_override="faq_lookup_tool", description_override="Lookup frequently asked questions."
)
async def faq_lookup_tool(question: str) -> str:
    question_lower = question.lower()
    if any(
        keyword in question_lower
        for keyword in ["bag", "baggage", "luggage", "carry-on", "hand luggage", "hand carry"]
    ):
        return (
            "You are allowed to bring one bag on the plane. "
            "It must be under 50 pounds and 22 inches x 14 inches x 9 inches."
        )
    elif any(keyword in question_lower for keyword in ["seat", "seats", "seating", "plane"]):
        return (
            "There are 120 seats on the plane. "
            "There are 22 business class seats and 98 economy seats. "
            "Exit rows are rows 4 and 16. "
            "Rows 5-8 are Economy Plus, with extra legroom. "
        )
    elif any(
        keyword in question_lower
        for keyword in ["wifi", "internet", "wireless", "connectivity", "network", "online"]
    ):
        return "We have free wifi on the plane, join Airline-Wifi"
    return "I'm sorry, I don't know the answer to that question."


@function_tool
async def update_seat(
    ctx: RunContextWrapper[AirlineAgentContext], confirmation_number: str, new_seat: str
) -> str:
    """
    Update the seat for a given confirmation number.

    Args:
        confirmation_number: The confirmation number for the flight.
        new_seat: The new seat to update to.
    """
    # Update the context based on the customer's input
    ctx.context.confirmation_number = confirmation_number
    ctx.context.seat_number = new_seat
    # Ensure that the flight number has been set by the incoming handoff
    assert ctx.context.flight_number is not None, "Flight number is required"
    return f"Updated seat to {new_seat} for confirmation number {confirmation_number}"

### AGENTS

model = LitellmModel(
    model="huggingface/nscale/Qwen/Qwen3-8B",
    api_key=os.environ["HF_TOKEN"],
)

faq_agent = Agent[AirlineAgentContext](
    name="FAQ Agent",
    handoff_description="A helpful agent that can answer questions about the airline.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Identify the last question asked by the customer.
    2. Use the faq lookup tool to answer the question. Do not rely on your own knowledge.
    3. If you cannot answer the question, transfer back to the triage agent.""",
    tools=[faq_lookup_tool],
    model=model,
)

seat_booking_agent = Agent[AirlineAgentContext](
    name="Seat Booking Agent",
    handoff_description="A helpful agent that can update a seat on a flight.",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are a seat booking agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    # Routine
    1. Ask for their confirmation number.
    2. Ask the customer what their desired seat number is (the user already know which seats are available).
    3. Use the update seat tool to update the seat on the flight.
    If the customer asks a question that is not related to the routine, transfer back to the triage agent. """,
    tools=[update_seat],
    model=model,
)

triage_agent = Agent[AirlineAgentContext](
    name="Triage Agent",
    handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX} "
        "You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents."
    ),
    handoffs=[
        faq_agent,
        seat_booking_agent
    ],
    model=model,
)

faq_agent.handoffs.append(triage_agent)
seat_booking_agent.handoffs.append(triage_agent)

async def main() -> None:
    from agents import run_demo_loop
    agent: Agent[AirlineAgentContext] = triage_agent
    context = AirlineAgentContext(
        flight_number="FLT-123",
        seat_number="A12",
        passenger_name="John Doe",
    )
    await run_demo_loop(agent, context=context)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
