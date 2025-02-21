import datetime
import pandas as pd
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field
from rich.prompt import Prompt

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.usage import Usage, UsageLimits

# ------------------------------------
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import os
import openai
import logfire

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

# ------------------------------------------

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

class FlightDetails(BaseModel):
    """Details of the most suitable flight."""

    flight_number: str
    price: int
    origin: str = Field(description='Three-letter airport code')
    destination: str = Field(description='Three-letter airport code')
    date: datetime.date

class NoFlightFound(BaseModel):
    """When no valid flight is found."""

@dataclass
class Deps:
    csv_file_path: str = "info.csv"

# This agent is responsible for controlling the flow of the conversation.
search_agent = Agent[Deps, FlightDetails | NoFlightFound](
    'groq:llama-3.3-70b-versatile',
    result_type=FlightDetails | NoFlightFound,  # type: ignore
    retries=4,
    system_prompt=(
        'Your job is to find the cheapest flight for the user on the given date. '
    ),
)

# This agent is responsible for extracting flight details from web page text.
extraction_agent = Agent(
    'groq:llama-3.3-70b-versatile',
    result_type=list[FlightDetails],
    system_prompt='Extract all the flight details from the given text.',
)

@search_agent.tool
async def extract_flights(ctx: RunContext[Deps]) -> list[FlightDetails]:
    """Get details of all person from CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(ctx.deps.csv_file_path)
        
        # Convert CSV data to FlightDetails objects
        flights = []
        for _, row in df.iterrows():
            if 'flight_number' in row and 'price' in row and 'origin' in row and 'destination' in row and 'date' in row:
                flight = FlightDetails(
                    flight_number=row['flight_number'],
                    price=row['price'],
                    origin=row['origin'],
                    destination=row['destination'],
                    date=datetime.datetime.strptime(row['date'], '%Y-%m-%d').date()
                )
                flights.append(flight)
                
        return flights
        
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return []

@search_agent.result_validator
async def validate_result(
    ctx: RunContext[Deps], result: FlightDetails | NoFlightFound
) -> FlightDetails | NoFlightFound:
    """Procedural validation that the flight meets the constraints."""
    if isinstance(result, NoFlightFound):
        return result

    errors: list[str] = []
    if result.origin != "SFO":
        errors.append(f'Flight should have origin SFO, not {result.origin}')
    if result.destination != "ANC":
        errors.append(f'Flight should have destination ANC, not {result.destination}')
    if result.date != datetime.date(2025, 1, 10):
        errors.append(f'Flight should be on 2025-01-10, not {result.date}')

    if errors:
        raise ModelRetry('\n'.join(errors))
    else:
        return result

class SeatPreference(BaseModel):
    row: int = Field(ge=1, le=30)
    seat: Literal['A', 'B', 'C', 'D', 'E', 'F']

class Failed(BaseModel):
    """Unable to extract a seat selection."""

# This agent is responsible for extracting the user's seat selection
seat_preference_agent = Agent[
    None, SeatPreference | Failed
](
    'groq:llama-3.3-70b-versatile',
    result_type=SeatPreference | Failed,  # type: ignore
    system_prompt=(
        "Extract the user's seat preference. "
        'Seats A and F are window seats. '
        'Row 1 is the front row and has extra leg room. '
        'Rows 14, and 20 also have extra leg room. '
    ),
)

# in reality this would be downloaded from a booking site,
# potentially using another agent to navigate the site
flights_web_page = """
1. Flight SFO-AK123
- Price: $350
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

2. Flight SFO-AK456
- Price: $370
- Origin: San Francisco International Airport (SFO)
- Destination: Fairbanks International Airport (FAI)
- Date: January 10, 2025

3. Flight SFO-AK789
- Price: $400
- Origin: San Francisco International Airport (SFO)
- Destination: Juneau International Airport (JNU)
- Date: January 20, 2025

4. Flight NYC-LA101
- Price: $250
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025

5. Flight CHI-MIA202
- Price: $200
- Origin: Chicago O'Hare International Airport (ORD)
- Destination: Miami International Airport (MIA)
- Date: January 12, 2025

6. Flight BOS-SEA303
- Price: $120
- Origin: Boston Logan International Airport (BOS)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 12, 2025

7. Flight DFW-DEN404
- Price: $150
- Origin: Dallas/Fort Worth International Airport (DFW)
- Destination: Denver International Airport (DEN)
- Date: January 10, 2025

8. Flight ATL-HOU505
- Price: $180
- Origin: Hartsfield-Jackson Atlanta International Airport (ATL)
- Destination: George Bush Intercontinental Airport (IAH)
- Date: January 10, 2025
"""

# restrict how many requests this app can make to the LLM
usage_limits = UsageLimits(request_limit=15)

async def main():
    deps = Deps()
    message_history: list[ModelMessage] | None = None
    usage: Usage = Usage()
    # run the agent until a satisfactory flight is found
    while True:
        result = await search_agent.run(
            f'Find me a flight from SFO to ANC on 2025-01-10',
            deps=deps,
            usage=usage,
            message_history=message_history,
            usage_limits=usage_limits,
        )
        if isinstance(result.data, NoFlightFound):
            print('No flight found')
            break
        else:
            flight = result.data
            print(f'Flight found: {flight}')
            answer = Prompt.ask(
                'Do you want to buy this flight, or keep searching? (buy/*search)',
                choices=['buy', 'search', ''],
                show_choices=False,
            )
            if answer == 'buy':
                seat = await find_seat(usage)
                await buy_tickets(flight, seat)
                break
            else:
                message_history = result.all_messages(
                    result_tool_return_content='Please suggest another flight'
                )

async def find_seat(usage: Usage) -> SeatPreference:
    message_history: list[ModelMessage] | None = None
    while True:
        answer = Prompt.ask('What seat would you like?')

        result = await seat_preference_agent.run(
            answer,
            message_history=message_history,
            usage=usage,
            usage_limits=usage_limits,
        )
        if isinstance(result.data, SeatPreference):
            return result.data
        else:
            print('Could not understand seat preference. Please try again.')
            message_history = result.all_messages()

async def buy_tickets(flight_details: FlightDetails, seat: SeatPreference):
    print(f'Purchasing flight {flight_details=!r} {seat=!r}...')

if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
