#!/usr/bin/env python3
"""
Simple simulation runner between DailyAgent and bot.
Captures transcript and audio from both sides.
"""

import asyncio
import json
import logging
import os
import datetime
from typing import Dict, List
from dotenv import load_dotenv
import requests

# Import your existing classes
from simple_daily_agent import SimpleDailyAgent

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleSimulation:
    def __init__(self):
        self.transcript = []
        self.audio_data = None
        self.simulation_complete = False

    def create_daily_room(self, agent_name: str, api_key: str):
        """Initialize the Daily Cloud agent and set up the pipeline"""
        if not agent_name or not api_key:
            raise ValueError("AGENT_NAME and PIPECAT_API_KEY are required")

        logger.info(f"Creating Daily room for agent: {agent_name}")

        # Make HTTP request to start session
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {"createDailyRoom": True, "body": {}}

        try:
            response = requests.post(
                f"https://api.pipecat.daily.co/v1/public/{agent_name}/start",
                headers=headers,
                json=data,
                timeout=30,  # Add timeout
            )
            response.raise_for_status()
            response_data = response.json()

            room_url = response_data["dailyRoom"]
            room_token = response_data["dailyToken"]

            logger.info(f"✅ Daily room created: {room_url}")
            return room_url, room_token

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to create Daily room: {e}")
            raise

    async def run_simulation(
        self,
        user_objective: str = "I want to schedule a meeting",
        system_prompt: str = None,
        duration_minutes: int = 2,
    ):
        """
        Run a simple simulation between DailyAgent (user) and bot (assistant)
        """
        logger.info("Starting simulation...")

        # Validate required environment variables
        required_vars = [
            "AGENT_NAME",
            "PIPECAT_API_KEY",
            "OPENAI_API_KEY",
            "ELEVENLABS_API_KEY",
            "DEEPGRAM_API_KEY",
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

        # Create a simple Daily room for testing
        room_url, room_token = self.create_daily_room(
            agent_name=os.getenv("AGENT_NAME"),
            api_key=os.getenv("PIPECAT_API_KEY"),
        )

        # User prompt for DailyAgent (plays the user role)
        user_prompt = f"""You are a USER talking to an assistant.
        You are explicitly ROLE-PLAYING as a USER. Under no circumstances do you act as an agent or assistant.

        Your objective is: {user_objective}

        You must follow these rules:
        1. Do not offer assistance or ask what you can do.
        2. Start the conversation by introducing your objective.
        3. Only respond when asked, don't volunteer extra information.
        4. Provide realistic details when asked.
        5. End the conversation when your objective is complete.
        6. Be natural and conversational.
        """

        try:
            # Start both agents concurrently
            tasks = [
                self._run_user_agent(room_url, room_token, user_prompt),
            ]

            # Run simulation with timeout
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=duration_minutes * 60,
            )

        except asyncio.TimeoutError:
            logger.info("Simulation timed out")
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        finally:
            await self._save_results()

        return self.transcript, self.audio_data

    async def _run_user_agent(self, room_url: str, room_token: str, user_prompt: str):
        """Run the DailyAgent as the user"""
        try:
            logger.info("Starting user agent...")

            # Create SimpleDailyAgent instance
            agent = SimpleDailyAgent(
                room_url=room_url, room_token=room_token, bot_name="User Bot"
            )

            # Run the pipeline with user prompt
            await agent.run_pipeline(user_prompt)

            # Collect transcript from user side
            user_transcript = agent.transcript
            self.transcript.extend(
                [{"source": "user_agent", "messages": user_transcript}]
            )

            # Collect audio if available
            if agent.audio_data:
                self.audio_data = {"user": agent.audio_data}

        except Exception as e:
            logger.error(f"User agent error: {e}")

    async def _save_results(self):
        """Save simulation results to files"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save transcript
        transcript_file = f"simulation_transcript_{timestamp}.json"
        with open(transcript_file, "w") as f:
            json.dump(self.transcript, f, indent=2)
        logger.info(f"Transcript saved to: {transcript_file}")

        # Save audio if available
        if self.audio_data and self.audio_data.get("user"):
            audio_file = f"simulation_audio_{timestamp}.wav"
            with open(audio_file, "wb") as f:
                # Get the audio data from the BytesIO buffer
                audio_buffer = self.audio_data["user"]
                audio_buffer.seek(0)  # Reset to beginning
                f.write(audio_buffer.read())
            logger.info(f"Audio saved to: {audio_file}")
        else:
            logger.info("No audio data captured")


async def main():
    """Main entry point for the simulation"""
    simulation = SimpleSimulation()

    # Run simulation with custom parameters
    transcript, audio = await simulation.run_simulation(
        user_objective="I want to schedule a job interview for a software engineering position",
        duration_minutes=3,  # Short simulation
    )

    print("\n" + "=" * 50)
    print("SIMULATION COMPLETE")
    print("=" * 50)
    print(f"Transcript entries: {len(transcript)}")
    print(f"Audio captured: {'Yes' if audio else 'No'}")
    print("Check the generated files for full results.")


if __name__ == "__main__":
    # Simple CLI for running simulations
    import argparse

    parser = argparse.ArgumentParser(description="Run Daily Agent simulation")
    parser.add_argument(
        "--objective",
        default="I want to schedule a meeting",
        help="User objective for the simulation",
    )
    parser.add_argument(
        "--duration", type=int, default=5, help="Simulation duration in minutes"
    )
    parser.add_argument(
        "--system-prompt", help="Custom system prompt for the assistant"
    )

    args = parser.parse_args()

    # Run the simulation
    asyncio.run(main())
