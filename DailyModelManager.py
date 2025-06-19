import asyncio
import datetime
import json
import logging
import os
import traceback

from CovalError import SimulationInvalidConfigError
from dotenv import load_dotenv
from evals.models import TestCase
from model_type_constants import DAILY
from models.DailyAgent import DailyAgent
from models.ModelManager import ModelManager
from utils.voice_utils import upload_to_s3

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyModelManager(ModelManager):
    def __init__(self, run_id):
        super().__init__(run_id)
        self.type = DAILY
        self.ready = False
        self.simulation_output_id = None
        self.transcripts = []
        self.agent = None

    def _start(self, config):
        """Initialize the Daily clients"""
        metadata = {
            "runId": self.run_id,
        }
        try:
            agent_name = config.get("agent_name")
            pipecat_api_key = config.get("pipecat_api_key")
            custom_data = config.get("custom_data")
            if custom_data is not None and custom_data != "":
                if not isinstance(custom_data, str):
                    raise SimulationInvalidConfigError("Custom data must be a string")
                if len(custom_data) > 1024 * 16:
                    raise SimulationInvalidConfigError(
                        "Custom data must be less than 16KB"
                    )
                try:
                    custom_data = json.loads(custom_data)
                except json.JSONDecodeError:
                    raise SimulationInvalidConfigError("Custom data must be valid JSON")
            self.agent = DailyAgent(
                agent_name,
                pipecat_api_key,
                coval_metadata=metadata,
                custom_data=custom_data,
            )

        except Exception as e:
            raise SimulationInvalidConfigError(f"Failed to start: {str(e)}")

    def _get_user_prompt(
        self,
        input_str: str,
        custom_prompt: str = None,
        language_instruction: str = None,
    ) -> str:
        return f"""        You are a USER talking to an assistant.
        You are explicitly ROLE-PLAYING as a USER. Under no circumstances do you act as an agent or assistant.

        {f"Additional context:\n{custom_prompt}" if custom_prompt else ""}

        Your objective is: {input_str}

        You must follow these rules:
        1. Do not offer assistance or ask what you can do.
        2. If the conversation has not started, you can act as the customer and introduce the objective.
        3. Do not volunteer information, only respond if asked.
        4. Provide personal details (like name, address, phone number) when asked, using information from the input or realistic fictional information if not provided.
        5. End the conversation by using the end_conversation function if:
            - You have completed the objective
            Never end the conversation if:
            - The agent is in the middle of asking a question
            - The agent is in the middle of a sentence
            - The agent has not finished their current thought
        6. Respond to the last message appropriately as a customer.
        7. If given alternative options, select one.
        8. Always let the agent complete their sentences and thoughts before responding.
        9. {language_instruction}
        This is the conversation thus far (you are the user):"""

    def run_simulation(self, test_case_id: str, simulation_output_id: str) -> bool:
        try:
            input_str = self.get_input_str_from_test_case_id(test_case_id)
            user_prompt = self._get_user_prompt(input_str)
            self.simulation_output_id = simulation_output_id
            print(f"Simulation output id: {self.simulation_output_id}")
            start_time = datetime.datetime.now()
            print(f"Start time: {start_time}")

            try:
                # Run the pipeline
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.agent.run_pipeline(user_prompt))
                loop.close()

                transcript = self.agent.transcript
                if self.agent.timeout:
                    transcript.append(
                        {
                            "role": "assistant",
                            "content": "error: Conversation ended due to assistant silence",
                        }
                    )
                # Get the conversation transcript
                transcript_str = json.dumps(self.agent.transcript)

            finally:
                print("Ending Daily client")

            if self.agent.audio_data:
                s3_key = upload_to_s3(self.agent.audio_data, self.simulation_output_id)
            else:
                print("No audio data to upload")
                s3_key = None

            self._populate_simulation_output(
                simulation_output_id=simulation_output_id,
                simulation_output_str=transcript_str,
                status="completed",
                start_time=start_time,
                end_time=datetime.datetime.now(),
                s3_key=s3_key,
                s3_bucket=os.getenv("AWS_S3_BUCKET_NAME"),
            )

        except Exception as e:
            traceback.print_exc()
            self._report_failed_simulation(e)
