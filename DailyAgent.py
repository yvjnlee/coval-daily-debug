from pathlib import Path
import sys
import json
import traceback
from dotenv import load_dotenv
from typing import Dict, List
import time
import logging
import os
import queue
import io
import wave
import requests
from openai import OpenAI, OpenAIError
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
from daily import Daily
from utils.voice_utils import upload_to_s3
from CovalError import *
import datetime
import asyncio

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.vad.silero import SileroVAD
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyAgent:
    def __init__(
        self,
        agent_name: str,
        api_key: str,
        coval_metadata: Dict[str, str] = {},
        custom_data: Dict[str, str] = {},
    ):
        # Get API keys from environment variables
        self._audio_data = None
        self._timeout = False
        self._transcript = []

        self._room_url, self._room_token = self._start_pipecat_cloud_agent(
            agent_name, api_key, coval_metadata, custom_data
        )

    @staticmethod
    def _start_pipecat_cloud_agent(
        agent_name: str,
        api_key: str,
        coval_metadata: Dict[str, str],
        custom_data: Dict[str, str],
    ):
        """Initialize the Daily Cloud agent and set up the pipeline"""
        # Make HTTP request to start session
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {"createDailyRoom": True, "body": {"coval": coval_metadata}}
        if custom_data:  # Ignore empty strings and empty JSON objects
            data["body"]["customData"] = custom_data
        response = requests.post(
            f"https://api.pipecat.daily.co/v1/public/{agent_name}/start",
            headers=headers,
            json=data,
        )
        response.raise_for_status()
        response_data = response.json()

        # Get Daily room URL and token from response
        return (response_data["dailyRoom"], response_data["dailyToken"])

    async def run_pipeline(self, user_prompt):
        # Create transport
        self.transport = DailyTransport(
            self._room_url,
            self._room_token,
            "Coval User Bot",
            DailyParams(
                audio_in_enabled=True,
                audio_in_sample_rate=16000,
                audio_in_channels=1,
                audio_out_enabled=True,
                audio_out_sample_rate=16000,
                audio_out_channels=1,
                audio_out_is_live=False,
            ),
        )

        # Initialize pipecat services
        self.stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"), audio_passthrough=True
        )
        self.vad = SileroVAD()
        self.tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"), voice_id="TX3LPaxmHKxFdv7VOQHJ"
        )
        self.llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        end_conversation_function = FunctionSchema(
            name="end_conversation",
            description="End the conversation when conversation with the agent is over.",
            properties={
                "response": {
                    "type": "string",
                    "description": "The final response to end the conversation",
                }
            },
            required=["response"],
        )
        tools = ToolsSchema(standard_tools=[end_conversation_function])

        # Create LLM context
        messages = [{"role": "developer", "content": user_prompt}]
        context = OpenAILLMContext(messages, tools=tools)
        self.context_aggregator = self.llm.create_context_aggregator(context)

        async def handle_end_conversation(
            function_name, tool_call_id, args, llm, context, result_callback
        ):
            logger.info(f"Ending conversation with response: {args}")
            await self.llm.push_frame(TTSSpeakFrame(args["response"]))
            await asyncio.sleep(5)
            logger.info("Stopping pipeline task")
            await self._pipeline_task.stop_when_done()
            logger.info("Pipeline task stopped")

        self.llm.register_function("end_conversation", handle_end_conversation)

        # Create an audio buffer processor
        self._audiobuffer = AudioBufferProcessor(
            sample_rate=16000,  # Optional: desired output sample rate
            num_channels=2,  # 1 for mono, 2 for stereo
            buffer_size=0,  # Size in bytes to trigger buffer callbacks
            user_continuous_stream=False,
            enable_turn_audio=True,
        )

        @self._audiobuffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            logger.info(f"Received audio data: {len(audio)} bytes")
            logger.info(f"Format {sample_rate} {num_channels}")
            await self.save_audio(audio, sample_rate, num_channels)

        print(f"Joining room link: {self._room_url}?t={self._room_token}")

        transcript = TranscriptProcessor()

        @transcript.event_handler("on_transcript_update")
        async def handle_update(processor, frame):
            for msg in frame.messages:
                if len(self._transcript) > 0:
                    latest_msg = self._transcript[-1]
                    if latest_msg["role"] == msg.role:
                        latest_msg["content"] += " " + msg.content
                        return
                self._transcript.append({"role": msg.role, "content": msg.content})

        # Create pipeline
        self.pipeline = Pipeline(
            [
                self.transport.input(),
                self.stt,
                transcript.user(),
                # self.vad,
                self.context_aggregator.user(),
                self.llm,
                self.tts,
                self.transport.output(),
                transcript.assistant(),
                self._audiobuffer,
                self.context_aggregator.assistant(),
            ]
        )

        # Create pipeline task
        self._pipeline_task = PipelineTask(
            self.pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000,
                heartbeats_period_secs=2.0,
            ),
            idle_timeout_secs=30,
            cancel_on_idle_timeout=True,
        )

        @self._pipeline_task.event_handler("on_idle_timeout")
        async def on_idle_timeout(task):
            self._timeout = True

        logger.info("Starting recording")
        await self._audiobuffer.start_recording()

        # Create pipeline runner
        pipeline_runner = PipelineRunner(handle_sigint=False)
        await pipeline_runner.run(self._pipeline_task)
        logger.info("Stopping recording")
        await self._audiobuffer.stop_recording()

    async def save_audio(self, audio: bytes, sample_rate: int, num_channels: int):
        if len(audio) > 0:
            self._audio_data = io.BytesIO()
            with wave.open(self._audio_data, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            self._audio_data.seek(0)

    @property
    def audio_data(self):
        return self._audio_data

    @property
    def timeout(self):
        return self._timeout

    @property
    def transcript(self):
        return [
            {
                "role": "assistant" if turn["role"] == "user" else "user",
                "content": turn["content"],
            }
            for turn in self._transcript
        ]
