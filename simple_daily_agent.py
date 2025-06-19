"""
Simplified DailyAgent that doesn't require Pipecat Cloud API.
Just connects directly to a provided Daily room.
"""

import asyncio
import io
import json
import logging
import os
import wave
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.vad.silero import SileroVADAnalyzer, VADParams
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleDailyAgent:
    def __init__(self, room_url: str, room_token: str, bot_name: str = "User Bot"):
        """
        Simple Daily agent that connects directly to a provided room.
        No Pipecat Cloud API required.
        """
        self._room_url = room_url
        self._room_token = room_token
        self._bot_name = bot_name
        self._audio_data = None
        self._timeout = False
        self._transcript = []

    async def run_pipeline(self, user_prompt: str):
        """Run the agent pipeline with the given user prompt"""

        # Create transport - connect directly to the provided room
        self.transport = DailyTransport(
            self._room_url,
            self._room_token,
            self._bot_name,
            params=DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )

        # Initialize pipecat services
        self.stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"), audio_passthrough=True
        )
        self.tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"), voice_id="TX3LPaxmHKxFdv7VOQHJ"
        )
        self.llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

        # Define end conversation function
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
            await asyncio.sleep(3)
            logger.info("Stopping pipeline task")
            await self._pipeline_task.stop_when_done()
            logger.info("Pipeline task stopped")

        self.llm.register_function("end_conversation", handle_end_conversation)

        # Create audio buffer processor
        self._audiobuffer = AudioBufferProcessor(
            sample_rate=16000,
            num_channels=1,  # Mono for simplicity
            buffer_size=0,
            user_continuous_stream=False,
            enable_turn_audio=True,
        )

        @self._audiobuffer.event_handler("on_audio_data")
        async def on_audio_data(buffer, audio, sample_rate, num_channels):
            logger.debug(f"Received audio data: {len(audio)} bytes")
            await self.save_audio(audio, sample_rate, num_channels)

        logger.info(f"Connecting to room: {self._room_url}")

        # Create transcript processor
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
            logger.info("Pipeline idle timeout")
            self._timeout = True

        logger.info("Starting audio recording")
        await self._audiobuffer.start_recording()

        # Run the pipeline
        pipeline_runner = PipelineRunner(handle_sigint=False)
        await pipeline_runner.run(self._pipeline_task)

        logger.info("Stopping audio recording")
        await self._audiobuffer.stop_recording()

    async def save_audio(self, audio: bytes, sample_rate: int, num_channels: int):
        """Save audio data to buffer"""
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
        """Return transcript with roles flipped (since this agent plays the user)"""
        return [
            {
                "role": "assistant" if turn["role"] == "user" else "user",
                "content": turn["content"],
            }
            for turn in self._transcript
        ]
