# yright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""
Ezra Interview Bot Implementation

This module implements a voice-first chatbot using OpenAI's Realtime API model,
integrated with Daily for real-time audio/video transport, Pipecat for pipeline orchestration,
and Firebase for audio storage. The pipeline handles:
  - Audio/video capture and streaming (Daily)
  - Voice activity detection (SileroVADAnalyzer)
  - Speech-to-text and text-to-speech (OpenAI Realtime)
  - Animated avatar frames and custom events (RTVI)
  - Audio recording and upload (Firebase Storage)
  - Conversation lifecycle management and graceful shut down

Anyone new to this code should first read the high-level flow in `main()`, then inspect
`ConversationManager` for conversation control logic, and finally review event handlers
that wire transport, VAD, transcript updates, and storage together.
"""

import os
import io
import sys
import uuid
import json
import wave
import base64
import asyncio
import datetime

import aiohttp
import aiofiles
from dotenv import load_dotenv
from loguru import logger
from PIL import Image

import firebase_admin
from firebase_admin import credentials, storage

# Pipecat imports for pipeline construction and processing
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.frames.frames import (
    Frame,
    SpriteFrame,
    OutputImageRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMMessagesFrame,
    EndTaskFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.transcript_processor import (
    TranscriptProcessor,
    TranscriptionMessage,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import (
    RTVIConfig,
    RTVIProcessor,
    RTVIObserver,
    RTVIServerMessageFrame,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.openai_realtime_beta import (
    OpenAIRealtimeBetaLLMService,
    SessionProperties,
    InputAudioTranscription,
)
from pipecat.services.deepgram import DeepgramSTTService, LiveOptions
from pipecatcloud.agent import DailySessionArguments

# Load environment variables from .env file
load_dotenv(override=True)

# -----------------------------------------------------------------------------------
# Global Audio Buffer Setup
# -----------------------------------------------------------------------------------
# We record mono audio at 44.1kHz for voice analysis and storage.
audiobuffer = AudioBufferProcessor(
    sample_rate=44100,
    num_channels=1,
    buffer_size=0,  # 0 => dynamic buffer, optimized for real-time
)


# -----------------------------------------------------------------------------------
# Conversation Lifecycle Manager
# -----------------------------------------------------------------------------------
class ConversationManager:
    """
    Controls conversation shutdown, final analysis, and audio saving.

    Inject dependencies (PipelineTask, RTVIProcessor) after initialization,
    then expose async methods used by our function-calling interface.
    """

    def __init__(
        self,
        llm,
        audiobuffer,
        program_type,
        user_email,
        job_id,
        active_session_id,
        job_interview: bool,
    ):
        # LLM service used for streaming conversation
        self._llm = llm
        # Audio buffer for saving recorded audio
        self._audiobuffer = audiobuffer
        self._program_type = program_type
        self._user_email = user_email
        self._job_id = job_id
        self._active_session_id = active_session_id
        self._job_interview = job_interview
        # Firebase bucket for uploading audio files
        self._bucket = storage.bucket()
        self.shutdown_initiated = False
        # Placeholders for cross-dependencies
        self._rtvi = None
        self._task = None

    def set_dependencies(self, task: PipelineTask, rtvi: RTVIProcessor):
        """Inject pipeline task and RTVI processor after creation."""
        self._task = task
        self._rtvi = rtvi

    async def end_conversation_function(self, params):
        """
        Registered as an LLM function: marks shutdown, sends goodbye,
        and triggers audio stop or final analysis based on context.
        """
        logger.info("LLM requested conversation end.")
        self.shutdown_initiated = True

        # Choose farewell message based on program type
        if self._program_type == "programInterviewForJob":
            text = "End politely, without repeating previous statements."
        else:
            text = "Say 'Looking forward to finding the perfect fit for the role', then end."

        # Return via function-calling callback
        await params.result_callback({"goodbye_message": text})

        # Stop recording if in job interview mode
        if self._job_interview:
            await self._audiobuffer.stop_recording()
        else:
            await self._perform_final_analysis()

    async def finalize_shutdown(self):
        """
        After last assistant transcript, send custom RTVI event,
        wait, then queue EndTaskFrame to terminate pipeline.
        """
        if not self.shutdown_initiated:
            return
        logger.info("Finalizing shutdown sequence...")

        # Send custom 'interview complete' signal to client UI
        signal_name = (
            "candidateInterviewComplete"
            if self._program_type == "programInterviewForJob"
            else "hmInterviewComplete"
        )
        frame = RTVIServerMessageFrame(
            data={"type": "custom-event", "payload": {"signal": signal_name}}
        )
        await self._rtvi.push_frame(frame)

        # Grace period for client to process
        await asyncio.sleep(15)
        logger.info("Grace period over, ending task.")
        await self._task.queue_frames([EndTaskFrame()])
        self.shutdown_initiated = False

    async def handle_saved_audio(self, filename: str):
        """
        Called when audio blob saved: kicks off final analysis step.
        """
        logger.info(f"Audio saved: {filename}. Starting analysis.")
        await self._perform_final_analysis(filename)

    async def _perform_final_analysis(self, filename: str = None):
        """
        POST conversation metadata to our Node.js server endpoint.
        """
        url = os.getenv("SERVER_URL") + "/postConversationAnalysis"
        payload = {
            "fileName": filename,
            "userEmail": self._user_email,
            "programType": self._program_type,
            "activeSessionId": self._active_session_id,
            "jobId": self._job_id,
            "jobInterview": self._job_interview,
            "authKey": os.getenv("EZRA_SERVER_AUTH_KEY"),
        }
        logger.debug(f"Sending analysis payload: {payload}")
        try:
            async with aiohttp.ClientSession() as session:
                resp = await session.post(url, json=payload)
                if resp.status == 200:
                    logger.info("Analysis POST succeeded.")
                else:
                    text = await resp.text()
                    logger.error(f"Analysis POST failed {resp.status}: {text}")
        except Exception:
            logger.exception("Error posting analysis")


# -----------------------------------------------------------------------------------
# LLM Function & Tool Definitions
# -----------------------------------------------------------------------------------
def define_tools(program_type: str) -> ToolsSchema:
    """
    Register LLM-callable functions based on program type.
    Currently supports single 'end_conversation' call.
    """
    end_tool = FunctionSchema(
        name="end_conversation",
        description="Ends conversation gracefully, saves audio, and triggers analysis.",
        properties={},
        required=[],
    )
    return ToolsSchema(standard_tools=[end_tool])


def define_function_instructions(program_type: str) -> str:
    """
    Instructions appended to system prompt for LLM function-calling.
    """
    if program_type == "programInterviewForJob":
        return "When interview is over, call function 'end_conversation'."
    elif program_type == "programHMInterview":
        return "When conversation is over, call function 'end_conversation'."
    return ""


# -----------------------------------------------------------------------------------
# Firebase Initialization
# -----------------------------------------------------------------------------------
try:
    b64 = os.getenv("FIREBASE_CREDENTIALS_BASE64")
    if not b64:
        raise RuntimeError("Missing FIREBASE_CREDENTIALS_BASE64")
    cred_json = base64.b64decode(b64).decode()
    cred_dict = json.loads(cred_json)

    # Remove any existing Firebase apps to start fresh
    for app in list(firebase_admin._apps.values()):
        firebase_admin.delete_app(app)
        logger.debug("Removed existing Firebase app.")

    # Initialize new Firebase app with storage bucket
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(
        cred, {"storageBucket": "ezraailabs.firebasestorage.app"}
    )
    storage.bucket()
    logger.info("✅ Firebase initialized.")
except Exception:
    logger.exception("Firebase init failed.")
    raise


# -----------------------------------------------------------------------------------
# Main Bot Execution
# -----------------------------------------------------------------------------------
async def main(
    room_url: str,
    token: str,
    system_prompt: str = None,
    user_email: str = None,
    job_id: str = None,
    job_interview: bool = False,
    active_session_id: str = None,
    program_type: str = None,
):
    """
    Bootstraps and runs the full chatbot pipeline:
      1. DailyTransport for real-time media
      2. LLM streaming via OpenAIRealtimeBeta
      3. Transcript & VAD processing
      4. RTVI event handling and avatar frames
      5. Audio recording & upload
      6. Conversation management and shutdown
    """
    logger.debug("Launching bot in Daily room: {}", room_url)

    # Compose system prompt with function instructions if not provided
    instructions = define_function_instructions(program_type)
    if not system_prompt:
        system_prompt = (
            "You are Ezra, a helpful career coach. Assist the user with their goals. "
            + instructions
        )

    # 1) Configure Daily transport for VAD and transcription
    transport = DailyTransport(
        room_url,
        token,
        "Chatbot",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            transcription_enabled=True,
        ),
    )

    # 2) Set up LLM streaming (Realtime API)
    llm = OpenAIRealtimeBetaLLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        session_properties=SessionProperties(
            model="gpt-4o-mini",
            voice="shimmer",
            instructions=system_prompt,
            input_audio_transcription=InputAudioTranscription(
                model="gpt-4o-transcribe", language="en", prompt=system_prompt
            ),
        ),
    )

    # 3) Build context aggregator with function schema
    context = OpenAILLMContext(
        initial_msgs=[{"role": "user", "content": "Hello!"}],
        tools=define_tools(program_type),
    )
    context_agg = llm.create_context_aggregator(context)

    # 4) Transcription processor logs each incoming/outgoing text
    transcript = TranscriptProcessor()

    # 5) RTVI for avatar animation and custom events
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))
    manager = ConversationManager(
        llm,
        audiobuffer,
        program_type,
        user_email,
        job_id,
        active_session_id,
        job_interview,
    )
    llm.register_function("end_conversation", manager.end_conversation_function)

    # 6) Assemble pipeline stages
    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            context_agg.user(),
            llm,
            transcript.user(),
            transport.output(),
            audiobuffer,
            transcript.assistant(),
            context_agg.assistant(),
        ]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True, enable_metrics=True, enable_usage_metrics=True
        ),
        observers=[RTVIObserver(rtvi)],
    )
    manager.set_dependencies(task, rtvi)

    async def save_audio(
        audio: bytes, sample_rate: int, num_channels: int, interview: bool
    ):
        """
        Helper to write WAV to memory, upload to Firebase,
        and return a filename for later analysis.
        """
        if not interview or not audio:
            return None
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        uid = uuid.uuid4().hex[:8]
        fname = f"conversation_{ts}_{uid}.wav"
        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            wav_bytes = buf.getvalue()
        blob = storage.bucket().blob(f"audio_recordings/{fname}")
        try:
            await asyncio.to_thread(
                lambda: blob.upload_from_string(wav_bytes, content_type="audio/wav")
            )
            logger.info(f"Uploaded audio as {fname}")
            return fname
        except Exception:
            logger.exception("Audio upload failed.")
            return None

    # -----------------------------------------------------------------------------------
    # Event Handlers: wire transport, VAD, transcript -> pipeline & manager
    # -----------------------------------------------------------------------------------
    @transport.event_handler("on_first_participant_joined")
    async def handle_join(transport, participant):
        """
        Start capturing and recording when the first user joins.
        """
        logger.info(f"Participant joined: {participant['id']}")
        await transport.capture_participant_transcription(participant["id"])
        if job_interview:
            await audiobuffer.start_recording()

    @transport.event_handler("on_participant_left")
    async def handle_leave(transport, participant, reason):
        """
        Stop recording and cancel task when user leaves.
        """
        logger.info(f"Participant left: {participant['id']}, reason={reason}")
        if job_interview:
            await audiobuffer.stop_recording()
        await task.cancel()
        await transport.cleanup()

    @audiobuffer.event_handler("on_audio_data")
    async def handle_audio_data(buffer, audio, sr, nch):
        """
        Called for each audio chunk: save and notify manager.
        """
        logger.debug(f"Audio chunk: {len(audio)} bytes")
        fname = await save_audio(audio, sr, nch, job_interview)
        if fname:
            await manager.handle_saved_audio(fname)

    @rtvi.event_handler("on_client_ready")
    async def handle_rtvi_ready(rtvi_obj):
        """Notify client UI that bot is ready."""
        await rtvi_obj.set_bot_ready()

    @transport.event_handler("on_client_connected")
    async def handle_client_connected(transport, client):
        """Once client connects, inject initial user context frame."""
        logger.info("Client connected")
        frame = context_agg.user().get_context_frame()
        await task.queue_frames([frame])

    @transcript.event_handler("on_transcript_update")
    async def handle_transcript_update(proc, frame):
        """
        Logs transcript lines and, if shutdown pending,
        finalizes on assistant's last message.
        Also forwards each message to our Node server.
        """
        for msg in frame.messages:
            if isinstance(msg, TranscriptionMessage):
                ts = f"[{msg.timestamp}] " if msg.timestamp else ""
                line = f"{ts}{msg.role}: {msg.content}"
                logger.info(f"Transcript: {line}")

                # After assistant finishes speaking, trigger shutdown steps
                if manager.shutdown_initiated and msg.role == "assistant":
                    await manager.finalize_shutdown()

                # POST each snippet to our backend for live display/storage
                try:
                    url = os.getenv("SERVER_URL") + "/sendTranscriptFromBot"
                    payload = {
                        "userEmail": user_email,
                        "activeSessionId": active_session_id,
                        "programType": program_type,
                        "userReceiver": msg.role != "user",
                        "transcript": msg.content,
                        "jobId": job_id,
                        "authKey": os.getenv("EZRA_SERVER_AUTH_KEY"),
                    }
                    async with aiohttp.ClientSession() as sess:
                        resp = await sess.post(url, json=payload)
                        if resp.status != 200:
                            text = await resp.text()
                            logger.error(
                                f"Transcript POST failed: {resp.status}, {text}"
                            )
                except Exception:
                    logger.exception("Error posting transcript.")

    # -----------------------------------------------------------------------------------
    # Run the pipeline
    # -----------------------------------------------------------------------------------
    runner = PipelineRunner()
    try:
        await runner.run(task)
    finally:
        await transport.cleanup()
        await task.cleanup()


# -----------------------------------------------------------------------------------
# FastAPI Entry Point
# -----------------------------------------------------------------------------------
async def bot(args: DailySessionArguments):
    """
    Adapter for FastAPI route: unpacks session args and calls main().
    """
    logger.info(f"Initializing bot for room {args.room_url}")
    body = args.body or {}
    await main(
        room_url=args.room_url,
        token=args.token,
        system_prompt=body.get("system_prompt"),
        user_email=body.get("user_email"),
        job_id=body.get("job_id"),
        job_interview=body.get("job_interview", False),
        active_session_id=body.get("active_session_id"),
        program_type=body.get("program_type"),
    )


# -----------------------------------------------------------------------------------
# Local Development CLI
# -----------------------------------------------------------------------------------
async def local_main(
    room_url=None,
    token=None,
    system_prompt=None,
    user_email=None,
    job_id=None,
    job_interview=False,
    active_session_id=None,
    program_type=None,
):
    """Helper entry for local testing: auto-configures a Daily room if needed."""
    from local_runner import configure  # dynamic import for dev

    async with aiohttp.ClientSession() as session:
        if not room_url or not token:
            room_url, token = await configure(session)
        logger.warning(f"Connect your client at: {room_url}")
        await main(
            room_url,
            token,
            system_prompt,
            user_email,
            job_id,
            job_interview,
            active_session_id,
            program_type,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Daily bot process CLI")
    parser.add_argument("-u", "--url", help="Daily room URL")
    parser.add_argument("-t", "--token", help="Daily room token")
    parser.add_argument("--system_prompt", help="Override system prompt")
    parser.add_argument("--user_email", help="User email for analytics")
    parser.add_argument("--job_id", help="Job ID for interviews")
    parser.add_argument(
        "--job_interview", action="store_true", help="Enable audio recording"
    )
    parser.add_argument("--active_session_id", help="Session ID for logging")
    parser.add_argument("--program_type", help="Interview or HM conversation type")
    args = parser.parse_args()

    # Decide CLI vs. auto-configured dev mode
    if args.url and args.token:
        asyncio.run(
            local_main(
                args.url,
                args.token,
                args.system_prompt,
                args.user_email,
                args.job_id,
                args.job_interview,
                args.active_session_id,
                args.program_type,
            )
        )
    else:
        asyncio.run(
            local_main(
                system_prompt=args.system_prompt,
                user_email=args.user_email,
                job_id=args.job_id,
                job_interview=args.job_interview,
                active_session_id=args.active_session_id,
                program_type=args.program_type,
            )
        )
