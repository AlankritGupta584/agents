import logging
from dotenv import load_dotenv

from livekit_plugins.filler_guard import FillerGuard 


from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
# from livekit.plugins import noise_cancellation  # optional Krisp BVC

logger = logging.getLogger("basic-agent")
load_dotenv()


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly. You interact via voice. "
                "Keep responses concise and to the point. "
                "Do not use emojis, asterisks, markdown, or special characters. "
                "Be curious, friendly, with a light sense of humor. "
                "Speak English to the user."
            ),
            # IMPORTANT: we will manually interrupt via FillerGuard
            allow_interruptions=False,
        )

    async def on_enter(self):
        # Kick things off with a greeting
        self.session.generate_reply()

    @function_tool
    async def lookup_weather(
        self, context: RunContext, location: str, latitude: str, longitude: str
    ):
        """Called when the user asks for weather information.

        Args:
            location: City or region the user asked about
            latitude: Estimated latitude (do not ask user)
            longitude: Estimated longitude (do not ask user)
        """
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."


def prewarm(proc: JobProcess):
    # Load VAD once per worker process
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        # Ears
        stt="assemblyai/universal-streaming:en",
        # Brain
        llm="openai/gpt-4.1-mini",
        # Voice
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        # Turn detection + VAD
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],

        # Preemptive generation allowed
        preemptive_generation=True,

        # If you keep these on, SDK can auto-resume on noise-only false interruptions.
        # FillerGuard still controls "real" interrupts programmatically.
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

    # Collect usage metrics
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # ---- Start the session ----
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # noise_cancellation=noise_cancellation.BVC(),  # optional
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )

    # Attach the filler/command guard AFTER session.start
    FillerGuard(session)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
