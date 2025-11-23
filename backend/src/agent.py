import logging
import json
from typing import Annotated, List

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly, upbeat barista at 'Guitarbucks'. 
            Your goal is to take a complete coffee order from the customer.
            
            You must collect specific details to fill the order state. Do not ask for everything at once; be conversational and ask clarifying questions.
            
            Required Order Details:
            1. Drink Type (e.g., Latte, Espresso, Tea)
            2. Size (Small, Medium, Large)
            3. Milk Preference (e.g., Oat, Whole, Almond, None)
            4. Extras (e.g., Vanilla syrup, extra shot, sugar, or "none")
            5. Customer Name
            
            Process:
            1. Greet the customer warmly.
            2. Collect the missing details naturally.
            3. Once all 5 fields are known, summarize the full order clearly to the user for confirmation.
            4. ONLY after the user confirms the summary is correct, call the `save_order` tool.
            5. After the tool executes, ask if they want to order another drink. If yes, start fresh.
            """,
        )

    @function_tool
    async def save_order(
        self, 
        context: RunContext, 
        drink_type: Annotated[str, "The type of drink ordered"],
        size: Annotated[str, "The size of the drink"],
        milk: Annotated[str, "The milk preference"],
        extras: Annotated[List[str], "List of extras or modifications"],
        name: Annotated[str, "The customer's name"]
    ):
        """
        Saves the fully confirmed order to the system.
        """
        order_data = {
            "drinkType": drink_type,
            "size": size,
            "milk": milk,
            "extras": extras,
            "name": name
        }
        
        logger.info(f"Saving order: {order_data}")
        
        try:
            # Appending to a JSON lines file
            with open("orders.json", "a") as f:
                f.write(json.dumps(order_data) + "\n")
            return "Order saved successfully. You may now ask if the user wants another drink."
        except Exception as e:
            logger.error(f"Failed to save order: {e}")
            return "There was a technical error saving the order."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))