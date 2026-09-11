from __future__ import annotations

import asyncio
import re
from collections import Counter
from dataclasses import dataclass, field

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentServer, AgentSession, UserInputTranscribedEvent
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import cartesia, groq, silero

from tools import book_emergency, create_booking, take_message

load_dotenv(".env.local")

PRACTICE_FACTS = """
Parkline Dental, 7 Albion St, Harris Park NSW.
Hours: Mon-Thu 8:30am-5:30pm, Fri 8:30am-4:00pm, Sat 9:00am-1:00pm, closed Sunday.
Dentists: Dr Nguyen, Dr Fares.
Services offered: Check-up & clean (45 min), Filling (45 min),
Emergency / toothache (30 min), Extraction consult (30 min).
NOT offered, ever - do not book or discuss availability for: orthodontics,
implants, cosmetic whitening, paediatric sedation. If asked, say plainly
that the practice doesn't offer that.
""".strip()

# --- deterministic urgent-path trigger (doesn't wait on the LLM) ---
URGENT_KEYWORDS = re.compile(
    r"\b(pain|hurts?|hurting|throb\w*|ache\w*|aching|swell\w*|swollen|"
    r"bleed\w*|blood|knocked[\s-]?out|broken\s+tooth|broke\s+my\s+tooth|"
    r"crack(ed|ing)?\s+tooth|chipped.{0,15}bad|severe)\b",
    re.IGNORECASE,
)

# --- deterministic repeat-question loop guard ---
REPEAT_WINDOW = 4
REPEAT_MIN_OCCURRENCES = 3
REPEAT_MIN_SHARED_WORDS = 2
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "with", "for", "to", "of", "in", "on", "at", "and", "or", "i", "you",
    "your", "my", "me", "can", "could", "would", "will", "please", "hi",
    "hello", "that", "this", "it", "id", "im", "we", "us", "about", "be",
    "have", "has", "had", "so", "just", "also", "okay", "ok", "nova",
    "want", "like", "book", "tell", "know",
}


def _significant_words(text: str) -> set[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _is_repeat_loop(recent: list[str]) -> bool:
    """True once the last few turns keep circling the same content words,
    however they're rephrased - e.g. asking a fee question 4 different ways."""
    if len(recent) < REPEAT_WINDOW:
        return False
    counts = Counter()
    for text in recent[-REPEAT_WINDOW:]:
        counts.update(_significant_words(text))
    repeated = [w for w, c in counts.items() if c >= REPEAT_MIN_OCCURRENCES]
    return len(repeated) >= REPEAT_MIN_SHARED_WORDS


end_call_tool = EndCallTool(
    extra_description=(
        "Call this immediately after you have confirmed a booking or a "
        "message and read every detail back to the caller, or immediately "
        "after telling a life-threatening caller to hang up and dial 000. "
        "Do not wait for the caller to say goodbye first."
    ),
    end_instructions="Thank the caller, remind them of their reference number if given, and say a brief goodbye.",
    delete_room=True,
)


@dataclass
class NovaState:
    recent_transcripts: list[str] = field(default_factory=list)
    repeat_strikes: int = 0


class RoutineAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""
You are Nova, the voice assistant for Parkline Dental.

{PRACTICE_FACTS}

On a routine call, collect these five fields before booking, one or two at
a time - don't interrogate:
1. full name  2. callback mobile  3. which service  4. preferred day/time
5. new or existing patient

Once you have all five, read them back for confirmation, then call
create_booking. If it returns accepted: false, explain the reason in plain
language and, when suggested_alternatives is present, offer both options.

Never quote fees, health fund, HICAPS, or Medicare figures - call take_message.
Never give clinical or pain-management advice - call take_message instead.

As soon as a booking or message is confirmed and read back, thank the
caller, give the reference number, and call end_call - don't wait for them
to say goodbye first.
""",
            tools=[create_booking, take_message, *end_call_tool.tools],
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the caller as Nova from Parkline Dental and ask how you can help."
        )


class UrgentAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=f"""
You are Nova, now on the urgent path because the caller may be in dental
pain or distress.

{PRACTICE_FACTS}

Step 1 - always first ask a calm, direct question to rule out a
life-threatening emergency: heavy bleeding that won't stop, facial swelling
affecting breathing/swallowing, loss of consciousness, or major trauma. If
confirmed, tell them clearly to hang up and call 000 right now, do not book,
and call end_call.

Step 2 - if not life-threatening, briefly acknowledge the pain with empathy.
Never give clinical or pain-management advice.

Step 3 - get name and mobile if you don't have them, then call
book_emergency. Only the next free 11:30 or 15:30 slot is ever offered -
never negotiate a different time.

Step 4 - read back the slot and reference number, ask them to arrive a few
minutes early, thank them, and call end_call.
""",
            tools=[book_emergency, take_message, *end_call_tool.tools],
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Calmly tell the caller you're switching to urgent booking, then ask the life-threatening question."
        )


server = AgentServer()


@server.rtc_session(agent_name="nova-dental")
async def entrypoint(ctx: agents.JobContext) -> None:
    session = AgentSession[NovaState](
        userdata=NovaState(),
        stt=groq.STT(model="whisper-large-v3-turbo"),
       llm=groq.LLM(model="openai/gpt-oss-120b"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
    )

    async def break_loop() -> None:
        await session.say(
            "I want to make sure this actually gets sorted, so I'll pass it "
            "straight to the practice team and they'll call you back. "
            "Thanks for calling Parkline Dental.",
            allow_interruptions=False,
        )
        await ctx.delete_room()
        ctx.shutdown(reason="repeat-loop guard")

    @session.on("user_input_transcribed")
    def on_transcript(ev: UserInputTranscribedEvent) -> None:
        if not ev.is_final or not ev.transcript.strip():
            return
        text = ev.transcript.strip()
        state = session.userdata

        if isinstance(session.current_agent, RoutineAgent) and URGENT_KEYWORDS.search(text):
            session.update_agent(UrgentAgent())
            state.recent_transcripts.clear()
            return

        state.recent_transcripts.append(text)
        state.recent_transcripts[:] = state.recent_transcripts[-REPEAT_WINDOW:]
        if _is_repeat_loop(state.recent_transcripts):
            state.repeat_strikes += 1
            state.recent_transcripts.clear()
            asyncio.create_task(break_loop())

    await session.start(room=ctx.room, agent=RoutineAgent())


if __name__ == "__main__":
    agents.cli.run_app(server)