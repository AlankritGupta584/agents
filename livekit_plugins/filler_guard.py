# SPDX-License-Identifier: Apache-2.0
# livekit_plugins/filler_guard.py
import asyncio, logging, os, re, unicodedata
from collections import deque
from typing import Iterable, List, Set

from livekit.agents import (
    AgentSession,
    UserInputTranscribedEvent,
    AgentStateChangedEvent,
    SpeechCreatedEvent,
)

log = logging.getLogger("filler_guard")

_DEFAULT_IGNORED = ["uh", "umm", "um", "hmm", "haan", "huh", "mmm", "erm", "hmmkay"]
_DEFAULT_COMMANDS = ["stop", "wait", "hold on", "one second", "pause", "no", "not that", "cancel"]

def _env_list(name: str, default: List[str]) -> List[str]:
    val = os.getenv(name)
    return [x.strip() for x in val.split(",")] if val else default[:]

def _norm(s: str) -> str:
    # Unicode normalize, lowercase, collapse repeated letters, drop punctuation
    s = unicodedata.normalize("NFKD", s).lower()
    s = re.sub(r"[^\w\s]", " ", s)            # remove punct/symbols
    s = re.sub(r"(\w)\1{2,}", r"\1\1", s)     # cap long elongations (hmmmm -> hmm)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokens(s: str) -> List[str]:
    return [t for t in _norm(s).split(" ") if t]

class FillerGuard:
    """
    Intercepts LiveKit transcript events to ignore filler-only speech
    while the agent is speaking; immediately interrupts on commands.
    """

    def __init__(
        self,
        session: AgentSession,
        ignored_words: Iterable[str] | None = None,
        interrupt_words: Iterable[str] | None = None,
        *,
        min_chars_when_speaking: int = 1,   # heuristic fallback when no confidence
        window_secs: float = 2.0,           # rolling buffer while agent speaks
    ) -> None:
        self.session = session
        self.ignored: Set[str] = set(_norm(w) for w in (ignored_words or _env_list("IGNORED_WORDS", _DEFAULT_IGNORED)))
        self.commands: List[str] = [_norm(w) for w in (interrupt_words or _env_list("INTERRUPT_WORDS", _DEFAULT_COMMANDS))]
        self._speaking = False
        self._buf = deque(maxlen=6)  # short rolling buffer of recent partials
        self._lock = asyncio.Lock()
        self._min_chars = min_chars_when_speaking
        self._window_secs = window_secs

        @session.on("agent_state_changed")
        def _on_state(evt: AgentStateChangedEvent):
            self._speaking = (getattr(evt, "new_state", None) and evt.new_state.name == "speaking")
            if not self._speaking:
                self._buf.clear()

        @session.on("speech_created")
        def _on_speech(_: SpeechCreatedEvent):
            # Reset window for new speech; we don't need the handle to ignore fillers.
            self._buf.clear()

        @session.on("user_input_transcribed")
        def _on_transcribed(evt: UserInputTranscribedEvent):
            # Handle asynchronously to avoid blocking internal loop.
            asyncio.create_task(self._handle_transcript(evt))

    def update_ignored(self, words: Iterable[str]) -> None:
        with self._lock:  # type: ignore
            self.ignored = set(_norm(w) for w in words)

    def update_commands(self, words: Iterable[str]) -> None:
        with self._lock:  # type: ignore
            self.commands = [_norm(w) for w in words]

    async def _handle_transcript(self, evt: UserInputTranscribedEvent) -> None:
        text = evt.transcript or ""
        if not text:
            return

        norm_text = _norm(text)
        if not self._speaking:
            # Agent is quiet: register speech normally (do nothing here)
            return

        # While the agent is speaking:
        toks = _tokens(norm_text)
        if not toks:
            return

        # 1) Mixed / command detection across a rolling window (handles "umm okay stop")
        self._buf.append(norm_text)
        window_text = " ".join(self._buf)[-256:]  # cheap tail cap
        if any(cmd in window_text for cmd in self.commands):
            log.info("Valid interruption detected: %r", window_text)
            # Allowed even if allow_interruptions=False
            self.session.interrupt()  # stop TTS immediately
            return

        # 2) Filler-only check
        if all(t in self.ignored for t in toks) and len(norm_text.replace(" ", "")) >= self._min_chars:
            # Intentionally ignore: do not interrupt, do not clear turn.
            log.debug("Ignored filler while speaking: %r", norm_text)
            return

        # 3) Non-filler but too weak? (Heuristic fallback if no confidence)
        # Keep it conservative: only treat as interrupt if we see a non-filler token
        # and sufficient characters (prevents tiny blips). Otherwise, let it pass silently.
        non_filler = [t for t in toks if t not in self.ignored]
        if non_filler and sum(len(t) for t in non_filler) >= self._min_chars:
            log.info("Non-filler speech during agent speech: interrupting (%r)", norm_text)
            self.session.interrupt()
