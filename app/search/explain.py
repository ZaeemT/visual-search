"""Turn measured signals into a sentence a shopper would understand.

Two stages, in this order on purpose:

1. A template builds a correct sentence from the signals. No network, no model,
   sub-millisecond, and always available.
2. Ollama rewrites those same facts into something that reads better, and
   translates the Russian category name into English on the way.

The LLM is a phrasing and translation step, never a source of facts. It is
handed a flat list of facts measured from the image and read from the
catalogue, and told to use nothing else. Its output is then checked against
those same facts, and anything that fails is discarded in favour of the
template. The worst case is a blander sentence, never an invented one.

Scores and model internals never enter the prompt — "match_strength" arrives
already worded as "very close" — so they cannot surface in the text.

Tuned for a small local model (llama3.2:1b). Two things matter at that size:
the facts are a flat bullet list rather than nested JSON, and there is no
few-shot example. An example makes a 1b model copy its nouns into unrelated
answers, which is exactly the failure the groundedness check exists to catch.
"""

from __future__ import annotations

import logging
import re

import httpx

from .signals import COLOR_EN, MATERIAL_EN, MatchSignals

logger = logging.getLogger(__name__)

PROMPT = """Translate the following Russian product name into English. \
Reply with only the English translation, nothing else.

{name}"""

# Machinery that must never reach a shopper.
FORBIDDEN = re.compile(
    r"\b(embed\w*|vector|cosine|similarit\w*|distance|score|model|json|field|"
    r"database|dataset|algorithm|nearest|neighbou?r|pixel|metadata)\b",
    re.IGNORECASE,
)

# Visual details the signals never contain, so any mention was invented.
INVENTED_DETAIL = re.compile(
    r"\b(stripe[sd]?|floral|polka|check(ed|s)?|plaid|tartan|paisley|embroider\w*|"
    r"sequin\w*|ruffle[sd]?|pleat\w*|neckline|v-neck|crew ?neck|collar\w*|"
    r"sleeve\w*|sleeveless|button\w*|zipper\w*|pocket\w*|hood(ed|ie)?)\b",
    re.IGNORECASE,
)

# The note describes an item to a shopper; neither the item nor the shop speaks.
FIRST_PERSON = re.compile(r"(^|\s)(i|i'm|i am|i've|my|me|we|our|us)(\s|[.,!?']|$)", re.IGNORECASE)

# Claims about the shopper rather than about the two images. Nothing in the
# signals says what they own, wear, or are trying to coordinate with — the
# uploaded picture is one photo, not a wardrobe.
INVENTED_CONTEXT = re.compile(
    r"\b(wardrobe|outfit|cohesive|complement\w*|pair(s|ed|ing)? (it|well|nicely)|"
    r"you'?(ve|ll|r)\b|your (style|look|collection|existing)|"
    r"perfect(ly)? (for|match|complement)|great choice|will add)\b",
    re.IGNORECASE,
)

# A refusal or meta-comment rather than a product note.
NON_ANSWER = re.compile(
    r"(as an ai|i'm not able|i am not able|i cannot|i can't|sorry|"
    r"would you like|here (is|'s) (a|the)|instead,)",
    re.IGNORECASE,
)

# Cyrillic left in the output means the translation instruction was ignored.
CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def fact_lines(signals: MatchSignals) -> str:
    """The facts as a flat bullet list, omitting anything unknown.

    Only non-empty facts are listed, so the model is never shown a null and
    never has the chance to write "no fit specified".
    """
    facts = signals.as_facts()
    lines = [f"- item name (Russian): {facts['item_category']}"]

    if facts["item_colours"]:
        lines.append(f"- item colours: {', '.join(facts['item_colours'])}")
    if facts["item_fit"]:
        lines.append(f"- item fit: {facts['item_fit']}")
    if facts["item_length"]:
        lines.append(f"- item length: {facts['item_length']}")
    if facts["item_materials"]:
        lines.append(f"- item material: {', '.join(facts['item_materials'])}")
    if facts["photo_colours"]:
        lines.append(f"- colours in the shopper's photo: {', '.join(facts['photo_colours'])}")
    if facts["shared_colours"]:
        lines.append(f"- colours they share: {', '.join(facts['shared_colours'])}")
    lines.append(f"- how close the match is: {facts['match_strength']}")
    return "\n".join(lines)


def compose(signals: MatchSignals, category: str) -> str:
    """Build the sentence from the signals and a (possibly translated) category.

    The structure is fixed in code rather than left to the model, so every
    clause corresponds to a measured fact: the closeness bucket, the colours
    the two images share, the item's annotated shape and material. `category`
    is the only part that varies by translation.
    """
    strength = signals.as_facts()["match_strength"]
    parts = [f"A {strength} match for your photo"]

    if signals.shared_colors:
        parts.append(f", in the same {' and '.join(signals.shared_colors)} tones")
    elif signals.colors:
        parts.append(f", in {' and '.join(signals.colors)}")

    shape = " ".join(d for d in (signals.fit, signals.length) if d)
    parts.append(f": {shape + ' ' if shape else ''}{category.lower()}")

    if signals.materials:
        parts.append(f" in {signals.materials[0]}")

    return "".join(parts) + "."


def template_explanation(signals: MatchSignals) -> str:
    """The sentence with the category left in its original wording."""
    return compose(signals, signals.category)


def is_valid_translation(text: str, original: str) -> bool:
    """Accept only something that looks like a translated product name.

    A 1b model asked for one sentence will write a paragraph, apologise, or
    invent detail. Asked for two or three words, it either translates or fails
    obviously — and this check catches the obvious failures: leftover Cyrillic,
    a sentence rather than a name, digits, or boilerplate.
    """
    if not text or len(text) > 40:
        return False
    if CYRILLIC.search(text) or any(ch.isdigit() for ch in text):
        return False
    if len(text.split()) > 4 or text.endswith((".", "!", "?", ":")):
        return False
    if NON_ANSWER.search(text) or FIRST_PERSON.search(text):
        return False
    return text.strip().lower() != original.strip().lower() or not CYRILLIC.search(original)


def is_grounded(text: str, signals: MatchSignals) -> bool:
    """Reject phrasing that went beyond the facts it was given.

    Each check corresponds to something the signals cannot support, so a hit
    means the phrasing step started describing a product it never saw:

    * a colour or material neither annotated on the item nor seen in the photo
    * a pattern, neckline or sleeve — never in the signals at all
    * Cyrillic left untranslated, first person, refusals, or leaked machinery

    Failing costs only the template sentence, so the check is deliberately
    strict: a plain correct note beats a vivid invented one.
    """
    if not text or len(text) > 300:
        return False
    if FORBIDDEN.search(text) or CYRILLIC.search(text):
        return False
    if INVENTED_DETAIL.search(text) or INVENTED_CONTEXT.search(text):
        return False
    if FIRST_PERSON.search(text) or NON_ANSWER.search(text):
        return False

    allowed_colors = set(signals.colors) | set(signals.query.dominant_colors)
    for colour in set(COLOR_EN.values()):
        if colour not in allowed_colors and re.search(
            rf"\b{re.escape(colour)}\b", text, re.IGNORECASE
        ):
            return False

    allowed_materials = set(signals.materials)
    for material in set(MATERIAL_EN.values()):
        if material not in allowed_materials and re.search(
            rf"\b{re.escape(material)}\b", text, re.IGNORECASE
        ):
            return False
    return True


class Explainer:
    """Phrases explanations, with or without a local LLM.

    `enabled=False`, an unreachable Ollama, a timeout, or a failed groundedness
    check all lead to the same place: the template sentence.
    """

    def __init__(
        self,
        model: str = "llama3.2:1b",
        host: str = "http://127.0.0.1:11434",
        timeout: float = 10.0,
        enabled: bool = True,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.enabled = enabled
        self._client = httpx.Client(timeout=timeout) if enabled else None
        self._translations: dict[str, str] = {}
        self.rejected = 0  # how often the check fired, worth knowing in the logs

    def available(self) -> bool:
        if not self.enabled or self._client is None:
            return False
        try:
            tags = self._client.get(f"{self.host}/api/tags", timeout=2.0).json()
            family = self.model.split(":")[0]
            return any(m["name"].startswith(family) for m in tags.get("models", []))
        except Exception:
            return False

    def translate_category(self, name: str) -> str:
        """English name for a category, via the LLM, cached per process.

        Categories repeat constantly — a top-10 result set is often three or
        four distinct ones, and the same ones recur across queries — so the
        cache means most requests make no LLM call at all.
        """
        key = name.strip().lower()
        if key in self._translations:
            return self._translations[key]

        translated = name
        if self.enabled and self._client is not None:
            try:
                response = self._client.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": PROMPT.format(name=name),
                        "stream": False,
                        # Deterministic, and short: this is a two-word answer.
                        "options": {"temperature": 0.0, "num_predict": 12},
                    },
                )
                response.raise_for_status()
                candidate = response.json().get("response", "")
                candidate = candidate.strip().split("\n")[0].strip().strip('"').strip()
                if is_valid_translation(candidate, name):
                    translated = candidate
                else:
                    self.rejected += 1
                    logger.info("rejected translation of %r: %r", name, candidate)
            except Exception as exc:
                logger.warning("ollama unavailable (%s); keeping original name", exc)

        self._translations[key] = translated
        return translated

    def explain(self, signals: MatchSignals) -> str:
        """The grounded sentence, with the category name in English where possible.

        The sentence itself is assembled in code from measured facts, so it
        cannot contain an invented claim. The LLM's only contribution is
        translating the category name — a task small enough for a 1b model, and
        one whose output is easy to check.
        """
        return compose(signals, self.translate_category(signals.category))

    def explain_many(self, signals: list[MatchSignals]) -> list[str]:
        return [self.explain(s) for s in signals]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
