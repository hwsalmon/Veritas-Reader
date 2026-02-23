"""Text preprocessing for TTS synthesis.

Cleans and normalises text before it reaches the TTS engine so that
numbers, abbreviations, currency symbols, and markdown artefacts are
spoken naturally rather than read literally.
"""

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abbreviation expansions — applied in order, case-sensitive first pass
# then a case-insensitive second pass for common lowercase abbreviations.
# ---------------------------------------------------------------------------

_ABBREV = [
    # Titles
    (r"\bDr\.", "Doctor"),
    (r"\bMr\.", "Mister"),
    (r"\bMrs\.", "Missus"),
    (r"\bMs\.", "Miss"),
    (r"\bProf\.", "Professor"),
    (r"\bSt\.", "Saint"),
    (r"\bGen\.", "General"),
    (r"\bCol\.", "Colonel"),
    (r"\bSgt\.", "Sergeant"),
    (r"\bCpt\.", "Captain"),
    (r"\bLt\.", "Lieutenant"),
    (r"\bRev\.", "Reverend"),
    (r"\bHon\.", "Honorable"),
    (r"\bGov\.", "Governor"),
    (r"\bPres\.", "President"),
    (r"\bSec\.", "Secretary"),
    # Common Latin / writing abbreviations
    (r"\be\.g\.", "for example"),
    (r"\bi\.e\.", "that is"),
    (r"\betc\.", "and so on"),
    (r"\bviz\.", "namely"),
    (r"\bcf\.", "compare"),
    (r"\bvs\.", "versus"),
    (r"\bvs\b", "versus"),
    (r"\bNo\.\s*(?=\d)", "number "),
    # Measurements (keep spacing tight)
    (r"\bkm/h\b", "kilometres per hour"),
    (r"\bm/s\b", "metres per second"),
    (r"\bkg\b", "kilograms"),
    (r"\blbs?\b", "pounds"),
    (r"\bkm\b", "kilometres"),
    (r"\bcm\b", "centimetres"),
    (r"\bmm\b", "millimetres"),
    (r"\bft\b", "feet"),
    (r"\bin\b", "inches"),
    (r"\bmi\b", "miles"),
    (r"\bMHz\b", "megahertz"),
    (r"\bGHz\b", "gigahertz"),
    (r"\bkWh\b", "kilowatt hours"),
    # US states — only expand inside parentheses or after comma to avoid
    # false positives (e.g. "NY" in proper nouns)
    # (skipped — too ambiguous without sentence context)
]

_ABBREV_COMPILED = [(re.compile(pat), repl) for pat, repl in _ABBREV]


def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREV_COMPILED:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

_CURRENCY_SYMBOLS = {
    "$": "dollars",
    "£": "pounds",
    "€": "euros",
    "¥": "yen",
    "₹": "rupees",
}

_CURRENCY_RE = re.compile(
    r"([£$€¥₹])\s*([\d,]+(?:\.\d{1,2})?)"
)


def _expand_currency(text: str) -> str:
    def _replace(m: re.Match) -> str:
        symbol = m.group(1)
        amount_str = m.group(2).replace(",", "")
        unit = _CURRENCY_SYMBOLS.get(symbol, "")
        try:
            from num2words import num2words
            amount = float(amount_str)
            if amount == int(amount):
                spoken = num2words(int(amount))
            else:
                spoken = num2words(amount, to="currency", lang="en").replace(
                    "euro", unit
                ).replace("dollar", unit).replace("pound", unit)
                return spoken
            return f"{spoken} {unit}"
        except Exception:
            return m.group(0)

    return _CURRENCY_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Percentages
# ---------------------------------------------------------------------------

_PCT_RE = re.compile(r"([\d,]+(?:\.\d+)?)\s*%")


def _expand_percentages(text: str) -> str:
    def _replace(m: re.Match) -> str:
        try:
            from num2words import num2words
            val = float(m.group(1).replace(",", ""))
            return f"{num2words(val if val != int(val) else int(val))} percent"
        except Exception:
            return m.group(0)

    return _PCT_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Ordinals  (1st, 2nd, 3rd, 4th …)
# ---------------------------------------------------------------------------

_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b")


def _expand_ordinals(text: str) -> str:
    def _replace(m: re.Match) -> str:
        try:
            from num2words import num2words
            return num2words(int(m.group(1)), to="ordinal")
        except Exception:
            return m.group(0)

    return _ORDINAL_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Plain numbers (integers and decimals, with optional thousand-separators)
# ---------------------------------------------------------------------------

# Match numbers NOT already handled (currency/percent/ordinal stripped above)
_NUMBER_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\b")


def _expand_numbers(text: str) -> str:
    def _replace(m: re.Match) -> str:
        raw = m.group(1).replace(",", "")
        try:
            from num2words import num2words
            val = float(raw)
            if val == int(val) and "." not in m.group(1):
                return num2words(int(val))
            return num2words(val)
        except Exception:
            return m.group(0)

    return _NUMBER_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Markdown / formatting artefacts
# ---------------------------------------------------------------------------

_MD_RULES = [
    # Headings → keep text, drop # prefix
    (re.compile(r"^#{1,6}\s+", re.MULTILINE), ""),
    # Bold / italic (** * __ _)
    (re.compile(r"\*{1,3}(.+?)\*{1,3}", re.DOTALL), r"\1"),
    (re.compile(r"_{1,3}(.+?)_{1,3}", re.DOTALL), r"\1"),
    # Inline code
    (re.compile(r"`(.+?)`"), r"\1"),
    # Fenced code blocks — replace with a short spoken note
    (re.compile(r"```[\s\S]+?```"), "[code block omitted]"),
    # Links [text](url) → text
    (re.compile(r"\[(.+?)\]\(.+?\)"), r"\1"),
    # Images ![alt](url) → alt
    (re.compile(r"!\[(.+?)\]\(.+?\)"), r"\1"),
    # Blockquote markers
    (re.compile(r"^>\s?", re.MULTILINE), ""),
    # Horizontal rules
    (re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE), ""),
    # HTML tags
    (re.compile(r"<[^>]+>"), ""),
    # Collapse multiple blank lines to two
    (re.compile(r"\n{3,}"), "\n\n"),
]


def _strip_markdown(text: str) -> str:
    for pattern, replacement in _MD_RULES:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Miscellaneous symbol replacements
# ---------------------------------------------------------------------------

_SYMBOL_MAP = [
    ("&", " and "),
    ("+", " plus "),
    ("=", " equals "),
    ("@", " at "),
    ("~", " approximately "),
    ("/", " slash "),
    ("\\", " backslash "),
    ("|", " "),
    ("<", " less than "),
    (">", " greater than "),
]


def _replace_symbols(text: str) -> str:
    # Only apply inside word boundaries where they look like operators,
    # not inside URLs or file paths
    for sym, replacement in _SYMBOL_MAP:
        # Surrounded by non-alpha on both sides (keeps / in URLs)
        text = re.sub(
            rf"(?<!\w){re.escape(sym)}(?!\w)", replacement, text
        )
    return text


# ---------------------------------------------------------------------------
# Prosodic punctuation — guides Kokoro's internal prosody model
# ---------------------------------------------------------------------------

# Introductory words/phrases that Kokoro handles more naturally with a comma
_INTRO_WORDS = [
    "However", "Therefore", "Nevertheless", "Furthermore", "Moreover",
    "Meanwhile", "Consequently", "Additionally", "Indeed", "Actually",
    "Honestly", "Frankly", "Clearly", "Naturally", "Obviously",
    "Unfortunately", "Fortunately", "Interestingly", "Importantly",
    "Surprisingly", "Remarkably", "Notably", "Admittedly",
    "Inevitably", "Ultimately", "Essentially", "Fundamentally",
    "In fact", "In turn", "As such", "Even so", "That said",
    "Of course", "After all", "At last", "In short", "In brief",
    "On the other hand", "On the contrary", "By contrast",
]

# Pre-compile: match intro word/phrase at a sentence start (after .!? or ^)
# only if NOT already followed by a comma
_INTRO_PATTERNS = []
for _w in _INTRO_WORDS:
    _esc = re.escape(_w)
    # After a sentence boundary
    _INTRO_PATTERNS.append(re.compile(
        rf'(?<=[.!?]\s)({_esc})\s+(?=[a-zA-Z])',
    ))
    # At the very start of the text / a line
    _INTRO_PATTERNS.append(re.compile(
        rf'^({_esc})\s+(?=[a-zA-Z])',
        re.MULTILINE,
    ))


def _add_prosodic_commas(text: str) -> str:
    """Insert commas after introductory words/phrases that lack one.

    Kokoro's acoustic model treats a comma as a breath/prosodic boundary,
    which makes the voice sound more natural and less rushed.
    """
    for pat in _INTRO_PATTERNS:
        # Only insert comma if one isn't already there
        text = pat.sub(lambda m: m.group(1) + ", ", text)
    return text


def _add_breath_points(text: str, max_words: int = 22) -> str:
    """Insert commas before coordinating conjunctions in long clauses.

    Lines with more than *max_words* words are scanned for ' and ', ' but ',
    ' so ', ' yet ', ' nor ' without a preceding comma — a comma is added so
    Kokoro knows to breathe there rather than barrel through.
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        if len(line.split()) >= max_words:
            line = re.sub(
                r'(?<![,;:])\s+(and|but|so|yet|nor)\s+',
                r', \1 ',
                line,
                flags=re.IGNORECASE,
            )
        result.append(line)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(text: str) -> str:
    """Clean and normalise *text* for natural TTS synthesis.

    Steps applied in order:
    1. Strip markdown formatting
    2. Expand currency
    3. Expand percentages
    4. Expand ordinal numbers
    5. Expand abbreviations
    6. Expand plain numbers
    7. Replace lone symbols
    8. Add prosodic commas after introductory words
    9. Add breath-point commas in long clauses
    10. Normalise whitespace
    """
    text = _strip_markdown(text)
    text = _expand_currency(text)
    text = _expand_percentages(text)
    text = _expand_ordinals(text)
    text = _expand_abbreviations(text)
    text = _expand_numbers(text)
    text = _replace_symbols(text)
    text = _add_prosodic_commas(text)
    text = _add_breath_points(text)
    # Collapse runs of spaces (but preserve newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()
    logger.debug("Preprocessed text (first 120 chars): %s", text[:120])
    return text
