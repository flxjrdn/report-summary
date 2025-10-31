import re

_NBSP = "\u00a0"
_THIN = "\u2009"
_NNBSP = "\u202f"

_RE_SOFT_HYPHEN = re.compile("\u00ad")

# NEW: remove a space that directly precedes a hyphenated line-break join
# e.g., "... Mindest kapitalanfor-\n derung" -> remove the space before "kapitalanfor-"
_RE_SPACE_BEFORE_HYPH_JOIN = re.compile(
    r"(?<=\w)\s+(?=\w+[ \t]*-\s*\r?\n\s*\w)", flags=re.UNICODE
)

# Existing: end-of-line hyphenation join (also allow optional spaces around)
_RE_EOL_HYPH = re.compile(r"(?<=\w)\s*-\s*\r?\n\s*(?=\w)", flags=re.UNICODE)

_RE_EOL_MIDWORD = re.compile(r"(?<=[A-Za-zÄÖÜäöüß])\r?\n(?=[a-zäöüß])")
_RE_SPECIAL_SPACES = re.compile("[{}]".format(re.escape(_NBSP + _THIN + _NNBSP)))
_RE_MANY_SPACES = re.compile(r"[ \t]{2,}")


def normalize_hyphenation(text: str) -> str:
    if not text:
        return text
    s = text

    # 1) Normalize exotic spaces and remove soft hyphens
    s = _RE_SOFT_HYPHEN.sub("", s)
    s = _RE_SPECIAL_SPACES.sub(" ", s)

    # 2) Remove stray space immediately before a hyphenated join
    s = _RE_SPACE_BEFORE_HYPH_JOIN.sub("", s)

    # 3) Join words broken by hyphen at end of line
    s = _RE_EOL_HYPH.sub("", s)

    # 4) Join words broken across newline without hyphen (conservative)
    s = _RE_EOL_MIDWORD.sub("", s)

    # 5) Collapse long runs of spaces (but keep single newlines)
    s = _RE_MANY_SPACES.sub(" ", s)

    return s
