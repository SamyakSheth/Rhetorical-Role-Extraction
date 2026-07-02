# sentence_splitter.py
# ---------------------------------------------------------
# A thin class wrapper around the existing pure-Python pipeline:
# - Cleans blocks while preserving newlines
# - Splits into sentences using regex rules + abbreviation guards + bullet-aware newlines
# - Removes bullet/enumeration prefixes per sentence
# - Outputs sentence-level rows with inherited labels/metadata (when provided)
#
# NOTE: This class preserves the original logic and patterns you provided.
#       No behavioral changes—only packaging into a class.
# ---------------------------------------------------------

from __future__ import annotations
import re
import html
import unicodedata
from typing import List, Optional, Sequence, Dict, Any
import pandas as pd


class sentence_splitter:
    def __init__(self) -> None:
        """Initialize and compile all patterns exactly as in the original script."""
        # ---------- Cleaning helpers ----------
        # Zero-width and soft hyphen characters that can sneak into text (PDF/HTML artifacts)
        self._ZW_CHARS = ["\u200B", "\u200C", "\u200D", "\u2060", "\u00AD"]
        self._ZW_RE = re.compile("|".join(map(re.escape, self._ZW_CHARS)))

        # Control characters (except \n, \t) that should be removed
        self._CTRL_RE = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]")

        # Hyphenation across line breaks: "woord-\nvervolg" => "woordvervolg"
        self._HYPH_WRAP_RE = re.compile(r"(\w)-\n([a-z\u00e0-\u024f])")

        # Fix accidental spaces before punctuation: "woord ," => "woord,"
        self._SPACE_BEFORE_PUNCT = re.compile(r"\s+([,;:.!?])")

        # Collapse 3+ newlines (extreme vertical spacing) down to 2
        self._MULTI_NL_RE = re.compile(r"\n{3,}")

        # Collapse runs of horizontal whitespace (spaces/tabs) inside a line
        self._MULTI_SPACE_RE = re.compile(r"[ \t\f\v]+")

        # Bullet/enumeration prefix at the start of a line/sentence (original pattern)
        self._BULLET_PREFIX_RE = re.compile(
            r"""(?mx)
            ^\s*(
                (?:\(?\d{1,3}(?:\.\d{1,3})*\)?[.)])      # 1. 1.2. (1). 1)
              | (?:\(?[a-zA-Z]{1,3}\)?[.)])             # (a) a) A.
              #| (?:\(?[ivxlcdmIVXLCDM]{1,6}\)?[.)])      # (iv) IV.
              | (?:[-•*])                                # -, •, *
            )\s*
            """
        )
        # Put near your other compiled regexes
        self._TRAILING_ENUM_RE = re.compile(r"""(?ix)
            (?:                                  # one or more trailing enumeration tokens
            \s+                                # MUST be separated by whitespace from sentence
            (?:                                # token forms:
                \(?\d{1,3}(?:\.\d{1,3})*\)?      # 1  or 1.2 or (1.2)
                | \(?[ivxlcdm]{1,6}\)?           # iv or (iv)
                | \(?[A-Za-z]{1,3}\)?            # a  or (a) or A.
            )
            [.)]?                              # optional trailing . or )
            )+\s*$
        """)





        # Abbreviations that should NOT cause sentence splits on their trailing period
        self.ABBR = {
            "art.", "mr.", "mevr.", "dr.", "prof.", "nr.", "nrs.", "jrg.", "p.", "blz.",
            "hr.", "rb.", "hof.", "vzr.", "bijv.", "b.v.", "enz.", "etc.", "i.v.m.",
            "o.a.", "o.m.", "t.o.v.", "m.i.", "m.n.", "m.b.t.", "ca.", "rov.", "r.o.", "B.","B. "
        }

        # Headings that should be kept as standalone sentences/units
        self.HEADING_RE = re.compile(
            r'^\s*(FEITEN|VASTSTAANDE FEITEN|PROCESVERLOOP|BEOORDELING|BESLISSING|GRIEF\s+\d+)\s*:?\s*$',
            re.I
        )

        # Primary sentence-ending punctuation. Abbreviation guard logic is applied later.
        self.SENT_END = re.compile(r"[.!?]")

    # ----------------------------- Public API -----------------------------

    def normalize_text_keep_newlines(self, s: str) -> str:
        """
        Conservative text normalization that preserves newlines.
        - Unicode NFKC normalization + HTML entity decode
        - Remove zero-width/control chars
        - Normalize bracketed party names like "[appellante \n]" => "[appellante]"
        - Normalize euro amounts like "€ 15.000,-"
        - Repair word-<newline>wrap hyphenation
        - Fix spacing before punctuation
        - Clamp excessive vertical spacing (3+ newlines -> 2)
        - Collapse multiple spaces per line (keep newlines)
        """
        s = html.unescape(unicodedata.normalize("NFKC", str(s)))
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = self._ZW_RE.sub("", s)
        s = self._CTRL_RE.sub("", s)
        # bracketed party names "[foo \n bar]" -> "[foo bar]"
        s = re.sub(
            r"\[\s*([^\]\n]+?)\s*\]",
            lambda m: "[" + re.sub(r"\s+", " ", m.group(1)).strip() + "]",
            s,
        )
        # euro like "€ 15.000,-"
        s = re.sub(
            r"€\s*(\d{1,3}(?:\.\d{3})*)(?:,(\d{2}))?-\b",
            lambda m: "€ " + m.group(1) + ("," + m.group(2) if m.group(2) else "") + "-",
            s,
        )
        # hyphen wrap repair word-\nlower -> wordlower
        s = self._HYPH_WRAP_RE.sub(r"\1\2", s)
        # tidy spaces before punctuation
        s = self._SPACE_BEFORE_PUNCT.sub(r"\1", s)
        # clamp huge gaps
        s = self._MULTI_NL_RE.sub("\n\n", s)
        # within-line space collapse
        s = "\n".join(self._MULTI_SPACE_RE.sub(" ", ln).strip() for ln in s.split("\n"))
        return s

    def split_sentences_rule_based(self, text: str) -> List[str]:
        """
        Split a block into sentences using:
        - paragraph/newline structure (headings, bullets start new 'piece')
        - punctuation-based splitting with abbreviation guards
        - bracket protection so citations inside () and [] don't split
        Post-processing removes bullet prefixes and normalizes whitespace.
        """
        # PROTECT punctuation inside (...) and [...] so it doesn't cause false splits
        text = self._protect_bracket_punct(text)

        lines = text.split("\n")
        pieces: List[str] = []

        # First: split by lines; start a new piece on headings and bullet lines
        buffer: List[str] = []
        for ln in lines:
            if not ln.strip():
                # paragraph break -> flush buffer
                if buffer:
                    pieces.append(" ".join(buffer).strip())
                    buffer = []
                continue

            if self.HEADING_RE.match(ln):
                if buffer:
                    pieces.append(" ".join(buffer).strip())
                    buffer = []
                pieces.append(ln.strip())
                continue

            if self._BULLET_PREFIX_RE.match(ln):
                # flush previous piece and start a new one at the bullet
                if buffer:
                    pieces.append(" ".join(buffer).strip())
                    buffer = []
                buffer.append(ln.strip())
            else:
                buffer.append(ln.strip())

        if buffer:
            pieces.append(" ".join(buffer).strip())

        # Now, within each piece, split on sentence enders while guarding abbreviations
        sents: List[str] = []
        for chunk in pieces:
            i = 0
            start = 0
            while i < len(chunk):
                m = self.SENT_END.search(chunk, i)
                if not m:
                    break
                end_idx = m.end()  # include the ender

                # last token before the period (to check abbreviations like 'art.')
                prev_span = chunk[:end_idx].rstrip()
                prev_word = (
                    prev_span[: prev_span.rfind(".") + 1].split()[-1].lower()
                    if "." in prev_span
                    else ""
                )

                # guard: abbreviations shouldn't split
                if prev_word in self.ABBR:
                    i = end_idx
                    continue

                # guard: period followed by digit (e.g., "art. 6:162") shouldn't split
                after = chunk[end_idx:].lstrip()
                if after[:1].isdigit():
                    i = end_idx
                    continue

                # heuristics: if next token looks like start of a sentence (cap/quote/bracket), split
                if after[:1] and re.match(r"""[\[\("']|[A-ZÀ-ÖØ-Þ]""", after[:1]):
                    sents.append(chunk[start:end_idx].strip())
                    start = end_idx
                    i = end_idx
                else:
                    # ambiguous case: split only if followed by double-space (layout hint)
                    if after.startswith("  "):
                        sents.append(chunk[start:end_idx].strip())
                        start = end_idx
                    i = end_idx

            # tail: whatever remains after the last ender
            tail = chunk[start:].strip()
            if tail:
                sents.append(tail)

        # Remove bullet prefixes from each sentence now and tidy
        sents = [self._BULLET_PREFIX_RE.sub("", s).strip() for s in sents if s.strip()]
        # Also remove bullet-like suffixes (rare trailing enumerations)
        sents = [
            re.sub(
            r"(?:\s*\(?[0-9ivxlcdmIVXLC]{1,3}(?:\.[0-9]{1,3})*\)?[.)]?\s*)+$",
            "",s,).strip() for s in sents]
        
        EDGE_NUM_LEAD = re.compile(r"""(?x)
            ^\s*                              # start + optional spaces
            (?:\(?\d{1,3}\)?[.)]?)            # 1 | (1) | 1. | 1) | (1).
            (?:\s+|$)                         # followed by space or end
        """)

        EDGE_NUM_TAIL = re.compile(r"""(?x)
            (?:\s+|^)                         # preceded by space or start
            (?:\(?\d{1,3}\)?[.)]?)            # 1 | (1) | 1. | 1) | (1).
            \s*$                              # to the end
        """)

        sents = [EDGE_NUM_LEAD.sub("", s).strip() for s in sents]
        sents = [EDGE_NUM_TAIL.sub("", s).strip() for s in sents]

        # restore punctuation we protected inside brackets
        sents = [self._unprotect_bracket_punct(s) for s in sents]

        # Replace ALL newline characters (single or multiple) with one space
        # (kept exactly as in your original code: this matches the literal '\n' sequence)
        sents = [re.sub(r'\s*\\n+\s*', ' ', s).strip() for s in sents if s and s.strip()]

        # normalize leftover spaces
        sents = [re.sub(r"\s+", " ", s).strip() for s in sents]
        #sents = [self._TRAILING_ENUM_RE.sub("", s).strip() for s in sents]
        # Replace all runs of whitespace (spaces, tabs, newlines) with a single space
        sents = [re.sub(r'\s+', ' ', s).strip() for s in sents if s and s.strip()]
        return sents

    def block_to_sentences(self, text: str) -> List[str]:
        """Convenience: clean a single block, split, and return a list of sentences."""
        cleaned = self.normalize_text_keep_newlines(text)
        return self.split_sentences_rule_based(cleaned)

    def dataframe_to_sentences(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        label_col: str = "label",
        meta_cols: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """
        Convert a dataframe of blocks to a sentence-level dataframe.
        - Preserves original row order by iterating as-is.
        - Each sentence inherits the block's label and any provided metadata columns.
        """
        rows: List[Dict[str, Any]] = []
        meta_cols = list(meta_cols or [])

        # Validate required columns exist
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Expected '{text_col}' and '{label_col}' columns in dataframe.")

        for row_idx, row in enumerate(df.itertuples(index=False), start=1):
            block_text = str(getattr(row, text_col))
            block_label = getattr(row, label_col)
            meta = {c: getattr(row, c) if c in df.columns else None for c in meta_cols}

            sents = self.block_to_sentences(block_text)

            for sent_idx, sent in enumerate(sents, start=1):
                rows.append(
                    {
                        "block_row": row_idx,        # original block order (1-based)
                        "sent_index": sent_idx,      # sentence order within the block
                        "sent_text": sent,           # final cleaned sentence
                        "sent_len": len(sent),       # quick sanity metric
                        label_col: block_label,      # inherits the block label
                        **meta,
                    }
                )

        return pd.DataFrame(rows)

    # ------------------------- internal helpers --------------------------

    def _protect_bracket_punct(self, text: str) -> str:
        """Protect punctuation inside (...) and [...] so internal periods don't split."""
        def _prot(inner: str) -> str:
            return (
                inner.replace(".", "<DOT>")
                .replace("!", "<EXC>")
                .replace("?", "<Q>")
            )

        text = re.sub(r"\(([^)]*)\)", lambda m: "(" + _prot(m.group(1)) + ")", text)
        text = re.sub(r"\[([^\]]*)\]", lambda m: "[" + _prot(m.group(1)) + "]", text)
        return text

    def _unprotect_bracket_punct(self, text: str) -> str:
        """Restore punctuation markers protected by _protect_bracket_punct."""
        return (
            text.replace("<DOT>", ".")
            .replace("<EXC>", "!")
            .replace("<Q>", "?")
        )
    
    def clean_block_preserve_paragraphs(self, text: str) -> str:
        text = re.sub(r'\\n', '\n', str(text))
        text = self.normalize_text_keep_newlines(text)

        out_lines = []
        for ln in text.split("\n"):
            raw = ln
            if not raw.strip():
                continue

            if self.HEADING_RE.match(raw):
                out_lines.append(raw.strip())
                continue

            ln = self._BULLET_PREFIX_RE.sub("", raw).strip()
            ln = re.sub(r'^\s*\d+(?:\.\d+)*\.\s*', "", ln)

            out_lines.append(ln)

        text = " ".join(out_lines)
        text = self._TRAILING_ENUM_RE.sub("", text)
        text = re.sub(r'\s+', ' ', text)

        return text.strip()


    def clean_cases_blocks(self, cases_dict: dict) -> dict:
        cleaned = {}
        for case_name, sections in cases_dict.items():
            cleaned[case_name] = {
                self.clean_block_preserve_paragraphs(sections)
            }
        return cleaned






# # --------------------------------------------------------------------

#     def clean_block_preserve_paragraphs(spl: sentence_splitter, text: str) -> str:
#         """
#         Clean a block/paragraph while flattening all extra newlines.
#         - Keeps your own normalization exactly as-is
#         - Removes bullet/enumeration prefixes at line starts (e.g., 3., 3.1., (a), a), -, •, *)
#         - Removes trailing dangling enumeration tokens (rare)
#         - Converts literal '\\n' into real newlines first (common in JSON dumps)
#         - Removes ALL newlines (joins lines into continuous text)
#         """
#         # 0) Convert literal '\n' into real newlines
#         text = re.sub(r'\\n', '\n', str(text))

#         # 1) Run your existing conservative normalizer
#         text = spl.normalize_text_keep_newlines(text)

#         # 2) Line-start de-bulleting / de-numbering; keep headings as-is
#         out_lines = []
#         for ln in text.split("\n"):
#             raw = ln
#             if not raw.strip():
#                 continue  # drop empty lines completely

#             # Keep headings intact
#             if spl.HEADING_RE.match(raw):
#                 out_lines.append(raw.strip())
#                 continue

#             # Remove bullet/enumeration prefixes like "3.", "3.1.", "(a)", etc.
#             ln = spl._BULLET_PREFIX_RE.sub("", raw).strip()
#             ln = re.sub(r'^\s*\d+(?:\.\d+)*\.\s*', "", ln)

#             out_lines.append(ln)

#         text = " ".join(out_lines)  # ✅ flatten all lines into one continuous block

#         # 3) Remove trailing enumeration fragments (rare)
#         text = spl._TRAILING_ENUM_RE.sub("", text)

#         # 4) Collapse any leftover whitespace (tabs/newlines/spaces)
#         text = re.sub(r'\s+', ' ', text)

#         return text.strip()

#     def clean_cases_blocks(cases_dict: dict, spl: sentence_splitter) -> dict:
#         """
#         Clean an in-memory dataset shaped like:
#         {
#             "ECLI:...txt": {
#                 "beoordeling": "...",
#                 "beslissing": "...",
#                 "materiele feiten": "...",
#                 "proceshandelingen": "..."
#             },
#             ...
#         }
#         Returns the same structure with cleaned block strings.
#         """
#         cleaned = {}
#         for case_name, sections in cases_dict.items():
#             cleaned[case_name] = {
#                 sec: clean_block_preserve_paragraphs(spl, txt)
#                 for sec, txt in sections.items()
#             }
#         return cleaned