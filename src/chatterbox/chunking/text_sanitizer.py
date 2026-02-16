"""Advanced text sanitization for optimal TTS quality."""
import logging
import re
import unicodedata
from typing import Dict, Optional, Tuple, List

# Optional dependencies for better text normalization.
# We keep these optional so the package can still run without them,
# but you should add them to requirements for best quality.
try:
    import inflect  # type: ignore
except Exception:  # pragma: no cover
    inflect = None

logger = logging.getLogger(__name__)


class AdvancedTextSanitizer:
    """Comprehensive text cleaning and normalization for optimal TTS quality"""
    
    # Language-specific allowed character sets
    # Characters outside these sets will be flagged as disallowed
    LANGUAGE_ALLOWED_CHARS = {
        'en': {
            # English: ASCII + common accented characters from Romance/Germanic/Nordic
            # (borrowed words, names, place names: café, naïve, Müller, Copenhagen, Åland)
            'allowed': set(range(32, 127)) | {ord(c) for c in 'áéíóúàèìòùâêîôûäëïöüñçæøåßÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜÑÇÆØÅ'},
            'description': 'English (ASCII + common accented characters)'
        },
        'es': {
            # Spanish: ASCII + Spanish-specific characters
            'allowed': set(range(32, 127)) | {ord(c) for c in 'áéíóúñüÁÉÍÓÚÑÜ¿¡'},
            'description': 'Spanish (ASCII + Spanish-specific characters)'
        },
        'fr': {
            # French: ASCII + French-specific characters
            'allowed': set(range(32, 127)) | {ord(c) for c in 'àâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ'},
            'description': 'French (ASCII + French-specific characters)'
        },
        'de': {
            # German: ASCII + German-specific characters
            'allowed': set(range(32, 127)) | {ord(c) for c in 'äöüßÄÖÜ'},
            'description': 'German (ASCII + German-specific characters)'
        },
        'it': {
            # Italian: ASCII + Italian-specific characters
            'allowed': set(range(32, 127)) | {ord(c) for c in 'àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ'},
            'description': 'Italian (ASCII + Italian-specific characters)'
        },
        'pt': {
            # Portuguese: ASCII + Portuguese-specific characters
            'allowed': set(range(32, 127)) | {ord(c) for c in 'áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ'},
            'description': 'Portuguese (ASCII + Portuguese-specific characters)'
        },
        'da': {
            # Danish: ASCII + Danish-specific characters (æ, ø, å)
            'allowed': set(range(32, 127)) | {ord(c) for c in 'æøåÆØÅ'},
            'description': 'Danish (ASCII + Danish-specific characters: æ, ø, å)'
        },
        'no': {
            # Norwegian: ASCII + Norwegian-specific characters (æ, ø, å)
            'allowed': set(range(32, 127)) | {ord(c) for c in 'æøåÆØÅ'},
            'description': 'Norwegian (ASCII + Norwegian-specific characters: æ, ø, å)'
        },
        'sv': {
            # Swedish: ASCII + Swedish-specific characters (ä, ö, å)
            'allowed': set(range(32, 127)) | {ord(c) for c in 'äöåÄÖÅ'},
            'description': 'Swedish (ASCII + Swedish-specific characters: ä, ö, å)'
        },
    }
    
    def __init__(self):
        # Problematic character mappings
        self.unicode_replacements = {
            # Unicode punctuation
            '…': '...',
            '–': '-', 
            '—': ' - ',
            ''': "'",
            ''': "'", 
            '"': '"',
            '"': '"',
            '«': '"',
            '»': '"',
            '„': '"',
            '"': '"',

            # Normalize apostrophe-like characters to ASCII apostrophe
            # (TTS tokenizers often behave better with U+0027 than typographic variants)
            '\u2019': "'",  # RIGHT SINGLE QUOTATION MARK
            '\u2018': "'",  # LEFT SINGLE QUOTATION MARK
            '\u02BC': "'",  # MODIFIER LETTER APOSTROPHE
            '\uFF07': "'",  # FULLWIDTH APOSTROPHE
            '\u2032': "'",  # PRIME (often mistaken for apostrophe)
            '`': "'",        # Grave accent commonly used as apostrophe
            
            # Story section breaks
            '⁂': ' <STORY_BREAK> ',  # Special marker for longer pauses between story sections
            
            # Mathematical symbols
            '×': ' times ',
            '÷': ' divided by ',
            '±': ' plus or minus ',
            '≤': ' less than or equal to ',
            '≥': ' greater than or equal to ',
            '≠': ' not equal to ',
            '≈': ' approximately ',
            '∞': ' infinity ',

            # Additional math operators / forms
            '−': ' minus ',          # Unicode minus
            '∙': ' times ',          # Bullet operator
            '·': ' times ',          # Middle dot
            '∝': ' proportional to ',
            '∴': ' therefore ',
            '∵': ' because ',

            # Common math constructs (kept simple)
            '√': ' square root ',
            '∑': ' sum ',
            '∫': ' integral ',

            # Superscripts
            '²': ' squared ',
            '³': ' cubed ',
            '⁴': ' to the fourth power ',

            # Common greek letters
            'π': ' pi ',
            'Π': ' pi ',
            'Δ': ' delta ',
            'δ': ' delta ',
            'λ': ' lambda ',
            'θ': ' theta ',
            'μ': ' mu ',
            'σ': ' sigma ',
            'Ω': ' omega ',
            'ω': ' omega ',
            
            # Currency symbols
            '€': ' euros',
            '£': ' pounds',
            '¥': ' yen',
            '₽': ' rubles',
            '₹': ' rupees',
            '₿': ' bitcoin',
            
            # Special symbols
            '©': ' copyright ',
            '®': ' registered ',
            '™': ' trademark ',
            '§': ' section ',
            '¶': ' paragraph ',
            '†': '',  # Remove dagger
            '‡': '',  # Remove double dagger
            '°': ' degrees ',
            '%': ' percent',
            '‰': ' per mille',
            
            # Arrows and symbols
            '→': ' arrow ',
            '←': ' arrow ',
            '↑': ' up arrow ',
            '↓': ' down arrow ',
            '⇒': ' implies ',
            '⇔': ' if and only if ',
            
            # Fractions
            '½': ' half',
            '⅓': ' one third',
            '⅔': ' two thirds',
            '¼': ' one quarter',
            '¾': ' three quarters',
            '⅛': ' one eighth',
            '⅜': ' three eighths',
            '⅝': ' five eighths',
            '⅞': ' seven eighths',
        }
        
        # Common abbreviations
        self.abbreviations = {
            'etc.': 'etcetera',
            'e.g.': 'for example',
            'i.e.': 'that is',
            'vs.': 'versus',
            'Mr.': 'Mister',
            'Mrs.': 'Missus',
            'Ms.': 'Miss',
            'Dr.': 'Doctor',
            'Prof.': 'Professor',
            'St.': 'Saint',
            'Ave.': 'Avenue',
            'Blvd.': 'Boulevard',
            'Rd.': 'Road',
            'Ct.': 'Court',
            'Ln.': 'Lane',
            'Pkwy.': 'Parkway',
            'Inc.': 'Incorporated',
            'Corp.': 'Corporation',
            'Ltd.': 'Limited',
            'Co.': 'Company',
            'Jr.': 'Junior',
            'Sr.': 'Senior',
            'Ph.D.': 'PhD',
            'M.D.': 'MD',
            'B.A.': 'BA',
            'M.A.': 'MA',
            'CEO': 'C E O',
            'CFO': 'C F O',
            'CTO': 'C T O',
            'USA': 'U S A',
            'UK': 'U K',
            'EU': 'E U',
            'FBI': 'F B I',
            'CIA': 'C I A',
            'NASA': 'N A S A',
            'GPS': 'G P S',
            'AI': 'A I',
            'API': 'A P I',
            'URL': 'U R L',
            'HTML': 'H T M L',
            'CSS': 'C S S',
            'JS': 'JavaScript',
            'iOS': 'i O S',
            'macOS': 'mac O S',
            'WiFi': 'Wi-Fi',
        }
        
        # Number patterns
        self.number_patterns = [
            (r'\b(\d{1,3}),(\d{3})\b', r'\1\2'),  # Remove commas in numbers
            (r'\$(\d+)', r'\1 dollars'),          # Currency
            (r'(\d+)%', r'\1 percent'),           # Percentages
            (r'(\d+)°([CF]?)', self._temperature_replace),  # Temperature
            (r'(\d+):\d{2}', self._time_replace), # Time format
        ]
        
        # Problematic sequences that cause TTS issues
        self.problematic_sequences = {
            '...': '. ',
            '!!': '!',
            '??': '?',
            '.,': '.',
            '.?': '?',
            '.!': '!',
            '!?': '!',
            '?!': '?',
            ';;': ';',
            '::': ':',
            '--': ' - ',
            '---': ' - ',
            '____': '',  # Remove long underscores
            '****': '',  # Remove asterisk patterns
            '####': '',  # Remove hash patterns
        }

        # Optional engines
        self._inflect_engine = inflect.engine() if inflect else None

        # Small digit vocabulary for decimal verbalization without heavy dependencies
        self._digit_words: Dict[str, str] = {
            '0': 'zero',
            '1': 'one',
            '2': 'two',
            '3': 'three',
            '4': 'four',
            '5': 'five',
            '6': 'six',
            '7': 'seven',
            '8': 'eight',
            '9': 'nine',
        }
    
    def _temperature_replace(self, match):
        """Convert temperature format"""
        num = match.group(1)
        unit = match.group(2) or 'F'
        unit_name = 'Fahrenheit' if unit == 'F' else 'Celsius' if unit == 'C' else 'degrees'
        return f"{num} degrees {unit_name}"
    
    def _time_replace(self, match):
        """Convert time format"""
        time_str = match.group(0)
        try:
            hour, minute = time_str.split(':')
            hour = int(hour)
            minute = int(minute)
            # Prefer "oh five" over "05" since later number verbalization would
            # otherwise turn "05" into "five" and lose the leading-zero cue.
            if minute == 0:
                minute_spoken = "o'clock"
            elif minute < 10:
                minute_spoken = f"oh {minute}"
            else:
                minute_spoken = f"{minute}"
            
            if hour == 0:
                return f"twelve {minute_spoken} AM"
            elif hour < 12:
                return f"{hour} {minute_spoken} AM"
            elif hour == 12:
                return f"twelve {minute_spoken} PM"
            else:
                return f"{hour-12} {minute_spoken} PM"
        except:
            return time_str

    def _number_to_words(self, n: int) -> str:
        """Convert integer to words with a stable 'and' style when available."""
        if self._inflect_engine is None:
            return str(n)
        try:
            # Use "and" (e.g., 1278 -> "one thousand two hundred and seventy-eight")
            out = self._inflect_engine.number_to_words(n, andword="and", zero="zero")
            # inflect can include commas; remove them for TTS cleanliness
            out = out.replace(",", "")
            return out
        except Exception:
            return str(n)

    def _digits_to_words(self, digits: str) -> str:
        """Convert a digit string (e.g. '012') to 'zero one two'."""
        return " ".join(self._digit_words.get(ch, ch) for ch in digits)

    def _verbalize_simple_equations(self, text: str) -> str:
        """
        Lightweight equation verbalizer for common inline forms like:
        - E=mc^{2}
        - x_1=3.14
        - a*b=c
        This is intentionally NOT a full LaTeX/math parser.
        """
        if not text:
            return text

        # Only do equation-level rewrites when the text looks "math-ish".
        # This avoids mangling prose that contains '-' or '/'.
        if not any(ch in text for ch in ("=", "^", "_", "{", "}")):
            return text

        # Exponents: x^{2} / x^2
        def _exp_repl(m: re.Match) -> str:
            base = m.group(1)
            exp = m.group(2)
            if exp == "2":
                return f"{base} squared"
            if exp == "3":
                return f"{base} cubed"
            return f"{base} to the power of {exp}"

        # Base can be a single letter/number or a closing paren/bracket.
        text = re.sub(r"([A-Za-z0-9\)\]])\s*\^\s*\{\s*([0-9]+)\s*\}", _exp_repl, text)
        text = re.sub(r"([A-Za-z0-9\)\]])\s*\^\s*([0-9]+)", _exp_repl, text)

        # Subscripts: x_{1} / x_1
        text = re.sub(r"([A-Za-z])\s*_\s*\{\s*([A-Za-z0-9]+)\s*\}", r"\1 sub \2", text)
        text = re.sub(r"([A-Za-z])\s*_\s*([A-Za-z0-9]+)", r"\1 sub \2", text)

        # Implicit multiplication between adjacent single-letter variables before we
        # expand operators into words (so we don't accidentally split real words like "by").
        # Target common compact variable forms around exponents, e.g. mc^{2} -> m c squared.
        if "=" in text:
            # If exponent is already verbalized, split the compact variable token.
            text = re.sub(r"\b([A-Za-z])([A-Za-z])\s+(squared|cubed)\b", r"\1 \2 \3", text)
            text = re.sub(r"\b([A-Za-z])([A-Za-z])\s+(to the power of)\b", r"\1 \2 \3", text)

        # Remove leftover braces
        text = text.replace("{", " ").replace("}", " ")

        # Basic operators (keep conservative; '*' and '/' can appear in prose)
        # Equals: allow unary minus on RHS (x=-2) but avoid assignments/paths (PATH=/usr/bin)
        text = re.sub(
            r"(?<=[A-Za-z0-9\)\]])\s*=\s*(?=[A-Za-z0-9\(\[\]-])",
            " equals ",
            text,
        )
        text = re.sub(r"(?<=\w)\s*\+\s*(?=\w)", " plus ", text)
        # Subtraction: treat '-' as minus in equation contexts.
        # We scope this to equation-like text (this function only runs when math-ish, and typically with '='),
        # so we don't clobber prose ranges like "5-10".
        text = re.sub(r"(?<=\w)\s*-\s*(?=\w)", " minus ", text)
        # Handle leading negatives like "x=-2" -> "x equals minus 2"
        text = re.sub(r"\bequals\s*-\s*(\d+)\b", r"equals minus \1", text)
        # Multiplication when explicitly marked
        text = re.sub(r"(?<=\w)\s*\*\s*(?=\w)", " times ", text)
        # Division only when at least one side is numeric (avoid and/or, path/to, etc.)
        text = re.sub(r"(\d)\s*/\s*(\w)", r"\1 divided by \2", text)
        text = re.sub(r"(\w)\s*/\s*(\d)", r"\1 divided by \2", text)
        # Also allow single-letter variable division in equation contexts: a/b -> a divided by b
        text = re.sub(r"\b([A-Za-z])\s*/\s*([A-Za-z])\b", r"\1 divided by \2", text)

        return text

    def _year_to_words(self, year: int) -> str:
        """
        Convert a year to spoken form.
        Preferences:
        - 19xx => 'nineteen ninety-nine' (and 1905 => 'nineteen oh five')
        - 20xx => 'two thousand and twelve'
        - Other years => standard cardinal number (857 => 'eight hundred and fifty-seven', 1278 => 'one thousand...')
        """
        try:
            y = int(year)
        except Exception:
            return str(year)

        if y < 0:
            return "minus " + self._year_to_words(abs(y))

        # 0-999: treat as cardinal (works for "year 857")
        if 0 <= y <= 999:
            return self._number_to_words(y)

        # 1000-1899: treat as cardinal (1278 => "one thousand two hundred and seventy-eight")
        if 1000 <= y <= 1899:
            return self._number_to_words(y)

        # 1900-1999: "nineteen ninety-nine" style
        if 1900 <= y <= 1999:
            last_two = y % 100
            if last_two == 0:
                return "nineteen hundred"
            if last_two < 10:
                return "nineteen oh " + self._number_to_words(last_two)
            return "nineteen " + self._number_to_words(last_two)

        # 2000-2099: "two thousand and twelve" style
        if 2000 <= y <= 2099:
            last_two = y % 100
            if last_two == 0:
                return "two thousand"
            return "two thousand and " + self._number_to_words(last_two)

        # Default: cardinal
        return self._number_to_words(y)

    def _should_expand_numeric_token(self, text: str, start: int, end: int) -> bool:
        """
        Heuristic: avoid expanding numbers that look like part numbers / versions (adjacent letters).
        """
        left = text[start - 1] if start > 0 else ""
        right = text[end] if end < len(text) else ""
        if left.isalpha() or right.isalpha():
            return False
        return True

    def _verbalize_ranges(self, text: str) -> str:
        """
        Convert numeric ranges like '5-10' or '2012 - 2014' to '5 to 10' / '2012 to 2014'.
        Skips ISO dates like '2026-01-22'.
        """
        range_re = re.compile(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)(?![A-Za-z])")

        def repl(m: re.Match) -> str:
            # Skip if this looks like the start of an ISO date: YYYY-MM-DD
            a = m.group(1)
            b = m.group(2)
            after = text[m.end():]
            if len(a) == 4 and len(b) == 2 and after.startswith("-") and len(after) >= 3 and after[1:3].isdigit():
                return m.group(0)
            if not self._should_expand_numeric_token(text, m.start(1), m.end(1)):
                return m.group(0)
            if not self._should_expand_numeric_token(text, m.start(2), m.end(2)):
                return m.group(0)
            return f"{a} to {b}"

        return range_re.sub(repl, text)

    def _verbalize_decimals(self, text: str) -> str:
        """
        Convert decimals like '3.14' to 'three point one four'.
        Skips semver-ish patterns like '2.1.3'.
        """
        # Also skip decimals that are in the middle of a multi-dot number (e.g. the "1.3" in "2.1.3")
        dec_re = re.compile(r"(?<![A-Za-z])(?<!\d\.)(\d+)\.(\d+)(?![A-Za-z])(?!(?:\.\d))")

        def repl(m: re.Match) -> str:
            if not self._should_expand_numeric_token(text, m.start(1), m.end(2)):
                return m.group(0)
            int_part = int(m.group(1))
            frac = m.group(2)
            return f"{self._number_to_words(int_part)} point {self._digits_to_words(frac)}"

        return dec_re.sub(repl, text)

    def _verbalize_year_like_numbers(self, text: str) -> str:
        """
        Convert year-ish numbers to preferred spoken forms.
        - Contextual 1-4 digit years after words like 'year', 'in', 'AD', etc.
        - Standalone 4-digit years (1000-2099) are also treated as years.
        """
        # Context words: common in narration
        ctx_re = re.compile(
            r"\b(in|year|since|from|around|circa|c\.|ad|a\.d\.|bc|b\.c\.)\s+(\d{1,4})\b",
            re.IGNORECASE,
        )

        def repl_ctx(m: re.Match) -> str:
            prefix = m.group(1)
            y = int(m.group(2))
            return f"{prefix} {self._year_to_words(y)}"

        text = ctx_re.sub(repl_ctx, text)

        # Standalone 4-digit years: 1000-2099
        standalone_re = re.compile(r"\b(1\d{3}|20\d{2})\b")

        def repl_standalone(m: re.Match) -> str:
            if not self._should_expand_numeric_token(text, m.start(1), m.end(1)):
                return m.group(0)
            y = int(m.group(1))
            if 1000 <= y <= 2099:
                return self._year_to_words(y)
            return self._number_to_words(y)

        return standalone_re.sub(repl_standalone, text)

    def _verbalize_plain_integers(self, text: str) -> str:
        """
        Convert remaining standalone integers (1-4 digits) to words.
        This improves general number pronunciation without exploding IDs.
        """
        int_re = re.compile(r"\b(\d{1,4})\b")

        def repl(m: re.Match) -> str:
            if not self._should_expand_numeric_token(text, m.start(1), m.end(1)):
                return m.group(0)
            token = m.group(1)
            # Skip leading-zero numbers (often times/codes) like 05, 007
            if len(token) > 1 and token.startswith("0"):
                return token
            n = int(m.group(1))
            return self._number_to_words(n)

        return int_re.sub(repl, text)
    
    def normalize_numbers(self, text: str) -> str:
        """Convert numeric patterns into spoken-friendly words for TTS."""
        if not text:
            return text

        # Protect patterns we do NOT want to touch (dates, semantic versions).
        protected: Dict[str, str] = {}
        protect_idx = 0

        def _protect(pattern: str, label: str) -> None:
            nonlocal text, protect_idx

            rx = re.compile(pattern)

            def _repl(m: re.Match) -> str:
                nonlocal protect_idx
                key = f"__{label}_{protect_idx}__"
                protect_idx += 1
                protected[key] = m.group(0)
                return key

            text = rx.sub(_repl, text)

        # ISO dates like 2026-01-22
        _protect(r"\b\d{4}-\d{2}-\d{2}\b", "PROTECTED_DATE")
        # Semantic versions like 2.1.3 or v2.1.3
        _protect(r"\b[vV]?\d+(?:\.\d+){2,}\b", "PROTECTED_VER")

        # Normalize thousands separators early (1,234 -> 1234)
        text = re.sub(r"\b(\d{1,3}),(\d{3})\b", r"\1\2", text)

        # Handle ordinals (21st -> 21)
        text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", text)

        # Handle phone numbers (basic format)
        text = re.sub(r"\b(\d{3})-(\d{3})-(\d{4})\b", r"\1 \2 \3", text)

        # Apply configured number patterns (currency, percent, temperature, time)
        # Expand currency to also accept decimals: $12.50 -> 12.50 dollars
        text = re.sub(r"\$(\d+(?:\.\d+)?)", r"\1 dollars", text)

        for pattern, replacement in self.number_patterns:
            # Skip the old $ pattern since we handle it above with decimal support
            if pattern == r"\$(\d+)":
                continue
            text = re.sub(pattern, replacement, text) if not callable(replacement) else re.sub(pattern, replacement, text)

        # Ranges and decimals before integer conversion
        text = self._verbalize_ranges(text)
        text = self._verbalize_decimals(text)

        # Year reading
        text = self._verbalize_year_like_numbers(text)

        # Remaining short integers (1-4 digits)
        text = self._verbalize_plain_integers(text)

        # Restore protected substrings
        for key, val in protected.items():
            text = text.replace(key, val)

        return text
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations"""
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Check for exact matches first
            if word in self.abbreviations:
                expanded_words.append(self.abbreviations[word])
            else:
                # Check for case-insensitive matches
                word_lower = word.lower()
                if word_lower in self.abbreviations:
                    expanded_words.append(self.abbreviations[word_lower])
                else:
                    expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def clean_spacing_and_punctuation(self, text: str) -> str:
        """Fix spacing and punctuation issues"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix space before punctuation
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        
        # Fix space after punctuation
        text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([,:;])([A-Za-z])', r'\1 \2', text)
        
        # Remove problematic sequences
        for problematic, replacement in self.problematic_sequences.items():
            text = text.replace(problematic, replacement)
        
        # Fix quotes spacing
        text = re.sub(r'\s*"\s*([^"]*)\s*"\s*', r' "\1" ', text)
        text = re.sub(r"\s*'\s*([^']*)\s*'\s*", r" '\1' ", text)
        
        return text.strip()
    
    def remove_markup_and_formatting(self, text: str) -> str:
        """Remove HTML, markdown, and other markup"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'__(.*?)__', r'\1', text)      # Bold
        text = re.sub(r'_(.*?)_', r'\1', text)        # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
        
        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [text](url)
        text = re.sub(r'https?://[^\s]+', '', text)           # URLs
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        return text

    def _expand_contractions_and_possessives(self, text: str) -> str:
        """
        Normalize possessives (Carl's -> Carls) and remove apostrophes.
        This avoids the model spelling out apostrophes as separate tokens ("Carl s").
        """
        if not text:
            return text

        # Possessives: Carl's -> Carls, boys' -> boys
        # Remove apostrophe to prevent the model from pronouncing it as a separate token.
        text = re.sub(r"\b([A-Za-z]+)'s\b", r"\1s", text)
        text = re.sub(r"\b([A-Za-z]+)s'\b", r"\1s", text)

        # Any remaining apostrophes inside words: rock'n'roll -> rocknroll
        # This is intentionally aggressive to prevent "X s" spelling artifacts.
        text = re.sub(r"(?<=\w)'(?=\w)", "", text)

        return text
    
    def _verbalize_urls(self, text: str) -> str:
        """
        Convert URLs/domains to a format that will be pronounced correctly.
        Examples:
        - "minstraly.com" -> "minstraly dot com"
        - "minstraly.us" -> "minstraly dot U S"
        - "example.co.uk" -> "example dot C O dot U K"
        """
        if not text:
            return text
        
        # Common TLDs that should be spelled out letter by letter (2-3 letter country codes, etc.)
        # For longer TLDs like .com, .org, .net, we keep them as words
        short_tlds = {
            'us', 'uk', 'io', 'ai', 'tv', 'co', 'cc', 'me', 'ly', 'to', 'be', 'de', 'fr', 
            'it', 'es', 'nl', 'se', 'no', 'dk', 'fi', 'pl', 'cz', 'at', 'ch', 'ie', 'au',
            'nz', 'jp', 'kr', 'cn', 'in', 'ru', 'br', 'mx', 'ar', 'cl', 'za', 'ae', 'sa'
        }
        
        # Pattern to match domain names: word characters, dots, and TLD
        # Matches patterns like: example.com, minstraly.us, subdomain.example.co.uk
        def replace_url(match):
            domain = match.group(0)
            
            # Split by dots
            parts = domain.split('.')
            
            # Process each part
            result_parts = []
            for i, part in enumerate(parts):
                if i > 0:  # Add "dot" before each part after the first
                    result_parts.append('dot')
                
                # Check if this part is a short TLD (regardless of position)
                # This handles cases like "example.co.uk" where both .co and .uk should be spelled out
                if part.lower() in short_tlds:
                    # Spell out short TLDs letter by letter: "co" -> "C O", "us" -> "U S"
                    spelled = ' '.join(part.upper())
                    result_parts.append(spelled)
                else:
                    # Keep domain parts as-is
                    result_parts.append(part)
            
            return ' '.join(result_parts)
        
        # Match domain patterns: word characters, dots, ending with a TLD
        # This matches: example.com, minstraly.us, subdomain.example.co.uk
        # Uses word boundaries to avoid matching within words
        # Pattern matches domains with at least one letter (to avoid matching pure numbers like "3.14")
        # Domain parts can be 1-63 characters
        pattern = r'\b(?=[a-zA-Z0-9]*[a-zA-Z])(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
        text = re.sub(pattern, replace_url, text)
        
        return text
    
    def _normalize_typographic_punctuation(self, text: str) -> str:
        """
        Replace typographic/smart punctuation with ASCII equivalents.
        Use this before validation so we replace rather than throw for
        common formatting issues like curly quotes/apostrophes.
        """
        if not text:
            return text
        for old_char, new_char in self.unicode_replacements.items():
            text = text.replace(old_char, new_char)
        return text

    def validate_text_for_language(self, text: str, language: str = 'en') -> Tuple[bool, Optional[str], Optional[List[str]]]:
        """
        Validate text for language-specific character support.
        
        Before validation, typographic punctuation (e.g. ' → ', " → ") is
        normalized to ASCII equivalents so we replace rather than throw.
        
        :param text: Text to validate
        :param language: Language code (e.g., 'en', 'da', 'no', 'sv')
        :return: Tuple of (is_valid, error_message, disallowed_chars)
                 - is_valid: True if text contains only allowed characters
                 - error_message: Human-readable error message if invalid, None if valid
                 - disallowed_chars: List of disallowed characters found, None if valid
        """
        if not text:
            return True, None, None
        
        # Normalize typographic punctuation first (replace ' with ', etc.)
        # so we don't throw for wrongly formatted characters we can fix
        text = self._normalize_typographic_punctuation(text)
        
        # Normalize language code
        language = language.lower().strip() if language else 'en'
        
        # Get allowed character set for language, default to English
        lang_config = self.LANGUAGE_ALLOWED_CHARS.get(language, self.LANGUAGE_ALLOWED_CHARS['en'])
        allowed_chars = lang_config['allowed']
        lang_description = lang_config['description']
        
        # Find disallowed characters
        disallowed_chars = []
        for char in text:
            char_ord = ord(char)
            # Skip whitespace and control characters (we handle these separately)
            if char_ord < 32:
                continue
            if char_ord not in allowed_chars:
                if char not in disallowed_chars:
                    disallowed_chars.append(char)
        
        if disallowed_chars:
            # Create user-friendly error message
            unique_chars = sorted(set(disallowed_chars))
            char_list = ', '.join(f"'{c}'" for c in unique_chars[:10])  # Show first 10
            if len(unique_chars) > 10:
                char_list += f" and {len(unique_chars) - 10} more"
            
            error_msg = (
                f"Text contains characters not supported for {lang_description}. "
                f"Disallowed characters found: {char_list}. "
                f"Please remove these characters or use a different language setting."
            )
            return False, error_msg, unique_chars
        
        return True, None, None
    
    def deep_clean(self, text: str) -> str:
        """Comprehensive text cleaning pipeline"""
        if not text or not text.strip():
            return "You need to add some text for me to talk."
        
        logger.debug(f"🧹 Starting deep clean for {len(text)} characters")
        
        # 1. Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # 2. Remove markup and formatting
        text = self.remove_markup_and_formatting(text)
        
        # 3. Replace problematic Unicode characters
        for old_char, new_char in self.unicode_replacements.items():
            text = text.replace(old_char, new_char)

        # 3.1 Expand contractions / normalize apostrophes before number & spacing normalization
        text = self._expand_contractions_and_possessives(text)

        # 3.15 Verbalize URLs/domains (e.g. minstraly.com -> minstraly dot com)
        text = self._verbalize_urls(text)

        # 3.25 Light equation verbalization (e.g. E=mc^{2})
        text = self._verbalize_simple_equations(text)
        
        # 4. Normalize numbers and special formats
        text = self.normalize_numbers(text)
        
        # 5. Expand abbreviations
        text = self.expand_abbreviations(text)
        
        # 6. Clean spacing and punctuation
        text = self.clean_spacing_and_punctuation(text)
        
        # 7. Remove story break markers (they're handled separately for pause timing)
        text = text.replace('<STORY_BREAK>', '')
        
        # 8. Final normalization
        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Ensure proper sentence ending
        text = text.rstrip()
        sentence_enders = {'.', '!', '?'}
        if not any(text.endswith(p) for p in sentence_enders):
            text += '.'
        
        # Remove any remaining problematic characters
        # Keep only printable ASCII plus common extended characters
        text = ''.join(char for char in text if ord(char) < 127 or char in 'áéíóúàèìòùâêîôûäëïöüñç')
        
        logger.debug(f"🧹 Deep clean completed: {len(text)} characters")
        
        return text.strip()

