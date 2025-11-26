"""Advanced text sanitization for optimal TTS quality."""
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class AdvancedTextSanitizer:
    """Comprehensive text cleaning and normalization for optimal TTS quality"""
    
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
            
            if hour == 0:
                return f"twelve {minute:02d} AM"
            elif hour < 12:
                return f"{hour} {minute:02d} AM"
            elif hour == 12:
                return f"twelve {minute:02d} PM"
            else:
                return f"{hour-12} {minute:02d} PM"
        except:
            return time_str
    
    def normalize_numbers(self, text: str) -> str:
        """Convert numbers to spoken form"""
        # Handle ordinals
        text = re.sub(r'\b(\d+)(st|nd|rd|th)\b', r'\1', text)
        
        # Handle years (1900-2099)
        text = re.sub(r'\b(19|20)(\d{2})\b', 
                     lambda m: f"{m.group(1)} {m.group(2)}" if int(m.group(2)) < 10 
                     else f"{m.group(1)}{m.group(2)[:1]} {m.group(2)[1:]}", text)
        
        # Handle phone numbers (basic format)
        text = re.sub(r'\b(\d{3})-(\d{3})-(\d{4})\b', 
                     r'\1 \2 \3', text)
        
        # Apply number patterns
        for pattern, replacement in self.number_patterns:
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
            else:
                text = re.sub(pattern, replacement, text)
        
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

