"""Adaptive parameter management based on content analysis."""
import logging
from typing import Dict

from ..chunking.types import ContentType, ChunkInfo

logger = logging.getLogger(__name__)


class AdaptiveParameterManager:
    """Manages adaptive parameters based on content analysis"""
    
    # Base parameter profiles for different content types
    CONTENT_PROFILES = {
        ContentType.DIALOGUE: {
            "temperature": 0.8,         # Slightly higher for dialogue expressiveness
            "exaggeration": 0.75,        # More expression for dialogue
            "cfg_weight": 0.55,          # Slightly lower for more natural flow
            "repetition_penalty": 1.2,   # Standard
            "min_p": 0.05,
            "top_p": 0.9,                # Slightly lower for more focused sampling
        },
        ContentType.NARRATIVE: {
            "temperature": 0.7,          # Balanced for narrative flow
            "exaggeration": 0.55,         # Moderate expression for storytelling
            "cfg_weight": 0.6,           # Good adherence for consistency
            "repetition_penalty": 1.2,
            "min_p": 0.05,
            "top_p": 0.92,               # Slightly higher for narrative variety
        },
        ContentType.DESCRIPTIVE: {
            "temperature": 0.68,         # Slightly higher for descriptive richness
            "exaggeration": 0.45,        # Moderate expression for descriptions
            "cfg_weight": 0.58,          # Balanced adherence
            "repetition_penalty": 1.15,  # Slightly lower to allow more natural flow
            "min_p": 0.05,
            "top_p": 0.94,               # Higher for descriptive variety
        },
        ContentType.TRANSITION: {
            "temperature": 0.72,         # Slightly higher for smooth transitions
            "exaggeration": 0.5,         # Moderate expression
            "cfg_weight": 0.55,          # Slightly lower for natural flow
            "repetition_penalty": 1.18,  # Lower for smoother transitions
            "min_p": 0.05,
            "top_p": 0.93,               # Higher for transition variety
        }
    }
    
    def __init__(self):
        self.complexity_adjustments = {
            "temperature": {
                "high_complexity": -0.1,    # Lower temp for complex text
                "low_complexity": 0.05      # Higher temp for simple text
            },
            "exaggeration": {
                "high_complexity": -0.1,    # Less exaggeration for complex content
                "low_complexity": 0.1       # More exaggeration for simple content
            },
            "cfg_weight": {
                "high_complexity": 0.1,     # Higher CFG for complex text clarity
                "low_complexity": -0.05     # Lower CFG for simple text creativity
            }
        }
        # Intro boost to make the first sentence more engaging
        self.enable_intro_boost = True
        self.intro_exaggeration_boost = 0.2   # +0.2 on first chunk
        self.intro_temperature_boost = 0.05   # +0.05 on first chunk
        self.intro_cfg_weight_factor = 0.9    # reduce CFG a bit for more expression (not below min)
        self.intro_boost_max_words = 35       # apply full boost only if first chunk is short
        # New safety guards for first-chunk stability
        self.intro_min_words_for_boost = 12   # disable boost for extremely short openers
        self.first_chunk_exaggeration_cap = 0.7
        self.first_chunk_min_cfg_weight = 0.5

        # Opener preset (stability over creativity)
        self.enable_opener_preset = True
        self.opener_temperature = 0.62
        self.opener_cfg_weight = 0.7
        self.opener_exaggeration = 0.35
        self.opener_top_p = 0.9
        self.opener_min_p = 0.05
        self.opener_repetition_penalty = 1.18
    
    def get_adaptive_parameters(self, chunk_info: ChunkInfo) -> Dict:
        """Get optimized parameters for a specific chunk"""
        
        # Start with base profile for content type
        params = self.CONTENT_PROFILES[chunk_info.content_type].copy()
        
        # Apply complexity adjustments
        if chunk_info.complexity_score > 6:  # High complexity
            params["temperature"] += self.complexity_adjustments["temperature"]["high_complexity"]
            params["exaggeration"] += self.complexity_adjustments["exaggeration"]["high_complexity"] 
            params["cfg_weight"] += self.complexity_adjustments["cfg_weight"]["high_complexity"]
        elif chunk_info.complexity_score < 3:  # Low complexity
            params["temperature"] += self.complexity_adjustments["temperature"]["low_complexity"]
            params["exaggeration"] += self.complexity_adjustments["exaggeration"]["low_complexity"]
            params["cfg_weight"] += self.complexity_adjustments["cfg_weight"]["low_complexity"]
        
        # Apply positional adjustments
        if chunk_info.is_first_chunk:
            # Original behavior favored stability; instead, apply a gentle intro boost for engagement.
            # For long first chunks, avoid raising temperature to prevent over-energizing long passages.
            if self.enable_intro_boost:
                # No boost for extremely short openers
                if chunk_info.word_count < self.intro_min_words_for_boost:
                    params["temperature"] = params.get("temperature", 0.8)
                    params["exaggeration"] = min(params.get("exaggeration", 0.5), self.first_chunk_exaggeration_cap)
                    params["cfg_weight"] = max(self.first_chunk_min_cfg_weight, params.get("cfg_weight", 0.5))
                elif chunk_info.word_count <= self.intro_boost_max_words:
                    params["temperature"] = max(0.5, min(1.2, params.get("temperature", 0.8) + self.intro_temperature_boost))
                    params["exaggeration"] = max(0.1, min(self.first_chunk_exaggeration_cap, params.get("exaggeration", 0.5) + self.intro_exaggeration_boost))
                    # Do not allow CFG to fall below the minimum for stability
                    boosted_cfg = params.get("cfg_weight", 0.5) * self.intro_cfg_weight_factor
                    params["cfg_weight"] = max(self.first_chunk_min_cfg_weight, boosted_cfg)
                else:
                    # Long first chunk: keep temperature unchanged, only a light expressiveness bump
                    params["exaggeration"] = max(0.1, min(self.first_chunk_exaggeration_cap, params.get("exaggeration", 0.5) + min(0.1, self.intro_exaggeration_boost * 0.5)))
                    params["cfg_weight"] = max(self.first_chunk_min_cfg_weight, params.get("cfg_weight", 0.5))

            # Apply opener preset ONLY for short openers; avoid clamping a long first chunk
            if self.enable_opener_preset and (chunk_info.word_count <= self.intro_boost_max_words or chunk_info.char_count <= 220):
                params["temperature"] = min(params.get("temperature", 0.8), self.opener_temperature)
                params["cfg_weight"] = max(params.get("cfg_weight", 0.5), self.opener_cfg_weight)
                params["exaggeration"] = min(params.get("exaggeration", 0.5), self.opener_exaggeration)
                params["top_p"] = min(params.get("top_p", 1.0), self.opener_top_p)
                params["min_p"] = max(params.get("min_p", 0.05), self.opener_min_p)
                params["repetition_penalty"] = max(params.get("repetition_penalty", 1.2), self.opener_repetition_penalty)
        else:
            # Gentle easing for the second chunk to avoid sudden jump after opener
            # Chunk IDs are sequential; id==1 typically indicates the chunk after opener.
            if chunk_info.id == 1:
                params["temperature"] = min(params.get("temperature", 0.8), max(0.58, self.opener_temperature + 0.05))
                params["exaggeration"] = min(params.get("exaggeration", 0.5), self.first_chunk_exaggeration_cap - 0.1)
                params["cfg_weight"] = max(params.get("cfg_weight", 0.5), max(self.first_chunk_min_cfg_weight, self.opener_cfg_weight - 0.02))
        
        if chunk_info.is_last_chunk:
            params["exaggeration"] *= 0.9      # Slightly calmer ending
        
        # Apply length-based adjustments
        if chunk_info.char_count > 500:
            params["repetition_penalty"] *= 1.05   # Prevent repetition in long chunks
        elif chunk_info.char_count < 200:
            params["temperature"] *= 1.05          # More variation in short chunks
        
        # Apply dialogue-specific adjustments
        if chunk_info.dialogue_ratio > 0.1:
            params["exaggeration"] = min(0.8, params["exaggeration"] * 1.15)  # More expression for dialogue
            params["temperature"] = max(0.6, params["temperature"] * 0.98)   # Slightly more consistency
        
        # Apply descriptive-specific adjustments for slower, more narrative delivery
        if chunk_info.content_type == ContentType.DESCRIPTIVE:
            params["temperature"] = max(0.65, params["temperature"] * 0.95)  # Slightly more consistent for descriptions
            params["cfg_weight"] = min(0.7, params["cfg_weight"] * 1.05)     # Slightly stronger adherence
            params["repetition_penalty"] = max(1.1, params["repetition_penalty"] * 0.98)  # Allow more natural repetition
        
        # Clamp all parameters to safe ranges
        params = self._clamp_parameters(params)
        
        logger.debug(f"🎛️ Chunk {chunk_info.id} ({chunk_info.content_type.value}): "
                    f"temp={params['temperature']:.2f}, "
                    f"exag={params['exaggeration']:.2f}, "
                    f"cfg={params['cfg_weight']:.2f}")
        
        return params
    
    def _clamp_parameters(self, params: Dict) -> Dict:
        """Clamp parameters to safe ranges"""
        clamps = {
            "temperature": (0.5, 1.2),
            "exaggeration": (0.1, 1.0),
            "cfg_weight": (0.2, 0.8),
            "repetition_penalty": (1.0, 1.5),
            "min_p": (0.01, 0.1),
            "top_p": (0.8, 1.0)
        }
        
        for param, (min_val, max_val) in clamps.items():
            if param in params:
                params[param] = max(min_val, min(max_val, params[param]))
        
        return params

