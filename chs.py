"""
Coordinate Heart System - Advanced Emotion Analysis

A sophisticated emotion analysis system using coordinate-based representation
for mapping complex emotional states in 2D space with stability metrics.

This system processes emotional data from language models and maps emotions
to coordinates, resolves conflicts between opposing emotions, and generates
human-readable emotion labels including complex emotional combinations.

Author: Omar Al-Desi
License: MIT
Version: 1.0.0
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

__version__ = "1.0.0"
__author__ = "Omat Al-Desi"
__license__ = "MIT"

# --- Constants for Emotion Labeling ---

# Define complex emotion combinations as validated by psychological research
# Two-emotion combinations represent basic emotional blends
# Three-emotion combinations are prioritized as they represent more nuanced states
COMPLEX_EMOTIONS: Dict[str, List[str]] = {
    # Two-emotion combinations - basic emotional blends
    "Bittersweet": ["joy", "sadness"],
    "Contempt": ["disgust", "pride"],
    "Envy": ["sadness", "anger"],
    "Jealousy": ["fear", "anger"],
    "Pity": ["sadness", "disgust"],
    "Nostalgia": ["love", "sadness"],
    "Admiration": ["love", "pride"],
    "Dread": ["fear", "sadness"],
    "Rage": ["anger", "disgust"],
    "Remorse": ["guilt", "sadness"],
    "Embarrassment": ["guilt", "fear"],
    "Resentment": ["anger", "guilt"],
    "Anticipation": ["joy", "fear"],
    "Compassion": ["love", "sadness"],
    "Schadenfreude": ["joy", "disgust"],
    
    # Three-emotion combinations (prioritized) - complex emotional states
    "Shame": ["guilt", "fear", "disgust"],
    "Melancholy": ["sadness", "love", "guilt"],
    "Righteous anger": ["anger", "pride", "disgust"],
    "Mortification": ["guilt", "fear", "anger"],
    "Vindication": ["pride", "joy", "anger"],
    "Heartbreak": ["love", "sadness", "fear"],
    "Betrayal": ["anger", "sadness", "disgust"],
    "Triumph": ["joy", "pride", "love"],
    "Despair": ["sadness", "fear", "guilt"],
    "Loathing": ["disgust", "anger", "fear"],
    "Adoration": ["love", "joy", "pride"],
    "Devastation": ["sadness", "anger", "fear"],
}

# Minimum intensity threshold for an emotion to be considered "active"
# in complex emotion labeling (15% intensity)
EMOTION_ACTIVE_THRESHOLD: float = 0.15

# Default emotion processing order (alphabetical for consistency)
DEFAULT_EMOTION_ORDER: List[str] = [
    "anger", "disgust", "fear", "guilt", 
    "joy", "love", "pride", "sadness"
]

# Default conflicting emotion pairs that cannot coexist at full intensity
DEFAULT_CONFLICTING_PAIRS: List[Tuple[str, str]] = [
    ("joy", "anger"),
    ("guilt", "pride"),
]


@dataclass(frozen=True)
class EmotionCoordinates:
    """
    Static coordinate definitions for emotions in 2D space.
    
    These coordinates are based on psychological models where emotions
    are positioned relative to each other in a meaningful geometric pattern.
    The coordinate system uses:
    - Center (0,0) as neutral/love
    - Positive Y as activating emotions (anger)
    - Negative Y as deactivating emotions (joy, fear, sadness, disgust)
    - X-axis representing internal vs external focus
    """
    
    love: Tuple[float, float] = (0.0, 0.0)      # Neutral center point
    joy: Tuple[float, float] = (0.0, -1.0)      # Pure positive, low activation
    anger: Tuple[float, float] = (0.0, 1.0)     # Pure negative, high activation
    guilt: Tuple[float, float] = (1.0, 0.0)     # Internal focus, self-directed
    pride: Tuple[float, float] = (-1.0, 0.0)    # External focus, self-directed
    fear: Tuple[float, float] = (0.5, -0.866)   # Avoidance-oriented, moderate internal
    sadness: Tuple[float, float] = (0.866, -0.5)  # Withdrawal-oriented, internal
    disgust: Tuple[float, float] = (-0.5, -0.866)  # Rejection-oriented, moderate external


@dataclass
class EmotionIntensities:
    """
    Container for emotion intensity scores.
    
    Each emotion is represented as a float between 0.0 and 1.0,
    where 0.0 indicates no presence and 1.0 indicates maximum intensity.
    """
    
    joy: float = field(default=0.0, metadata={"description": "Happiness, contentment, elation"})
    anger: float = field(default=0.0, metadata={"description": "Frustration, rage, irritation"})
    guilt: float = field(default=0.0, metadata={"description": "Self-blame, remorse, regret"})
    pride: float = field(default=0.0, metadata={"description": "Self-satisfaction, accomplishment"})
    love: float = field(default=0.0, metadata={"description": "Affection, care, attachment"})
    fear: float = field(default=0.0, metadata={"description": "Anxiety, worry, apprehension"})
    sadness: float = field(default=0.0, metadata={"description": "Sorrow, grief, melancholy"})
    disgust: float = field(default=0.0, metadata={"description": "Revulsion, aversion, contempt"})

    def __post_init__(self) -> None:
        """Validate that all intensities are within valid range [0.0, 1.0]."""
        for emotion_name, intensity in self.to_dict().items():
            if not (0.0 <= intensity <= 1.0):
                raise ValueError(
                    f"Emotion intensity for '{emotion_name}' must be between 0.0 and 1.0, "
                    f"got {intensity}"
                )

    def to_dict(self) -> Dict[str, float]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dict mapping emotion names to their intensity values.
        """
        return asdict(self)

    def to_list(self, order: List[str]) -> List[float]:
        """
        Convert to list using specified emotion order.
        
        Args:
            order: List of emotion names in desired order.
            
        Returns:
            List of intensity values in the specified order.
            
        Raises:
            AttributeError: If an emotion in the order doesn't exist.
        """
        return [getattr(self, emotion) for emotion in order]

    def get_total_intensity(self) -> float:
        """Calculate the sum of all emotion intensities."""
        return sum(self.to_dict().values())

    def get_dominant_emotion(self) -> Optional[str]:
        """
        Get the emotion with the highest intensity.
        
        Returns:
            Name of the dominant emotion, or None if all intensities are 0.0.
        """
        intensities = self.to_dict()
        if not any(intensities.values()):
            return None
        return max(intensities.items(), key=lambda x: x[1])[0]


@dataclass
class ContextualDrain:
    """
    Represents contextual factors that affect emotional stability.
    
    These factors represent external circumstances or internal conditions
    that drain emotional resources and reduce overall stability.
    """
    
    factors: List[str] = field(default_factory=list)
    drain_value: float = field(default=0.0)
    
    def __post_init__(self) -> None:
        """Validate drain value is non-negative."""
        if self.drain_value < 0.0:
            raise ValueError(f"Drain value must be non-negative, got {self.drain_value}")


@dataclass
class EmotionalState:
    """
    Complete representation of an emotional state.
    
    This class encapsulates all aspects of an emotional analysis including:
    - Spatial coordinates in the emotion space
    - Individual emotion intensities
    - Stability metrics and drain factors
    - Human-readable labels and dominant emotions
    - Metadata for tracking and serialization
    """
    
    coordinates: Tuple[float, float]
    intensities: EmotionIntensities
    stability: float
    contextual_drain: ContextualDrain
    conflict_drain: float
    emotional_load_drain: float
    token: str
    raw_llm_data: Dict[str, Any]
    timestamp: str
    dominant_emotion: Optional[str] = None
    emotion_label: str = ""

    def __post_init__(self) -> None:
        """Validate emotional state parameters."""
        # Validate stability is within valid range
        if not (0.0 <= self.stability <= 1.0):
            raise ValueError(f"Stability must be between 0.0 and 1.0, got {self.stability}")
        
        # Validate drain values are non-negative
        if self.conflict_drain < 0.0:
            raise ValueError(f"Conflict drain must be non-negative, got {self.conflict_drain}")
        if self.emotional_load_drain < 0.0:
            raise ValueError(f"Emotional load drain must be non-negative, got {self.emotional_load_drain}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "coordinates": self.coordinates,
            "intensities": self.intensities.to_dict(),
            "stability": self.stability,
            "contextual_drain": {
                "factors": self.contextual_drain.factors,
                "drain_value": self.contextual_drain.drain_value,
            },
            "conflict_drain": self.conflict_drain,
            "emotional_load_drain": self.emotional_load_drain,
            "token": self.token,
            "raw_llm_data": self.raw_llm_data,
            "timestamp": self.timestamp,
            "dominant_emotion": self.dominant_emotion,
            "emotion_label": self.emotion_label,
        }

    def is_stable(self, threshold: float = 0.6) -> bool:
        """
        Check if the emotional state is considered stable.
        
        Args:
            threshold: Minimum stability value to be considered stable.
            
        Returns:
            True if stability is above threshold.
        """
        return self.stability >= threshold

    def get_active_emotions(self, threshold: float = EMOTION_ACTIVE_THRESHOLD) -> List[Tuple[str, float]]:
        """
        Get emotions that are above the activity threshold.
        
        Args:
            threshold: Minimum intensity to be considered active.
            
        Returns:
            List of (emotion_name, intensity) tuples sorted by intensity descending.
        """
        active = [
            (emotion, intensity)
            for emotion, intensity in self.intensities.to_dict().items()
            if intensity >= threshold
        ]
        return sorted(active, key=lambda x: x[1], reverse=True)


class CoordinateHeartSystem:
    """
    Advanced emotion analysis system using coordinate-based representation.
    
    This system processes emotional data from language models and maps emotions
    to a 2D coordinate space, resolves conflicts between opposing emotions,
    calculates stability metrics, and generates human-readable emotion labels
    including complex emotional combinations.
    
    The system supports:
    - Emotion intensity analysis and conflict resolution
    - Spatial coordinate mapping in 2D emotion space
    - Psychological stability calculation with multiple drain factors
    - Complex emotion labeling (e.g., "Bittersweet", "Shame", "Triumph")
    - State encoding/decoding for persistence and smoothing
    - Temporal smoothing based on previous emotional states
    
    Example:
        >>> system = CoordinateHeartSystem()
        >>> llm_response = {
        ...     "emotions": {"joy": 0.7, "sadness": 0.3},
        ...     "contextual_drain": {"factors": ["stress"], "drain_value": 0.1}
        ... }
        >>> state = system.analyze_emotions(llm_response)
        >>> print(state.emotion_label)  # "Bittersweet (Stable/Functional)"
    """

    def __init__(
        self,
        emotion_order: Optional[List[str]] = None,
        conflicting_pairs: Optional[List[Tuple[str, str]]] = None,
        emotional_capacity: float = 1.0,
        epsilon: float = 1e-3,
    ) -> None:
        """
        Initialize the Coordinate Heart System.

        Args:
            emotion_order: Order for processing emotions. Defaults to alphabetical order.
            conflicting_pairs: Pairs of mutually exclusive emotions that create conflict.
            emotional_capacity: Maximum sustainable total emotional intensity before overload.
            epsilon: Small constant to prevent division by zero in calculations.
        """
        self.coordinates = EmotionCoordinates()
        self.emotion_order = emotion_order or DEFAULT_EMOTION_ORDER.copy()
        self.conflicting_pairs = conflicting_pairs or DEFAULT_CONFLICTING_PAIRS.copy()
        self.emotional_capacity = emotional_capacity
        self.epsilon = epsilon

        # Validate emotion order contains all required emotions
        required_emotions = set(DEFAULT_EMOTION_ORDER)
        provided_emotions = set(self.emotion_order)
        if not required_emotions.issubset(provided_emotions):
            missing = required_emotions - provided_emotions
            raise ValueError(f"Missing required emotions in emotion_order: {missing}")

        # Create coordinate mapping for efficient lookups
        self.coord_map: Dict[str, Tuple[float, float]] = {
            emotion: getattr(self.coordinates, emotion)
            for emotion in self.emotion_order
        }

    def analyze_emotions(
        self,
        llm_response: Dict[str, Any],
        last_analysis_token: Optional[str] = None,
        token_everything: bool = False,
    ) -> EmotionalState:
        """
        Analyze emotions from LLM response with optional temporal smoothing.

        This is the main entry point for emotion analysis. The method:
        1. Parses raw LLM response data
        2. Applies temporal smoothing if previous state is provided
        3. Resolves emotional conflicts
        4. Calculates spatial coordinates and stability metrics
        5. Generates human-readable emotion labels
        6. Creates encoded token for state persistence

        Args:
            llm_response: Dictionary containing emotions and contextual_drain data.
            last_analysis_token: Token from previous analysis for temporal smoothing.
            token_everything: Whether to create full token (True) or compact 11-byte token (False).

        Returns:
            Complete EmotionalState with all analysis results.

        Raises:
            KeyError: If required keys are missing from llm_response.
            ValueError: If emotion values are invalid.
        """
        # Step 1: Parse the raw LLM response for current input
        current_intensities = self._parse_intensities(llm_response)
        contextual_drain = self._parse_contextual_drain(llm_response)

        # Step 2: Apply temporal smoothing if previous state is available
        final_intensities = self._apply_temporal_smoothing(
            current_intensities, last_analysis_token
        )

        # Step 3: Run core emotion analysis pipeline
        resolved_intensities, conflict_drain = self._resolve_conflicts(final_intensities)
        coordinates = self._calculate_coordinates(resolved_intensities)
        stability, emotional_load_drain = self._calculate_stability(
            resolved_intensities, conflict_drain, contextual_drain.drain_value
        )

        # Step 4: Create initial emotional state (before token and labels)
        state = EmotionalState(
            coordinates=coordinates,
            intensities=resolved_intensities,
            stability=stability,
            contextual_drain=contextual_drain,
            conflict_drain=conflict_drain,
            emotional_load_drain=emotional_load_drain,
            token="",  # Will be generated after labels are set
            raw_llm_data=llm_response,
            timestamp=datetime.now().isoformat(),
        )

        # Step 5: Generate emotion labels and identify dominant emotion
        state.dominant_emotion = self._get_dominant_emotion(state.intensities)
        state.emotion_label = self._get_emotion_label(state)

        # Step 6: Encode final token with complete state information
        state.token = self._encode_token(state, token_everything)

        return state

    def _parse_intensities(self, llm_response: Dict[str, Any]) -> EmotionIntensities:
        """
        Parse emotion intensities from LLM response.
        
        Args:
            llm_response: Dictionary containing 'emotions' key with emotion intensities.
            
        Returns:
            EmotionIntensities object with parsed values.
        """
        emotions_data = llm_response.get("emotions", {})
        
        # Extract intensities for each emotion, defaulting to 0.0 if not present
        intensity_values = {
            emotion: emotions_data.get(emotion, 0.0)
            for emotion in self.emotion_order
        }
        
        return EmotionIntensities(**intensity_values)

    def _parse_contextual_drain(self, llm_response: Dict[str, Any]) -> ContextualDrain:
        """
        Parse contextual drain information from LLM response.
        
        Args:
            llm_response: Dictionary containing 'contextual_drain' key.
            
        Returns:
            ContextualDrain object with parsed factors and drain value.
        """
        contextual_data = llm_response.get("contextual_drain", {})
        return ContextualDrain(
            factors=contextual_data.get("factors", []),
            drain_value=contextual_data.get("drain_value", 0.0),
        )

    def _apply_temporal_smoothing(
        self, current_intensities: EmotionIntensities, last_token: Optional[str]
    ) -> EmotionIntensities:
        """
        Apply temporal smoothing between current and previous emotional states.
        
        Smoothing helps prevent rapid emotional oscillations by blending
        the current state with the previous state based on their relative stabilities.
        
        Args:
            current_intensities: Current emotion intensities from LLM.
            last_token: Token from previous emotional state.
            
        Returns:
            Smoothed emotion intensities.
        """
        if not last_token:
            return current_intensities

        try:
            previous_state = self.decode_token(last_token)
        except ValueError:
            # If token is invalid, proceed without smoothing
            return current_intensities

        # Calculate stability of current state to determine blend weight
        resolved_current, conflict_drain_current = self._resolve_conflicts(current_intensities)
        stability_current, _ = self._calculate_stability(
            resolved_current, conflict_drain_current, 0.0
        )

        # Calculate blend weight based on relative stabilities
        # Higher stability of previous state increases its influence
        previous_stability = previous_state.stability
        current_stability = stability_current

        weight = (previous_stability + self.epsilon) / (
            previous_stability + current_stability + 2 * self.epsilon
        )

        # Blend intensities using calculated weight
        previous_intensities = previous_state.intensities.to_dict()
        current_intensities_dict = current_intensities.to_dict()

        smoothed_values = {
            emotion: weight * previous_intensities[emotion] + (1 - weight) * current_intensities_dict[emotion]
            for emotion in self.emotion_order
        }

        return EmotionIntensities(**smoothed_values)

    def _resolve_conflicts(
        self, intensities: EmotionIntensities
    ) -> Tuple[EmotionIntensities, float]:
        """
        Resolve conflicts between mutually exclusive emotions.
        
        When conflicting emotions (e.g., joy and anger) are both present,
        the system reduces both by the minimum intensity value, representing
        the psychological impossibility of fully experiencing opposing emotions.
        
        Args:
            intensities: Raw emotion intensities before conflict resolution.
            
        Returns:
            Tuple of (resolved_intensities, total_conflict_drain).
        """
        # Create a copy to avoid modifying the original
        resolved_dict = intensities.to_dict()
        total_conflict_drain = 0.0

        # Process each conflicting pair
        for emotion1, emotion2 in self.conflicting_pairs:
            intensity1 = resolved_dict[emotion1]
            intensity2 = resolved_dict[emotion2]

            # Calculate conflict amount (minimum of the two intensities)
            conflict_amount = min(intensity1, intensity2)
            
            if conflict_amount > 0:
                # Reduce both emotions by the conflict amount
                resolved_dict[emotion1] = intensity1 - conflict_amount
                resolved_dict[emotion2] = intensity2 - conflict_amount
                total_conflict_drain += conflict_amount

        return EmotionIntensities(**resolved_dict), total_conflict_drain

    def _calculate_coordinates(self, intensities: EmotionIntensities) -> Tuple[float, float]:
        """
        Calculate final coordinates using sequential pairwise interpolation.
        
        This method maps emotions to a 2D coordinate space by:
        1. Identifying all active emotions (intensity > 0)
        2. For single emotions: scaling coordinate by intensity
        3. For multiple emotions: sequentially blending coordinates based on relative intensities
        
        Args:
            intensities: Resolved emotion intensities after conflict resolution.
            
        Returns:
            Final (x, y) coordinates in emotion space.
        """
        # Get all emotions with non-zero intensity
        active_emotions = [
            (emotion, getattr(intensities, emotion))
            for emotion in self.emotion_order
            if getattr(intensities, emotion) > 0
        ]

        # Handle edge cases
        if not active_emotions:
            return (0.0, 0.0)  # Neutral state

        # Single emotion: scale coordinate by intensity
        if len(active_emotions) == 1:
            emotion, intensity = active_emotions[0]
            base_coord = self.coord_map[emotion]
            return (base_coord[0] * intensity, base_coord[1] * intensity)

        # Multiple emotions: sequential pairwise interpolation
        current_coord = self.coord_map[active_emotions[0][0]]
        current_intensity = active_emotions[0][1]

        for i in range(1, len(active_emotions)):
            next_emotion, next_intensity = active_emotions[i]
            next_coord = self.coord_map[next_emotion]

            # Calculate interpolation weight
            total_intensity = current_intensity + next_intensity
            if total_intensity > 0:
                weight = next_intensity / total_intensity
                
                # Linear interpolation between current and next coordinates
                x = current_coord[0] + weight * (next_coord[0] - current_coord[0])
                y = current_coord[1] + weight * (next_coord[1] - current_coord[1])
                
                current_coord = (x, y)
                current_intensity = total_intensity

        return current_coord

    def _calculate_stability(
        self,
        intensities: EmotionIntensities,
        conflict_drain: float,
        contextual_drain_value: float,
    ) -> Tuple[float, float]:
        """
        Calculate psychological stability and emotional load drain.
        
        Stability represents the individual's ability to maintain emotional equilibrium.
        It's reduced by three factors:
        1. Emotional overload (total intensity exceeding capacity)
        2. Emotional conflicts (opposing emotions)
        3. Contextual stressors (external factors)
        
        Args:
            intensities: Resolved emotion intensities.
            conflict_drain: Amount of stability lost to emotional conflicts.
            contextual_drain_value: Amount of stability lost to external factors.
            
        Returns:
            Tuple of (final_stability, emotional_load_drain).
        """
        total_intensity = intensities.get_total_intensity()
        
        # Calculate emotional load drain (intensity exceeding capacity)
        emotional_load_drain = max(0.0, total_intensity - self.emotional_capacity)

        # Calculate final stability (bounded between 0 and 1)
        stability = 1.0 - emotional_load_drain - conflict_drain - contextual_drain_value
        final_stability = max(0.0, min(1.0, stability))

        return final_stability, emotional_load_drain

    def _get_dominant_emotion(self, intensities: EmotionIntensities) -> Optional[str]:
        """
        Identify the emotion with the highest intensity.
        
        Args:
            intensities: Emotion intensities to analyze.
            
        Returns:
            Name of dominant emotion, or None if all intensities are zero.
        """
        return intensities.get_dominant_emotion()

    def _get_stability_phrase(self, stability: float) -> str:
        """
        Generate descriptive phrase based on stability value.
        
        Args:
            stability: Stability value between 0.0 and 1.0.
            
        Returns:
            Human-readable description of stability level.
        """
        if stability >= 1.0:
            return "Optimal Equilibrium"
        elif stability >= 0.80:
            return "Highly Stable/Resilient"
        elif stability >= 0.60:
            return "Stable/Functional"
        elif stability >= 0.40:
            return "Mildly Stressed/Overwhelmed"
        elif stability >= 0.20:
            return "Unstable/Struggling"
        elif stability > 0.0:
            return "Crisis/Near Shutdown"
        else:  # stability == 0.0
            return "Complete Breakdown/Critical State"

    def _get_emotion_label(self, state: EmotionalState) -> str:
        """
        Generate human-readable emotion label based on intensities and stability.
        
        This method creates intuitive labels by:
        1. Identifying active emotions above threshold
        2. Matching complex emotion patterns (3-emotion, then 2-emotion combinations)
        3. Falling back to dominant or mixed emotion descriptions
        4. Including stability context in all labels
        
        Args:
            state: Complete emotional state to label.
            
        Returns:
            Human-readable emotion label with stability context.
        """
        active_emotions = state.get_active_emotions(EMOTION_ACTIVE_THRESHOLD)
        stability_phrase = self._get_stability_phrase(state.stability)

        # Case 1: Neutral/Optimal state (no active emotions, high stability)
        if not active_emotions and state.stability >= 0.99:
            return f"Neutral State ({stability_phrase})"

        # Case 2: Complete breakdown with no dominant emotions
        if state.stability == 0.0 and not active_emotions:
            return "Complete Emotional Shutdown (Critical State)"

        # Case 3: Try to match complex emotion combinations
        # Sort by number of constituents (3-emotion combinations first)
        sorted_complex_emotions = sorted(
            COMPLEX_EMOTIONS.items(), key=lambda item: len(item[1]), reverse=True
        )

        for label, constituents in sorted_complex_emotions:
            # Check if all constituents are active and among top N emotions
            top_emotions = [e[0] for e in active_emotions[:len(constituents)]]
            
            if (
                all(constituent in top_emotions for constituent in constituents)
                and len(active_emotions) <= len(constituents) + 1
            ):  # Allow one extra low-intensity emotion
                return f"{label} ({stability_phrase})"

        # Case 4: Use dominant emotion or mixed emotion description
        if state.dominant_emotion:
            if len(active_emotions) == 1:
                return f"{state.dominant_emotion.capitalize()} ({stability_phrase})"
            elif len(active_emotions) > 1:
                # List top 2-3 active emotions
                top_emotion_names = [e[0].capitalize() for e in active_emotions[:3]]
                return f"Mixed {', '.join(top_emotion_names)} ({stability_phrase})"

        # Case 5: Fallback for subtle emotional states
        if state.stability < 0.99:
            return f"Subtle Emotional State ({stability_phrase})"

        return f"Undefined Emotional State ({stability_phrase})"

    def _encode_token(self, state: EmotionalState, token_everything: bool = False) -> str:
        """
        Encode emotional state into a token for persistence and transmission.
        
        Two token formats are supported:
        1. Full token: Complete JSON representation (for full state preservation)
        2. Limited token: Compact 11-byte representation (for efficient storage)
        
        Args:
            state: Emotional state to encode.
            token_everything: If True, create full JSON token. If False, create compact token.
            
        Returns:
            Base64-encoded token string.
        """
        if token_everything:
            # Create comprehensive token with all data
            full_data = state.to_dict()
            json_str = json.dumps(full_data, separators=(",", ":"))
            return base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
        else:
            # Create compact 11-byte token
            # Format: [x_coord, y_coord, 8_emotion_intensities, stability]
            x, y = state.coordinates
            
            # Convert coordinates from [-1,1] range to [0,255] byte range
            token_x = round(255 * (x + 1) / 2)
            token_y = round(255 * (y + 1) / 2)

            # Convert intensities to byte values [0,255]
            intensity_list = state.intensities.to_list(self.emotion_order)
            token_intensities = [round(255 * intensity) for intensity in intensity_list]
            
            # Convert stability to byte value [0,255]
            token_stability = round(255 * state.stability)

            # Clamp all values to valid byte range and create token
            byte_values = [
                max(0, min(255, token_x)),
                max(0, min(255, token_y)),
                *[max(0, min(255, val)) for val in token_intensities],
                max(0, min(255, token_stability)),
            ]
            
            token_bytes = bytes(byte_values)
            return base64.b64encode(token_bytes).decode("utf-8")

    def decode_token(self, token: str) -> EmotionalState:
        """
        Decode a token back into an EmotionalState.
        
        Automatically detects token format based on decoded length:
        - 11 bytes: Compact token format
        - Other lengths: Full JSON token format
        
        Args:
            token: Base64-encoded token string.
            
        Returns:
            Reconstructed EmotionalState object.
            
        Raises:
            ValueError: If token format is invalid or cannot be decoded.
        """
        try:
            decoded_data = base64.b64decode(token)

            if len(decoded_data) == 11:
                return self._decode_limited_token(decoded_data, token)
            else:
                return self._decode_full_token(decoded_data, token)

        except Exception as e:
            raise ValueError(f"Failed to decode token: {e}")

    def _decode_limited_token(self, token_bytes: bytes, original_token: str) -> EmotionalState:
        """
        Decode a compact 11-byte token into an EmotionalState.
        
        Limited tokens sacrifice some information for efficiency. The dominant_emotion
        and emotion_label cannot be reliably reconstructed from the compact format
        and are set to generic values.
        
        Args:
            token_bytes: 11 bytes of decoded token data.
            original_token: Original base64 token string.
            
        Returns:
            EmotionalState reconstructed from limited token data.
        """
        # Decode coordinates from byte values back to [-1,1] range
        x = (token_bytes[0] / 255) * 2 - 1
        y = (token_bytes[1] / 255) * 2 - 1

        # Decode emotion intensities from byte values back to [0,1] range
        intensity_values = [token_bytes[i + 2] / 255 for i in range(8)]
        intensities = EmotionIntensities(
            **{emotion: intensity_values[i] for i, emotion in enumerate(self.emotion_order)}
        )
        
        # Decode stability from byte value back to [0,1] range
        stability = token_bytes[10] / 255

        # Create state with placeholder values for missing data
        return EmotionalState(
            coordinates=(x, y),
            intensities=intensities,
            stability=stability,
            contextual_drain=ContextualDrain(factors=[], drain_value=0.0),  # Placeholder
            conflict_drain=0.0,  # Cannot be reconstructed from limited token
            emotional_load_drain=0.0,  # Cannot be reconstructed from limited token
            token=original_token,
            raw_llm_data={},  # Placeholder - original data not preserved
            timestamp="",  # Placeholder - timestamp not preserved
            dominant_emotion=None,  # Cannot reliably determine from limited data
            emotion_label="Decoded from Limited Token",  # Generic label
        )

    def _decode_full_token(self, token_bytes: bytes, original_token: str) -> EmotionalState:
        """
        Decode a full JSON token into an EmotionalState.
        
        Full tokens preserve all state information and can be perfectly reconstructed.
        
        Args:
            token_bytes: Decoded JSON token data.
            original_token: Original base64 token string.
            
        Returns:
            Complete EmotionalState reconstructed from full token.
        """
        json_str = token_bytes.decode("utf-8")
        data = json.loads(json_str)

        # Reconstruct all components from JSON data
        intensities = EmotionIntensities(**data["intensities"])
        contextual_drain = ContextualDrain(
            factors=data["contextual_drain"]["factors"],
            drain_value=data["contextual_drain"]["drain_value"],
        )

        return EmotionalState(
            coordinates=tuple(data["coordinates"]),
            intensities=intensities,
            stability=data["stability"],
            contextual_drain=contextual_drain,
            conflict_drain=data["conflict_drain"],
            emotional_load_drain=data["emotional_load_drain"],
            token=original_token,
            raw_llm_data=data["raw_llm_data"],
            timestamp=data["timestamp"],
            dominant_emotion=data.get("dominant_emotion"),
            emotion_label=data.get("emotion_label", ""),
        )

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the current system configuration.
        
        Returns:
            Dictionary containing system configuration details.
        """
        return {
            "version": __version__,
            "emotion_order": self.emotion_order.copy(),
            "conflicting_pairs": self.conflicting_pairs.copy(),
            "emotional_capacity": self.emotional_capacity,
            "epsilon": self.epsilon,
            "coordinate_mapping": self.coord_map.copy(),
            "complex_emotions_count": len(COMPLEX_EMOTIONS),
            "emotion_active_threshold": EMOTION_ACTIVE_THRESHOLD,
        }

    def validate_llm_response(self, llm_response: Dict[str, Any]) -> bool:
        """
        Validate that an LLM response contains the required structure.
        
        Args:
            llm_response: Response dictionary to validate.
            
        Returns:
            True if response is valid, False otherwise.
        """
        try:
            # Check required top-level keys
            if "emotions" not in llm_response:
                return False
            
            emotions = llm_response["emotions"]
            if not isinstance(emotions, dict):
                return False
            
            # Validate emotion values
            for emotion in self.emotion_order:
                if emotion in emotions:
                    value = emotions[emotion]
                    if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                        return False
            
            # Validate contextual_drain if present
            if "contextual_drain" in llm_response:
                contextual_drain = llm_response["contextual_drain"]
                if not isinstance(contextual_drain, dict):
                    return False
                
                if "factors" in contextual_drain:
                    if not isinstance(contextual_drain["factors"], list):
                        return False
                
                if "drain_value" in contextual_drain:
                    drain_value = contextual_drain["drain_value"]
                    if not isinstance(drain_value, (int, float)) or drain_value < 0:
                        return False
            
            return True
            
        except (KeyError, TypeError, ValueError):
            return False


def create_sample_llm_response(
    emotions: Optional[Dict[str, float]] = None,
    contextual_factors: Optional[List[str]] = None,
    drain_value: float = 0.0
) -> Dict[str, Any]:
    """
    Create a sample LLM response for testing purposes.
    
    Args:
        emotions: Dictionary of emotion intensities. Defaults to neutral state.
        contextual_factors: List of contextual stress factors.
        drain_value: Contextual drain value.
        
    Returns:
        Properly formatted LLM response dictionary.
    """
    if emotions is None:
        emotions = {emotion: 0.0 for emotion in DEFAULT_EMOTION_ORDER}
    
    if contextual_factors is None:
        contextual_factors = []
    
    return {
        "emotions": emotions,
        "contextual_drain": {
            "factors": contextual_factors,
            "drain_value": drain_value
        }
    }


# Example usage and testing functions
def run_example() -> None:
    """
    Demonstrate basic usage of the Coordinate Heart System.
    """
    print(f"Coordinate Heart System v{__version__}")
    print("=" * 50)
    
    # Initialize the system
    system = CoordinateHeartSystem()
    
    # Example 1: Simple joy
    print("\nExample 1: Pure Joy")
    simple_joy = create_sample_llm_response(
        emotions={"joy": 0.8, "love": 0.2},
        contextual_factors=["celebration"],
        drain_value=0.05
    )
    
    state1 = system.analyze_emotions(simple_joy)
    print(f"Emotion Label: {state1.emotion_label}")
    print(f"Coordinates: {state1.coordinates}")
    print(f"Stability: {state1.stability:.2f}")
    print(f"Dominant: {state1.dominant_emotion}")
    
    # Example 2: Complex emotion (Bittersweet)
    print("\nExample 2: Bittersweet Moment")
    bittersweet = create_sample_llm_response(
        emotions={"joy": 0.6, "sadness": 0.4},
        contextual_factors=["farewell", "nostalgia"],
        drain_value=0.15
    )
    
    state2 = system.analyze_emotions(bittersweet)
    print(f"Emotion Label: {state2.emotion_label}")
    print(f"Coordinates: {state2.coordinates}")
    print(f"Stability: {state2.stability:.2f}")
    
    # Example 3: Using temporal smoothing
    print("\nExample 3: Temporal Smoothing")
    new_emotion = create_sample_llm_response(
        emotions={"anger": 0.7, "disgust": 0.3},
        drain_value=0.2
    )
    
    # Use previous state's token for smoothing
    state3 = system.analyze_emotions(new_emotion, last_analysis_token=state2.token)
    print(f"Emotion Label: {state3.emotion_label}")
    print(f"Stability: {state3.stability:.2f}")
    
    # Demonstrate token encoding/decoding
    print("\nToken Demonstration:")
    compact_token = system._encode_token(state1, token_everything=False)
    full_token = system._encode_token(state1, token_everything=True)
    
    print(f"Compact token length: {len(compact_token)} characters")
    print(f"Full token length: {len(full_token)} characters")
    
    # Decode and verify
    decoded_state = system.decode_token(full_token)
    print(f"Decoded emotion label: {decoded_state.emotion_label}")
    print(f"Original == Decoded: {state1.emotion_label == decoded_state.emotion_label}")


if __name__ == "__main__":
    # Run example if this file is executed directly
    run_example()