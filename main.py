import time
from datetime import datetime
from llm_provider import get_llm_response
from chs import CoordinateHeartSystem

def format_emotional_state(emotional_state):
    """Format EmotionalState object for readable terminal output"""
    
    print("\n" + "="*60)
    print("🧠 EMOTIONAL ANALYSIS RESULTS")
    print("="*60)
    
    # Basic info
    print(f"📅 Analysis Time: {emotional_state.timestamp}")
    print(f"🎯 Dominant Emotion: {emotional_state.dominant_emotion.upper()}")
    print(f"🏷️  Emotion Label: {emotional_state.emotion_label}")
    print(f"📍 Coordinates: ({emotional_state.coordinates[0]:.3f}, {emotional_state.coordinates[1]:.3f})")
    print(f"⚖️  Stability: {emotional_state.stability:.3f}")
    
    # Emotion intensities
    print("\n" + "-"*40)
    print("💫 EMOTION INTENSITIES")
    print("-"*40)
    
    emotions = emotional_state.intensities.__dict__
    for emotion, intensity in emotions.items():
        if intensity > 0:
            # Create a simple bar visualization
            bar_length = int(intensity * 20)  # Scale to 20 chars max
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{emotion.capitalize():>8}: {bar} {intensity:.3f}")
        else:
            print(f"{emotion.capitalize():>8}: {'░' * 20} {intensity:.3f}")
    
    # Contextual drain
    print("\n" + "-"*40)
    print("⚡ CONTEXTUAL DRAIN")
    print("-"*40)
    print(f"Drain Value: {emotional_state.contextual_drain.drain_value:.3f}")
    print("Contributing Factors:")
    for factor in emotional_state.contextual_drain.factors:
        print(f"  • {factor}")
    
    # Additional drain info
    print(f"\n🔥 Conflict Drain: {emotional_state.conflict_drain:.3f}")
    print(f"📊 Emotional Load Drain: {emotional_state.emotional_load_drain:.3f}")
    
    # Raw LLM data (condensed)
    print("\n" + "-"*40)
    print("🤖 RAW LLM EMOTIONS")
    print("-"*40)
    raw_emotions = emotional_state.raw_llm_data.get('emotions', {})
    for emotion, value in raw_emotions.items():
        if value > 0:
            print(f"{emotion.capitalize():>8}: {value:.3f}")
    
    print("\n" + "="*60)
    print()

def main(text: str):
    print(f"\n🔍 Analyzing text: '{text}'")
    print("⏳ Processing...")
    
    # Start total timing
    total_start_time = time.time()
    
    # Time LLM response
    print("🤖 Getting LLM response...")
    llm_start_time = time.time()
    llm_response = get_llm_response(text).get("data", None)
    llm_end_time = time.time()
    llm_time = llm_end_time - llm_start_time
    
    # Time CHS analysis
    print("🧠 Running emotion analysis...")
    chs_start_time = time.time()
    chs = CoordinateHeartSystem()
    result = chs.analyze_emotions(llm_response=llm_response)
    chs_end_time = time.time()
    chs_time = chs_end_time - chs_start_time
    
    # End total timing
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Format and display results
    format_emotional_state(result)
    
    # Display detailed timing info
    print("⏱️  PERFORMANCE BREAKDOWN")
    print("-" * 30)
    print(f"🤖 LLM Response Time: {llm_time:.3f} seconds")
    print(f"🧠 CHS Analysis Time: {chs_time:.5f} seconds")
    print(f"📊 Total Time: {total_time:.3f} seconds")
    print(f"📝 Token: {result.token}")
    
    return result

if __name__ == "__main__":
    print("🌟 Emotional Analysis System")
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = main(" finally got the lead role in the play, and I’m thrilled! But my best friend,who also tried out, didn’t get it, and I feel so guilty celebrating in front of her")
    
    print(f"\n✅ Analysis complete at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")