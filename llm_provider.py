from dotenv import load_dotenv
import os
import json
import requests
from typing import Dict, Any,Tuple

load_dotenv()


def create_llm_prompt(text: str) -> str:
    """Create structured prompt for LLM emotion analysis."""
    prompt = f"""Analyze the following text and identify the intensity of the eight core emotions: Joy, Anger, Guilt, Pride, Love, Fear, Sadness, and Disgust. Also, identify any non-emotional contextual factors that could negatively impact psychological stability.

    **EMOTION ANALYSIS GUIDELINES:**
    - Rate each emotion's intensity from 0.0 (not present) to 1.0 (maximum intensity)
    - Base intensity on linguistic cues and emotional indicators:
    * Mild expressions ("a bit sad", "slightly annoyed"): 0.1-0.3
    * Moderate expressions ("I'm happy", "feeling worried"): 0.4-0.6
    * Strong expressions ("I'm devastated", "absolutely furious"): 0.7-0.9
    * Extreme expressions ("overwhelmed with joy", "completely terrified"): 0.9-1.0
    - Consider context, metaphors, and implicit emotional content
    - If no clear emotion is present, use 0.0

    **CONTEXTUAL DRAIN ANALYSIS:**
    Identify non-emotional stressors that could impact psychological stability:
    - Physical factors: tiredness, illness, hunger, pain
    - Environmental factors: noise, weather, crowded spaces
    - Social factors: work pressure, relationship conflicts, social obligations
    - Situational factors: deadlines, financial stress, major life changes
    - Rate the overall impact from 0.0 (no impact) to 1.0 (severe impact)

    **RESPONSE FORMAT:**
    Return a valid JSON object with exactly this structure:

    {{
    "emotions": {{
        "joy": <float 0.0-1.0>,
        "anger": <float 0.0-1.0>,
        "guilt": <float 0.0-1.0>,
        "pride": <float 0.0-1.0>,
        "love": <float 0.0-1.0>,
        "fear": <float 0.0-1.0>,
        "sadness": <float 0.0-1.0>,
        "disgust": <float 0.0-1.0>
    }},
    "contextual_drain": {{
        "factors": ["factor1", "factor2", ...],
        "drain_value": <float 0.0-1.0>
    }}
    }}

    **EXAMPLES:**

    Text: "I'm a bit tired but excited about the weekend!"
    Response: {{
    "emotions": {{
        "joy": 0.6,
        "anger": 0.0,
        "guilt": 0.0,
        "pride": 0.0,
        "love": 0.0,
        "fear": 0.0,
        "sadness": 0.0,
        "disgust": 0.0
    }},
    "contextual_drain": {{
        "factors": ["tiredness"],
        "drain_value": 0.3
    }}
    }}

    Text: "I can't believe I messed up that presentation. Everyone was watching..."
    Response: {{
    "emotions": {{
        "joy": 0.0,
        "anger": 0.0,
        "guilt": 0.7,
        "pride": 0.0,
        "love": 0.0,
        "fear": 0.4,
        "sadness": 0.5,
        "disgust": 0.0
    }},
    "contextual_drain": {{
        "factors": ["work pressure", "social anxiety"],
        "drain_value": 0.6
    }}
    }}

    **TEXT TO ANALYZE:**
    "{text}"

    Provide your analysis as a valid JSON object following the exact format above."""
    return prompt


def get_available_llm() -> Tuple[str, str, str]:
    """
    Check which LLM API keys are available and return the first one found.
    
    Returns:
        Tuple of (llm_name, api_key, base_url)
    
    Raises:
        ValueError: If no API keys are found
    """
    llm_configs = [
        ("anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"),
        ("openai", "OPENAI_API_KEY", "OPENAI_BASE_URL"),
        ("gemini", "GEMINI_API_KEY", "GEMINI_BASE_URL")
    ]
    
    for llm_name, key_env, url_env in llm_configs:
        api_key = os.getenv(key_env, "").strip()
        base_url = os.getenv(url_env, "").strip()
        
        if api_key and base_url:
            return llm_name, api_key, base_url
    
    raise ValueError("No valid API keys found. Please set at least one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY")


def call_anthropic(prompt: str, api_key: str, base_url: str) -> Dict[str, Any]:
    """Call Anthropic Claude API"""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(base_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    result = response.json()
    return {"content": result["content"][0]["text"]}


def call_openai(prompt: str, api_key: str, base_url: str) -> Dict[str, Any]:
    """Call OpenAI API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4000
    }
    
    response = requests.post(base_url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    result = response.json()
    return {"content": result["choices"][0]["message"]["content"]}


def call_gemini(prompt: str, api_key: str, base_url: str) -> Dict[str, Any]:
    """Call Google Gemini API"""
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add API key to URL for Gemini
    url_with_key = f"{base_url}?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "maxOutputTokens": 4000
        }
    }
    
    response = requests.post(url_with_key, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    
    result = response.json()
    return {"content": result["candidates"][0]["content"]["parts"][0]["text"]}



def get_llm_response(text: str) -> Dict[str, Any]:
    """
    Get LLM output using the first available API key.
    
    Args:
        text: Input text to be processed by the LLM
        
    Returns:
        Dict containing the LLM response and metadata
        
    Raises:
        ValueError: If no API keys are available or if response cannot be parsed as JSON
        requests.RequestException: If API call fails
    """
    try:
        # Get the first available LLM
        llm_name, api_key, base_url = get_available_llm()
        
        # Create the prompt
        prompt = create_llm_prompt(text)
        
        # Call the appropriate LLM
        llm_functions = {
            "anthropic": call_anthropic,
            "openai": call_openai,
            "gemini": call_gemini
        }
        
        llm_response = llm_functions[llm_name](prompt, api_key, base_url)
        content = llm_response["content"].strip()
        
        # Try to parse the response as JSON
        try:
            # First, try to parse the entire content as JSON
            parsed_response = json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from the content
            # Look for JSON-like content between curly braces
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    parsed_response = json.loads(json_match.group())
                except json.JSONDecodeError:
                    raise ValueError(f"LLM response contains invalid JSON format. Raw response: {content}")
            else:
                raise ValueError(f"LLM response does not contain valid JSON format. Raw response: {content}")
        
        return {
            "success": True,
            "llm_used": llm_name,
            "data": parsed_response,
            "raw_response": content
        }
        
    except ValueError as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": "ValueError"
        }
    except requests.RequestException as e:
        return {
            "success": False,
            "error": f"API request failed: {str(e)}",
            "error_type": "RequestException"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "error_type": "Exception"
        }


# Example usage:
if __name__ == "__main__":
    # Test the function
    test_text = "I love programming!"
    result = get_llm_response(test_text)
    
    if result["success"]:
        print(f"LLM used: {result['llm_used']}")
        print(f"Response: {result['data']}")
    else:
        print(f"Error: {result['error']}")
