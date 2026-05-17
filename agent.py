# agent.py
import re
from api_client import query_api

def generate_heuristic(task: str, trajectory: str, outcome: str, ocr_info: dict, det_info: dict) -> str:
    """
    Post‑mortem reflection: analyse the reasoning trajectory and produce a
    Trigger → Action heuristic rule.
    """
    prompt = f"""
You are an intelligent agent engaging in self-reflection for SEM scale bar analysis.

Context:
- Task: {task}
- Outcome: {outcome}
- Trajectory: {trajectory}
- OCR Output: {ocr_info.get('value', '?')} {ocr_info.get('unit', '?')} (conf: {ocr_info.get('confidence', 0)})
- Detection Confidence: {det_info.get('confidence', 0)}

Instructions:
IF FAILURE:
1. Pinpoint the breakpoint (unit misinterpretation, OCR misread, hallucination).
2. Diagnose the cause.
3. Define a correction rule (Trigger → Action).

IF SUCCESS:
1. Identify the winning move.
2. Derive a best practice.

Output format (strict):
1. Analysis: <brief explanation>
2. Learned Guideline:
   Trigger: When I encounter <condition>
   Action: I must <procedure>
"""
    return query_api(prompt)

def run_agent(image_np, ocr_info, det_info, user_query, heuristic_memory):
    """
    Execute the ERL agent:
    1. Base reasoning (no heuristics)
    2. Generate a new heuristic from the base outcome
    3. Retrieve relevant past heuristics
    4. Refined reasoning with those heuristics
    Returns a dictionary with base, refined, heuristic, retrieved_heuristics, outcome.
    """
    system_prompt = """
You are an expert SEM scale bar analysis agent.
STRICT RULES:
- Final answer MUST be the scale bar value (e.g., "5.0 µm").
- NEVER output FoV or magnification.
- Always validate unit plausibility (nm, µm only).
- If OCR unit is cm/m, correct to µm/nm.
"""

    # Step 1: Base reasoning
    base_prompt = f"""
## CONTEXT
OCR extracted scale: {ocr_info.get('value', '?')} {ocr_info.get('unit', '')} (confidence: {ocr_info.get('confidence', 0)})
Detection confidence: {det_info.get('confidence', 0)}

## TASK
{user_query}

## OUTPUT FORMAT
Thought: ...
Action: ...
Observation: ...
Final Answer: <scale bar value>
"""
    base_response = query_api(base_prompt, image_np=image_np, system_msg=system_prompt)

    # Determine outcome (success/failure) for post‑mortem
    final_match = re.search(r'(\d+(?:\.\d+)?)\s*(nm|µm|μm|um|cm|m)', base_response, re.I)
    if final_match:
        value, unit = float(final_match.group(1)), final_match.group(2).lower()
        if unit in ["nm", "µm", "μm", "um"] and value < 10000:
            outcome = f"SUCCESS: valid {unit} scale, value {value}"
        elif unit in ["cm", "m"]:
            outcome = f"FAILURE: invalid unit {unit} (should be µm/nm)"
        elif ocr_info.get("confidence", 0) < 0.5:
            outcome = "FAILURE: low OCR confidence, may have hallucinated"
        else:
            outcome = "AMBIGUOUS: needs verification"
    else:
        outcome = "FAILURE: no valid scale bar extracted"

    # Generate new heuristic from base experience
    heuristic_text = generate_heuristic(user_query, base_response, outcome, ocr_info, det_info)
    heuristic_memory.add(user_query, heuristic_text)

    # Retrieve relevant past heuristics
    retrieved = heuristic_memory.retrieve_llm(user_query, k=2)
    heuristics_str = "\n\n".join([f"### Heuristic {i+1}\n{h['heuristic']}" for i, h in enumerate(retrieved)])

    # Step 2: Refined reasoning with heuristics
    refined_prompt = f"""
## CONTEXT
OCR: {ocr_info.get('value')} {ocr_info.get('unit')} (conf={ocr_info.get('confidence')})
Detection confidence: {det_info.get('confidence')}

## RELEVANT HEURISTICS
{heuristics_str if heuristics_str else "None available"}

## TASK
{user_query}

## OUTPUT FORMAT (same)
Thought: ...
Action: ...
Observation: ...
Final Answer: <scale bar value>
"""
    refined_response = query_api(refined_prompt, image_np=image_np, system_msg=system_prompt)

    return {
        "base": base_response,
        "refined": refined_response,
        "heuristic": heuristic_text,
        "retrieved_heuristics": retrieved,
        "outcome": outcome
    }