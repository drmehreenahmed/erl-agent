# memory.py
import re
from sentence_transformers import SentenceTransformer
from api_client import query_api

class HeuristicMemory:
    """
    Stores heuristics (task + learned guideline) and provides
    LLM‑based retrieval of the most relevant ones for a new task.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.heuristics = []      # list of {"task": str, "heuristic": str}
        self.embeddings = []

    def add(self, task: str, heuristic: str):
        """Store a new heuristic if not already present."""
        if heuristic in [h["heuristic"] for h in self.heuristics]:
            return
        emb = self.embedder.encode([task])[0]
        self.embeddings.append(emb)
        self.heuristics.append({
            "task": task,
            "heuristic": heuristic
        })

    def retrieve_llm(self, current_task: str, k: int = 2) -> list:
        """
        Use the LLM to rank and select the top‑k most relevant heuristics.
        Returns a list of heuristic dicts.
        """
        if not self.heuristics:
            return []

        # Format available heuristics
        heuristics_list_str = ""
        for idx, h in enumerate(self.heuristics):
            heuristics_list_str += f"\nHeuristic ID {idx}:\nTask: {h['task']}\nLearned guideline:\n{h['heuristic']}\n"

        prompt = f"""
ROLE
You are an expert assistant specializing in selecting the most relevant heuristics to improve SEM scale bar analysis.

GOAL
Select the TOP {k} most relevant heuristics for the given task.

AVAILABLE HEURISTICS:
{heuristics_list_str}

TASK:
{current_task}

CRITERIA:
- Similarity of the task (unit inconsistency, OCR error, visual mismatch)
- Error relevance (cm/m → µm/nm, low OCR confidence)
- Verification strength (plausibility checks)

OUTPUT FORMAT (strict):
For each selected heuristic, output:
<ID>: ["justification", score]

Example:
2: ["Same unit correction issue", 95]
0: ["Visual verification strategy", 82]

Return ONLY the IDs with justification and scores.
"""
        response = query_api(prompt)
        # Parse numeric IDs from response
        ids = []
        for line in response.strip().split('\n'):
            match = re.match(r'^(\d+):', line.strip())
            if match:
                ids.append(int(match.group(1)))
        # Return dicts for valid IDs
        selected = [self.heuristics[i] for i in ids if i < len(self.heuristics)]
        return selected[:k]