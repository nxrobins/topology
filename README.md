# EVOLVE

Discovers optimal AI agent team structures through evolutionary competition. 10 teams of LLM agents with random organizational genomes compete on a task. The weakest dissolve, the strongest mutate and propagate. Over generations, evolution finds what no one would design.

## Setup

```bash
pip install -r requirements.txt
export NEBIUS_API_KEY=your_key_here
python server.py
# Open index.html in a browser
```

## Future Work
- **Genome as deployable artifact:** Export winning structures as CrewAI/LangGraph configs
- **Cross-task transfer learning:** Train a classifier on evolutionary results to predict optimal structure from task description alone
- **Co-evolutionary fitness:** Evolve the evaluation rubric alongside the teams to escape evaluator bias
