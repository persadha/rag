1. Identify the current limitations of the CRAG architecture and address them in a new, separate architecture (preserving the original CRAG implementation intact for future benchmarking). Additionally, extract the core strength of the Auto architecture—specifically, its sub-question re-retrieval mechanism—and integrate it into this new design.
2. Review the codebase to diagnose why the RAGAS evaluation could not be completed successfully. While the exact root cause is unrecorded, this blocker prompted the transition to DeepEval; please investigate the repository to identify the original issues with the RAGAS setup..
3. I plan to re-run the evaluations to compare the Standard architecture against the new 4th architecture using DeepEval. Due to a tight time constraint, I will conduct two separate evaluation runs: one using a proprietary model and one using an open-source model from our research.
- Proprietary: Please recommend the best option between OpenAI and Anthropic models for this use case.
- Open-Source: Please recommend the ideal model to select among Llama, Gemma (Gamma), or DeepEval’s supported open-source LLMs.
4. The codebase is currently hosted on a local machine equipped only with a CPU. Recognizing that GPU acceleration is required for this evaluation, what are the best methods or cloud alternatives to access GPU compute from my laptop? Additionally, please suggest strategies to improve retrieval quality (e.g., upgrading the embedding model, implementing alternative retrieval strategies, or appending metadata to each embedded vector).
Suggest method to improve retrieval quality. Better embedding? Different retrieval strategy? Embed metadata to every embedded vectors?
5. Conduct a comprehensive review of the user interface (UI) and identify any existing usability issues, bugs, or areas for improvement.
6. Is there an existing mechanism to visualize or log the retrieved context? I need to compare the retrieved text against the initial prompt to assess semantic proximity. Finally, please provide recommendations on architectural improvements required to scale the system and make it production-ready.


