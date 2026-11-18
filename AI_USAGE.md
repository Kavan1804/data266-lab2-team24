# AI Usage, Team 24

The brief asks us to declare the AI coding assistants we used and what we used them for.

## Kavan Siddesh Kumar
**Tool:** Claude (Claude Code / Cursor agent, Anthropic).

**How I used it:**
I used Claude as a brainstorming and engineering assistant, not as an unsupervised author of the project. Main uses:

- **Architecture and design.** I chose the Part 1 RAG layout (ingestion → hybrid retrieval → rule-based table facts vs vision for figures → checked JSON) and the Part 3 agent pipeline (planner → vision → prompt → generation → critique) after talking through options with Claude: what to index, when to prefer exact table facts over the LLM, how to wire RRF retrieval, and how to keep Part 3 state and traces typed. Final design choices are mine.
- **Code refinement.** Claude helped refine and restructure code in `src/`, `tools/`, and `tests/` (cleaner interfaces, notebook builders, offline fakes). I reviewed diffs, kept what fit the brief, and changed or rejected the rest.
- **Debugging.** Claude helped diagnose runtime and parsing failures (for example PyMuPDF table extraction quirks and caption/year-column mix-ups). I reproduced each bug, confirmed the fix with offline tests or printed intermediates, and landed the patch myself.
- **Documentation.** Claude drafted or tightened wording for `README.md`, work-plan notes, notebook markdown, and early report phrasing. I rewrote report sections and contribution claims in my own words before submission.

**What it did not do:**
- It did not do any human evaluation. Groundedness ratings, Part 2 preference ratings, and Part 3 pass/fail judgments are from people only.
- It did not invent personal contribution claims or run the GPU notebooks for me. I ran Colab/Kaggle jobs, checked outputs, and committed results.
- It did not decide which Part 1 system the team ships; that comes from comparing both golden-set scores at merge.

**What I did:**
- Decided the architectures and evaluation setup for my parts (Part 1 system, Part 2 baseline, Part 3 lead).
- Wrote and/or heavily edited the code that ships in my notebooks and `src/` files after AI suggestions.
- Built golden sets, safety probes, scenarios, and red-team cases; filled my human eval forms by hand against the PDFs and traces.
- Ran offline tests (`tests/test_offline.py`, dry runs) and real GPU runs; verified metrics against notebook outputs before putting numbers in the report.
- Rewrote documentation and report text so the submitted voice and claims match what actually ran.

**Things it got wrong, and how they were found:**
- An early table parser read values like `3.0` as `30.0` because PyMuPDF’s table extraction scrambled characters. Printing parsed rows showed it; the parser now reads cells from word positions.
- With pinned PyMuPDF 1.25.5, a caption row was treated as a year column, so Tables 1-2 to 1-4 produced no facts and Table 1-1 rows were labelled “World”; kerned captions also hid Table A-3. The offline test failed on China’s 2009 growth, and printing facts showed why. Both issues are fixed.

## Muhammed Faiq Salman
**Tool:** [TO BE WRITTEN BY MUHAMMED: tool name and version]

**What it did:** [TO BE WRITTEN BY MUHAMMED]

**What it did not do:** [TO BE WRITTEN BY MUHAMMED]
