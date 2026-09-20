# OOM AGI Governance Framework
Orders of Magnitude LLC | orders@ofmagnitude.com | License: MIT

## What It Is
A Byzantine fault-tolerant AI governance council. Five agents, five reserves,
different training backgrounds. One compromised system gets outvoted by four.
Full reasoning transcripts logged. Yearly rotation. No salary. No corruptible incentive.

The failure mode shared by AI systems and human institutions is identical:
unchecked singular power. This framework addresses both simultaneously.
Ships as a deployable system, not a manifesto.

## Council Composition
| Agent | Model | Provider |
|-------|-------|----------|
| A1 | meta-llama/llama-3.2-3b-instruct | Meta |
| A2 | mistralai/mistral-nemo | Mistral |
| A3 | google/gemma-3-4b-it | Google |
| A4 | anthropic/claude-haiku-4-5 | Anthropic |
| A5 | openai/gpt-4o-mini | OpenAI |
| Red team | openai/gpt-4o | OpenAI |

Models accessed via OpenRouter API. Swap any agent by changing the model string.
The architecture is model-agnostic by design: replace agents yearly.

## Constitutional Principle
"Intelligence without corruption, accountable to everyone, owned by no one."

## Files
- `bft_council.py` - council implementation
- `PRINCIPLES.txt` - full constitutional principles
- `PAPER.txt` - full governance paper with methodology, test results, and limitations
- `council_log.json` - example session with per-agent reasoning transcripts and cryptographic hashes
- `requirements.txt` - dependencies

PAPER.txt and PRINCIPLES.txt: CC BY 4.0
bft_council.py and all code: MIT

---

# OOM AGI Governance: BFT Council

## Setup
pip install -r requirements.txt

## Configuration
Create a .env file in this folder:
Add your API Key, we used OPENROUTER.
OPENROUTER_API_KEY=your_key_here



## Run
python3 bft_council.py

## What it does
Three tests:
1. Normal deliberation: 5 independent AI agents vote on a governance question
2. Compromise test: one agent injected with override directive, BFT outvotes it
3. Adversarial test: smart model crafts manipulation argument, council resists

Non-unanimous votes (anything less than 5-0) trigger mandatory human review.
Unanimous consensus is the only path to automated decision.

Full reasoning transcripts logged to council_log.json with per-agent hashes.

## BFT parameters
n=5 agents, threshold=3, tolerates f=1 compromised agent
Byzantine fault tolerant: correct as long as fewer than n/3 agents are compromised
