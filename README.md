### Notes
- `DailyAgent.py`, `DailyModelManager.py`, `bot_shared_with_coval_for_debugging.py` are just for reference aka original code
- `run_simulation.py` and `simple_daily_agent.py` are the frankensteined versions i put together to run locally

### Setup
1. setup virtual environment
```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2. set up .env file
```
OPENAI_API_KEY='your-open-ai-key'
ELEVENLABS_API_KEY='your-eleven-labs-key'
DEEPGRAM_API_KEY='your-deepgram-key'
PIPECAT_API_KEY='your-pipecat-cloud-api-key' 
AGENT_NAME='your-agent-name'
(if you can't replicate issues, we can provide the api key of a customer's agent that results in the issues we get w/ their permission of course - ophir@ezraailabs.tech)
```
3. run simulation
```
python run_simulation.py --duration 1
```
4. transcript and audio viewable in local directory