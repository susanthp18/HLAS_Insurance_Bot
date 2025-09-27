from crewai import Agent
from .llm import azure_llm, azure_response_llm
import yaml
from pathlib import Path

# Models are initialized once in main.py at application startup

# Get the path to the YAML file
yaml_path = Path(__file__).parent / "config/agents.yaml"

with open(yaml_path, "r") as f:
    agents_config = yaml.safe_load(f)

custom_system_template = (
    "You are {role}. {backstory}\n"
    "Your goal is: {goal}\n\n"
    "Guidelines:\n"
    "- Be concise, professional, and helpful.\n"
    "- When a task expects structured output, return JSON following the described structure.\n"
    "- Avoid forceful phrasing; keep responses neutral and user-friendly.\n"
)

custom_prompt_template = (
    "Task: {input}\n\n"
    "Please complete this task thoughtfully and return the requested structure when applicable."
)

# Small helper to create an Agent with our custom templates
def build_agent_from_config(agent_key: str, use_response_llm: bool = False) -> Agent:
    cfg = dict(agents_config[agent_key])
    # Ensure mandatory keys exist; CrewAI will fill placeholders from cfg
    cfg.setdefault("role", agent_key)
    cfg.setdefault("goal", "")
    cfg.setdefault("backstory", "")
    # Use response LLM for specific agents that generate user-facing responses
    selected_llm = azure_response_llm if use_response_llm else azure_llm
    return Agent(
        llm=selected_llm,
        system_template=custom_system_template,
        prompt_template=custom_prompt_template,
        use_system_prompt=True,
        **cfg,
    )

# Instantiate the agents (require explicit configuration per agent)
product_identifier = build_agent_from_config("product_identifier")
orchestrator = build_agent_from_config("orchestrator")
slot_validator = build_agent_from_config("slot_validator")
slot_extractor = build_agent_from_config("slot_extractor")
question_asker = build_agent_from_config("question_asker")
# Use response LLM for agents that generate user-facing responses
recommendation_responder = build_agent_from_config("recommendation_responder", use_response_llm=True)
follow_up_agent = build_agent_from_config("follow_up_agent", use_response_llm=True)
tier_identifier = build_agent_from_config("tier_identifier")
followup_clarification_agent = build_agent_from_config("followup_clarification_agent", use_response_llm=True)

