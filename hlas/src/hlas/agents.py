from crewai import Agent
from .llm import azure_llm
import yaml
from pathlib import Path

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
def build_agent_from_config(agent_key: str) -> Agent:
    cfg = dict(agents_config[agent_key])
    # Ensure mandatory keys exist; CrewAI will fill placeholders from cfg
    cfg.setdefault("role", agent_key)
    cfg.setdefault("goal", "")
    cfg.setdefault("backstory", "")
    return Agent(
        llm=azure_llm,
        system_template=custom_system_template,
        prompt_template=custom_prompt_template,
        use_system_prompt=True,
        **cfg,
    )

# Instantiate the agents
questionnaire_agent = build_agent_from_config("questionnaire_agent")
product_identifier = build_agent_from_config("product_identifier") if "product_identifier" in agents_config else build_agent_from_config("questionnaire_agent")
orchestrator = build_agent_from_config("orchestrator") if "orchestrator" in agents_config else build_agent_from_config("questionnaire_agent")
slot_validator = build_agent_from_config("slot_validator") if "slot_validator" in agents_config else questionnaire_agent
slot_extractor = build_agent_from_config("slot_extractor") if "slot_extractor" in agents_config else questionnaire_agent
question_asker = build_agent_from_config("question_asker") if "question_asker" in agents_config else questionnaire_agent
recommendation_responder = build_agent_from_config("recommendation_responder") if "recommendation_responder" in agents_config else questionnaire_agent
follow_up_agent = build_agent_from_config("follow_up_agent") if "follow_up_agent" in agents_config else questionnaire_agent
tier_identifier = build_agent_from_config("tier_identifier") if "tier_identifier" in agents_config else questionnaire_agent
followup_clarification_agent = build_agent_from_config("followup_clarification_agent") if "followup_clarification_agent" in agents_config else questionnaire_agent

