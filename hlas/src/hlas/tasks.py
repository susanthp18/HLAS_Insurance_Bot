from crewai import Task
import yaml
from pathlib import Path
from .agents import (
    product_identifier,
    orchestrator,
    slot_validator,
    slot_extractor,
    question_asker,
    recommendation_responder,
    follow_up_agent,
    tier_identifier,
    followup_clarification_agent,
)

# Get the path to the YAML file
yaml_path = Path(__file__).parent / "config/tasks.yaml"

# Read the actual YAML content from the config/tasks.yaml file
with open(yaml_path, "r", encoding="utf-8") as f:
    yaml_content = f.read()

# Now parse the actual YAML content
tasks_config = yaml.safe_load(yaml_content)

agent_map = {
    "product_identifier": product_identifier,
    "orchestrator": orchestrator,
    "slot_validator": slot_validator,
    "slot_extractor": slot_extractor,
    "question_asker": question_asker,
    "recommendation_responder": recommendation_responder,
    "follow_up_agent": follow_up_agent,
    "tier_identifier": tier_identifier,
    "followup_clarification_agent": followup_clarification_agent,
}

def build_task(config_key: str) -> Task:
    config = dict(tasks_config[config_key])
    agent_name = config.get("agent")
    if isinstance(agent_name, str):
        agent_obj = agent_map.get(agent_name)
        if agent_obj is None:
            raise ValueError(f"Unknown agent '{agent_name}' referenced in task '{config_key}'.")
        config["agent"] = agent_obj
    # Provide a safe default expected_output if missing to avoid validation errors
    config.setdefault("expected_output", "")
    return Task(**config)

# Instantiate the tasks with proper Agent instances
identify_product_task = build_task("identify_product")
synthesize_response_task = build_task("synthesize_response")
route_decision_task = build_task("route_decision")
construct_follow_up_query_task = build_task("construct_follow_up_query")
identify_tiers_task = build_task("identify_tiers")
validate_slot_task = build_task("validate_slot")
extract_slots_task = build_task("extract_slots")
ask_question_task = build_task("ask_question")


# Expose a name -> Task mapping for routing convenience
task_by_name = {
    "identify_product": identify_product_task,
    "synthesize_response": synthesize_response_task,
    "route_decision": route_decision_task,
    "identify_tiers": identify_tiers_task,
    "validate_slot": validate_slot_task,
    "extract_slots": extract_slots_task,
    "ask_question": ask_question_task,
    "followup_clarification": build_task("followup_clarification"),
}
