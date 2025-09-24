from crewai import Task
import yaml
from pathlib import Path
from .agents import (
    product_identifier,
    orchestrator,
    travel_recommendation_manager,
    maid_recommendation_manager,
    car_recommendation_manager,
    questionnaire_agent,
    recommendation_agent,
    summary_agent,
    comparison_agent,
    rag_agent,
    explanation_agent,
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
    "questionnaire_agent": questionnaire_agent,
    "recommendation_agent": recommendation_agent,
    "summary_agent": summary_agent,
    "comparison_agent": comparison_agent,
    "rag_agent": rag_agent,
    "explanation_agent": explanation_agent,
    "orchestrator": orchestrator,
    "travel_recommendation_manager": travel_recommendation_manager,
    "maid_recommendation_manager": maid_recommendation_manager,
    "car_recommendation_manager": car_recommendation_manager,
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
questionnaire_ask_next_slot_task = build_task("questionnaire_ask_next_slot")
questionnaire_capture_pending_slot_task = build_task("questionnaire_capture_pending_slot")
provide_recommendation_task = build_task("provide_recommendation")
retrieve_information_task = build_task("retrieve_information")
compare_products_task = build_task("compare_products")
summarize_product_task = build_task("summarize_product")
synthesize_response_task = build_task("synthesize_response")
route_decision_task = build_task("route_decision")
construct_follow_up_query_task = build_task("construct_follow_up_query")
identify_tiers_task = build_task("identify_tiers")
manage_travel_recommendation_flow_task = build_task("manage_travel_recommendation_flow")
manage_maid_recommendation_flow_task = build_task("manage_maid_recommendation_flow")
manage_car_recommendation_flow_task = build_task("manage_car_recommendation_flow")
validate_slot_task = build_task("validate_slot")
extract_slots_task = build_task("extract_slots")
ask_question_task = build_task("ask_question")

# Create a list of all tasks
recommendation_tasks = [
    provide_recommendation_task,
    retrieve_information_task,
    compare_products_task,
    summarize_product_task,
    synthesize_response_task,
]

# Expose a name -> Task mapping for routing convenience
task_by_name = {
    "identify_product": identify_product_task,
    "questionnaire_ask_next_slot": questionnaire_ask_next_slot_task,
    "questionnaire_capture_pending_slot": questionnaire_capture_pending_slot_task,
    "provide_recommendation": provide_recommendation_task,
    "retrieve_information": retrieve_information_task,
    "compare_products": compare_products_task,
    "summarize_product": summarize_product_task,
    "synthesize_response": synthesize_response_task,
    "route_decision": route_decision_task,
    "manage_travel_recommendation_flow": manage_travel_recommendation_flow_task,
    "manage_maid_recommendation_flow": manage_maid_recommendation_flow_task,
    "manage_car_recommendation_flow": manage_car_recommendation_flow_task,
    "validate_slot": validate_slot_task,
    "extract_slots": extract_slots_task,
    "ask_question": ask_question_task,
    "followup_clarification": build_task("followup_clarification"),
}