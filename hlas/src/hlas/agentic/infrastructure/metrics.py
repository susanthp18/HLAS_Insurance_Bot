"""
Prometheus metrics for the agentic chatbot.
"""

from prometheus_client import Counter, Histogram

# Agentic message processing
AGENTIC_MESSAGES_TOTAL = Counter(
    'agentic_messages_total', 
    'Total agentic messages processed', 
    ['result', 'product']
)

# Agentic response latency
AGENTIC_LATENCY = Histogram(
    'agentic_latency_seconds',
    'Agentic response latency in seconds',
    ['endpoint'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

# Session cache metrics
SESSION_CACHE_HITS = Counter('agentic_session_cache_hits_total', 'Agentic session cache hits')
SESSION_CACHE_MISSES = Counter('agentic_session_cache_misses_total', 'Agentic session cache misses')

# Redis lock metrics
REDIS_LOCK_TIMEOUTS = Counter(
    'agentic_redis_lock_timeouts_total', 
    'Agentic Redis lock acquisition timeouts', 
    ['scope']
)

# Live agent metrics
LIVE_AGENT_HANDOFFS = Counter(
    'agentic_live_agent_handoffs_total',
    'Total live agent handoff requests'
)

# Policy validation metrics
POLICY_VIOLATIONS = Counter(
    'agentic_policy_violations_total',
    'Total policy violations detected',
    ['type']
)

# WhatsApp specific metrics
WA_MESSAGES_PROCESSED_TOTAL = Counter(
    'agentic_wa_messages_processed_total',
    'Total WhatsApp messages processed by agentic handler',
    ['result']
)
