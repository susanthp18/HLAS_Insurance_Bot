"""
Prometheus metrics for HLAS chatbot.
"""
from prometheus_client import Counter, Histogram

# HTTP requests
REQUESTS_TOTAL = Counter(
    'hlas_requests_total', 'Total HTTP requests', ['endpoint', 'status']
)

# WhatsApp processing outcomes
WA_MESSAGES_PROCESSED_TOTAL = Counter(
    'hlas_wa_messages_processed_total', 'Total WhatsApp messages processed grouped by result', ['result']
)

# Session cache metrics
SESSION_CACHE_HITS = Counter('hlas_session_cache_hits_total', 'Session cache hits')
SESSION_CACHE_MISSES = Counter('hlas_session_cache_misses_total', 'Session cache misses')

# Redis locks
REDIS_LOCK_TIMEOUTS = Counter('hlas_redis_lock_timeouts_total', 'Redis lock acquisition timeouts', ['scope'])
