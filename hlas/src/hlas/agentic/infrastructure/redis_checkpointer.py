"""
Redis-based LangGraph Checkpointer for production conversation persistence.

This checkpointer stores LangGraph conversation state in Redis, ensuring
conversation context survives server restarts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

try:
    import orjson
    def _dumps(obj: Any) -> str:
        return orjson.dumps(obj, default=str).decode("utf-8")
    def _loads(s: str) -> Any:
        return orjson.loads(s)
except ImportError:
    import json
    def _dumps(obj: Any) -> str:
        return json.dumps(obj, default=str)
    def _loads(s: str) -> Any:
        return json.loads(s)

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from .redis_utils import get_redis

logger = logging.getLogger(__name__)

# Default TTL for checkpoints (24 hours)
CHECKPOINT_TTL_SECONDS = 86400


class RedisCheckpointer(BaseCheckpointSaver):
    """
    Redis-based checkpoint saver for LangGraph.
    
    Stores conversation checkpoints in Redis with configurable TTL.
    This ensures conversation state persists across server restarts
    while automatically expiring old conversations.
    """

    def __init__(
        self,
        prefix: str = "agentic:checkpoint",
        ttl_seconds: int = CHECKPOINT_TTL_SECONDS,
    ):
        super().__init__(serde=JsonPlusSerializer())
        self._prefix = prefix
        self._ttl = ttl_seconds
        self._client = None

    def _get_client(self):
        """Lazy Redis client initialization."""
        if self._client is None:
            self._client = get_redis()
        return self._client

    def _checkpoint_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        """Generate Redis key for a checkpoint."""
        return f"{self._prefix}:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _metadata_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        """Generate Redis key for checkpoint metadata."""
        return f"{self._prefix}:meta:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def _index_key(self, thread_id: str, checkpoint_ns: str) -> str:
        """Generate Redis key for checkpoint index (sorted set)."""
        return f"{self._prefix}:index:{thread_id}:{checkpoint_ns}"

    def _writes_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        """Generate Redis key for pending writes."""
        return f"{self._prefix}:writes:{thread_id}:{checkpoint_ns}:{checkpoint_id}"

    def get_tuple(self, config: Dict[str, Any]) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple by config."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        client = self._get_client()

        if checkpoint_id is None:
            # Get the latest checkpoint
            index_key = self._index_key(thread_id, checkpoint_ns)
            result = client.zrevrange(index_key, 0, 0)
            if not result:
                return None
            checkpoint_id = result[0]

        checkpoint_key = self._checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        metadata_key = self._metadata_key(thread_id, checkpoint_ns, checkpoint_id)

        checkpoint_data = client.get(checkpoint_key)
        metadata_data = client.get(metadata_key)

        if not checkpoint_data:
            return None

        try:
            checkpoint = self.serde.loads_typed(("json", checkpoint_data.encode() if isinstance(checkpoint_data, str) else checkpoint_data))
            metadata = _loads(metadata_data) if metadata_data else {}
        except Exception as e:
            logger.error("Failed to deserialize checkpoint: %s", e)
            return None

        # Get parent checkpoint id from metadata
        parent_checkpoint_id = metadata.get("parent_checkpoint_id")
        parent_config = None
        if parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }

        # Get pending writes
        writes_key = self._writes_key(thread_id, checkpoint_ns, checkpoint_id)
        pending_writes_data = client.lrange(writes_key, 0, -1)
        pending_writes = []
        for write_data in pending_writes_data:
            try:
                task_id, channel, value = _loads(write_data)
                pending_writes.append((task_id, channel, value))
            except Exception:
                pass

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def list(
        self,
        config: Optional[Dict[str, Any]],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread."""
        if config is None:
            return

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")

        client = self._get_client()
        index_key = self._index_key(thread_id, checkpoint_ns)

        # Get checkpoint IDs in reverse chronological order
        if before:
            before_id = before["configurable"]["checkpoint_id"]
            # Get rank of the before checkpoint
            rank = client.zrevrank(index_key, before_id)
            if rank is None:
                return
            start = rank + 1
        else:
            start = 0

        end = start + (limit - 1) if limit else -1
        checkpoint_ids = client.zrevrange(index_key, start, end)

        for checkpoint_id in checkpoint_ids:
            checkpoint_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }
            result = self.get_tuple(checkpoint_config)
            if result:
                yield result

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Store a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        client = self._get_client()

        # Serialize checkpoint
        checkpoint_data = self.serde.dumps_typed(checkpoint)[1]
        if isinstance(checkpoint_data, bytes):
            checkpoint_data = checkpoint_data.decode("utf-8")

        # Add parent checkpoint id to metadata
        meta_to_store = dict(metadata) if metadata else {}
        if parent_checkpoint_id:
            meta_to_store["parent_checkpoint_id"] = parent_checkpoint_id
        metadata_data = _dumps(meta_to_store)

        # Store checkpoint and metadata
        checkpoint_key = self._checkpoint_key(thread_id, checkpoint_ns, checkpoint_id)
        metadata_key = self._metadata_key(thread_id, checkpoint_ns, checkpoint_id)
        index_key = self._index_key(thread_id, checkpoint_ns)

        pipe = client.pipeline()
        pipe.set(checkpoint_key, checkpoint_data, ex=self._ttl)
        pipe.set(metadata_key, metadata_data, ex=self._ttl)
        # Use checkpoint_id as score (assumes monotonically increasing IDs)
        pipe.zadd(index_key, {checkpoint_id: float(checkpoint_id.replace("-", "")[:16])})
        pipe.expire(index_key, self._ttl)
        pipe.execute()

        logger.debug("Stored checkpoint %s for thread %s", checkpoint_id, thread_id)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: Dict[str, Any],
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store pending writes for a checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        client = self._get_client()
        writes_key = self._writes_key(thread_id, checkpoint_ns, checkpoint_id)

        pipe = client.pipeline()
        for channel, value in writes:
            write_data = _dumps([task_id, channel, value])
            pipe.rpush(writes_key, write_data)
        pipe.expire(writes_key, self._ttl)
        pipe.execute()

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints for a thread."""
        client = self._get_client()
        
        # Handle both dict config and string thread_id
        if isinstance(thread_id, dict):
            thread_id = thread_id.get("configurable", {}).get("thread_id", str(thread_id))
        
        pattern = f"{self._prefix}:*:{thread_id}:*"
        keys = list(client.scan_iter(pattern))
        
        # Also delete index keys
        index_pattern = f"{self._prefix}:index:{thread_id}:*"
        keys.extend(list(client.scan_iter(index_pattern)))
        
        if keys:
            client.delete(*keys)
            logger.info("Deleted %d checkpoint keys for thread %s", len(keys), thread_id)
