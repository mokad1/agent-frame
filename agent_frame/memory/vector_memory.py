"""长期向量记忆。

使用 FAISS + sentence-transformers 存储记忆向量，
通过语义检索返回与当前查询最相关的历史记忆。

适用场景：长对话中的知识回溯、跨会话记忆检索。
"""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from agent_frame.config import config
from agent_frame.memory.base import BaseMemory
from agent_frame.utils.logger import get_logger

logger = get_logger("memory.vector")


class VectorMemory(BaseMemory):
    """向量记忆。

    每条记忆存储为 (文本, 向量) 对，检索时通过语义相似度召回。
    """

    def __init__(
        self,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        dim: int = 384,
        device: str = "cpu",
    ) -> None:
        """初始化。

        Args:
            embedding_model: SentenceTransformer 模型名。
            dim: 向量维度。
            device: 推理设备。
        """
        self.dim = dim
        self._messages: list[dict[str, Any]] = []
        self._embeddings: list[np.ndarray] = []

        logger.info("Loading embedding model for vector memory: %s", embedding_model)
        self._model = SentenceTransformer(embedding_model, device=device)
        self._index: faiss.IndexFlatIP | None = None

    def add(self, role: str, content: str, **meta: Any) -> None:
        """添加记忆并生成向量。"""
        embedding = self._model.encode(
            [content], normalize_embeddings=True,
        )[0]

        self._messages.append({"role": role, "content": content, **meta})
        self._embeddings.append(embedding)
        self._index = None  # 索引失效，下次检索时重建

    def get_context(self, query: str = "", max_tokens: int = 2000) -> str:
        """语义检索最相关的历史记忆。

        Args:
            query: 当前查询文本（用于语义匹配）。
            max_tokens: 返回文本的 token 上限。

        Returns:
            拼接后的相关记忆文本。
        """
        if not self._messages:
            return ""

        # 重建 FAISS 索引
        if self._index is None and self._embeddings:
            emb_array = np.array(self._embeddings, dtype=np.float32)
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(emb_array)

        if self._index is None:
            return ""

        # 检索 top_k
        top_k = min(5, len(self._messages))
        if query:
            q_emb = self._model.encode(
                [query], normalize_embeddings=True,
            )[0].reshape(1, -1).astype(np.float32)
            scores, indices = self._index.search(q_emb, top_k)
        else:
            # 无查询时返回最近的记忆
            return self._recent_context(max_tokens)

        # 拼接结果
        lines: list[str] = []
        char_count = 0
        char_limit = max_tokens * 2

        for idx in indices[0]:
            if idx < 0 or idx >= len(self._messages):
                continue
            msg = self._messages[idx]
            line = f"{msg['role']}: {msg['content']}"
            if char_count + len(line) > char_limit:
                break
            lines.append(line)
            char_count += len(line)

        return "\n".join(lines)

    def _recent_context(self, max_tokens: int) -> str:
        """返回最近的记忆（无查询时的降级方案）。"""
        lines: list[str] = []
        char_count = 0
        for msg in reversed(self._messages[-5:]):
            line = f"{msg['role']}: {msg['content']}"
            if char_count + len(line) > max_tokens * 2:
                break
            lines.insert(0, line)
        return "\n".join(lines)

    def clear(self) -> None:
        self._messages.clear()
        self._embeddings.clear()
        self._index = None

    @property
    def count(self) -> int:
        return len(self._messages)
