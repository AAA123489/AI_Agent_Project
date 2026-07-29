import logging
import re

logger = logging.getLogger(__name__)


class RecursiveTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):

        if chunk_overlap>=chunk_size:
            raise ValueError("chunk_overlap块必须小于chunk_size块")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n","\n","。","."," "]


    def split_text(self,text: str) -> list[str]:
        if not text.strip():
            return []
        final_chunks = self._recursive_split(text, separator_index=0)

        return self._apply_overlap(final_chunks)


    def _recursive_split(self, text: str, separator_index: int) -> list[str]:
        if len(text)<=self.chunk_size:
            return [text]

        if separator_index >= len(self.separators):
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator = self.separators[separator_index]
        splits = re.split(f'({re.escape(separator)})', text)

        merged_splits = []
        for i in range(0, len(splits) - 1, 2):
            merged_splits.append(splits[i]+splits[i+1])
        if len(splits) %2 !=0:
            merged_splits.append(splits[-1])

        result_chunk = []
        for split in merged_splits:
            if len(split) <=self.chunk_size:
                result_chunk.append(split)
            else :
                result_chunk.extend(
                    self._recursive_split(split, separator_index + 1)
                )
        return result_chunk

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """合并相邻小块至接近 chunk_size，并在块间添加重叠窗口。"""
        if not chunks:
            return []
        overlapped_chunks = []
        current_chunk = chunks[0]

        for i in range(1, len(chunks)):
            # 尝试累积：若合并下一块不超限，则直接拼接（解决上游切分产出过小块的问题）
            if len(current_chunk) + len(chunks[i]) <= self.chunk_size:
                current_chunk += chunks[i]
            else:
                # 当前块已足够大，输出并利用重叠开启新块
                overlapped_chunks.append(current_chunk)
                overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk = overlap_text + chunks[i]

                # 若新块（重叠 + 下一块文本）超出限制，截断保留尾部
                if len(current_chunk) >= self.chunk_size:
                    trimmed = len(current_chunk) - self.chunk_size
                    logger.debug(
                        "chunk 合并后超出 chunk_size（当前 %d，上限 %d），截断丢弃 %d 字符",
                        len(current_chunk), self.chunk_size, trimmed
                    )
                    current_chunk = current_chunk[-self.chunk_size:]
        overlapped_chunks.append(current_chunk)
        return overlapped_chunks