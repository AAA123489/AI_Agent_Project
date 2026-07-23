import re


class RecursiveTextSplitter:
    def __init__(self,chunk_size:int,chunk_overlap:int):

        if chunk_overlap>=chunk_size:
            raise ValueError("chunk_overlap块必须小于chunk_size块")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n","\n","。","."," "]


    def split_text(self,text: str) -> list[str]:
        final_chunks = self._recursive_split(text, separator_index=0)

        return self._apply_overlap(final_chunks)


    def _recursive_split(self, text: str, separator_index: int) -> list[str]:
        if len(text)<=self.chunk_size:
            return [text]

        if separator_index >= len(self.separators):
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separators = self.separators[separator_index]
        splits = re.split(f'({re.escape(separators)})',text)

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

    def _apply_overlap(self,chunks:list[str]) -> list[str]:

        if not chunks:
            return []
        overlapped_chunks = []
        current_chunk = chunks[0]

        for i in range(1,len(chunks)):
            overlapped_chunks.append(current_chunk)
            overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
            current_chunk = overlap_text + chunks[i]

            if len(current_chunk) >= self.chunk_size:
                current_chunk = current_chunk[-self.chunk_size:]
        overlapped_chunks.append(current_chunk)
        return overlapped_chunks