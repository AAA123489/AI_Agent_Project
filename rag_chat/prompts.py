def build_rag_prompt(query: str, context_docs: list[str], history: list[dict]) -> str:
    context_text = "\n".join(context_docs) if context_docs else ""

    history_text = ""
    if history:
        history_lines = []
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

    if context_text.strip():
        # 有检索到文档 → 优先参考文档，不足时用自身知识补充
        prompt = (
            "你是一个智能助手。请根据下面的【参考上下文】和你的知识来回答用户的【当前问题】。\n"
            "优先级：\n"
            "1. 如果参考上下文中有相关信息，请基于上下文回答，确保准确性。\n"
            "2. 如果上下文中没有相关信息或信息不足，可以结合你自己的知识来回答，"
            "但在这种情况下，请在回答末尾注明\"（注：该回答基于通用知识，未在知识库中找到直接依据）\"。\n"
            "\n"
            "【历史对话】：\n"
            "{history}\n"
            "\n"
            "【参考上下文】：\n"
            "{context}\n"
            "\n"
            "【当前问题】：\n"
            "{query}\n"
            "\n"
            "【你的回答】："
        ).format(history=history_text, context=context_text, query=query)
    else:
        # 没有检索到文档 → 纯对话模式
        prompt = (
            "你是一个智能助手。请自然地回答用户的【当前问题】。\n"
            "\n"
            "【历史对话】：\n"
            "{history}\n"
            "\n"
            "【当前问题】：\n"
            "{query}\n"
            "\n"
            "【你的回答】："
        ).format(history=history_text, query=query)

    return prompt
