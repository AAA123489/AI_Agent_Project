

def build_rag_prompt(query: str, context_docs: list[str], history: list[dict]) -> str:

    context_docs = "\n".join(context_docs)if context_docs else "无相关上下文"

    history_text = ""
    if history:
        history_lines = []
        for msg in history:
            role = msg.get("role","unknown")
            content = msg.get("content","")
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)


    prompt_template = """你是一个智能助手。请严格根据下面提供的【参考上下文】来回答用户的【当前问题】。
如果上下文中没有相关信息，请如实回答“根据现有资料无法回答”，不要编造内容。

【历史对话】：
{history}

【参考上下文】：
{context}

【当前问题】：
{query}

【你的回答】："""

    final_prompt = prompt_template.format(
        history = history_text,
        context = context_docs,
        query = query
    )
    return final_prompt