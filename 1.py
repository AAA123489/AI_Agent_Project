'''
    logger.info(f"收到请求: user_id={data.user_id}, message={data.message[:50]}...")
    api_url = config.api_url

    try:
        result = await call_llm_client(data.message, api_key, api_url)
        if result.get("error"):
            logger.error(f"LLM 调用失败: user_id={data.user_id}, error={result.get('message')}")
            return {"reply":"抱歉，服务器出错了"}
        
        
        reply_text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                reply_text += block.get("text", "")

        if not reply_text:
            logger.warning(f"LLM 未返回文本内容: user_id={data.user_id}")
            return {"reply": "抱歉，AI 未生成回复"}

        logger.info(f"请求成功: user_id={data.user_id}")
        return {"reply": reply_text}
        

    except Exception:
        logger.error(f"未预料的服务器错误: user_id={data.user_id}", exc_info=True)
        return {"reply": "服务器内部错误"}
'''