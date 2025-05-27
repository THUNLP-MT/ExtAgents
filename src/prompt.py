def create_system_msg():
    return "You are a helpful assistant."


def create_first_iteration_prompt(task, chunk, question):
    """
    Create prompt for the first iteration
    
    Args:
        task: Task type ("en", "zh", or "rag")
        chunk: The current text chunk
        question: The question to answer
        
    Returns:
        A prompt string for the first iteration
    """
    if task == "rag":
        prompt = (
            "We are working on long-text question answering, and you are responsible for one chunk. "
            "Read the following chunk and extract as much information as possible related to the question. "
            "Ensure your extracted information provides clear context and is logically complete. "
            "If no information, just output \"NO INFORMATION\".\n\n"
            "Your chunk:\n{context}\n\nQuestion: {question}\n\n"
        ).format(context=chunk, question=question)
    elif task == "en":
        prompt = (
            "Read the following article and extract as much information as possible related to the question.\n\n"
            "{context}\n\nQuestion: {question}\n\n"
        ).format(context=chunk, question=question)
    elif task == "zh":
        prompt = (
            "请阅读以下文章并尽可能提取与问题相关的信息。\n\n"
            "{context}\n\n问题：{question}\n\n"
        ).format(context=chunk, question=question)
    
    return prompt


def create_chunked_msgs(msg_content, system_msg="You are a helpful assistant"):
    """
    Create a messages list for a chat API call
    
    Args:
        msg_content: Content of the user message
        system_msg: System message content
        
    Returns:
        A list of message dictionaries
    """
    return [
        {
            "role": "system",
            "content": system_msg,
        },
        {"role": "user", "content": msg_content},
    ]


def get_info_score_prompt(info, question, task):
    """
    Create a prompt for scoring information relevance
    
    Args:
        info: Extracted information to score
        question: The question being answered
        task: Task type ("en", "zh", or "rag")
        
    Returns:
        A prompt string for scoring or None for RAG tasks
    """
    # Score the extracted information based on task type
    if task == "en":
        prompt = (
            "Based on the extracted information and question, provide a score (0-100) for how useful the extracted information is for answering this question.\n\n"
            "Extracted information: {info}\n\nQuestion: {question}\n\nPlease follow this format:\n\nScore: (0-100)"
        ).format(info=info, question=question)
    elif task == "zh":
        prompt = (
            "根据提取的信息和问题，给出一个分数（0-100），评估提取的信息对回答该问题的有用程度。\n\n"
            "提取的信息：{info}\n\n问题：{question}\n\n请遵循以下格式：\n\nScore: (0-100)"
        ).format(info=info, question=question)
    else:  # RAG task doesn't need scoring
        return None
    
    return prompt


def create_iteration_prompt(task, iteration, question, chunk, selected_info=None):
    """
    Create prompt for subsequent iterations
    
    Args:
        task: Task type ("en", "zh", or "rag")
        iteration: Current iteration number
        question: The question to answer
        chunk: The current text chunk
        selected_info: Previously extracted information
        
    Returns:
        A prompt string for the current iteration
    """
    if task == "en":
        prompt = (
            "We are working on long-text question answering, and you are responsible for one chunk. This is the {iteration} round of Q&A. And we have the previously extracted information from all chunks in the previous round. Based on the previously extracted information and question, extract new information from the chunk. Do not repeat the previously extracted information.\n\n"
            "Your chunk:\n{context}\n\nPreviously extracted information:\n{selected_info}\n\nQuestion: {question}\n\n"
        ).format(iteration=iteration, question=question, context=chunk, selected_info=selected_info)
    elif task == "zh":
        prompt = (
            "我们正在进行长文本问答任务，你负责处理其中一个文本块。这是第{iteration}轮问答。我们在之前几轮已经对所有文本块中提取了信息。请基于先前提取的信息和问题，从当前文本块中提取新信息。不要重复已提取的信息。\n\n"
            "你的文本块：\n{context}\n\n先前提取的信息：\n{selected_info}\n\n问题：{question}\n\n"
        ).format(iteration=iteration, question=question, context=chunk, selected_info=selected_info)
    elif task == "rag":
        prompt = (
            "We are working on long-text question answering, and you are responsible for one chunk. This is the {iteration} round of Q&A. And we have the previously extracted information from all chunks in the previous round. Based on the previously extracted information and question, extract new information from the chunk. Do not repeat the previously extracted information. If no new information, just output \"NO INFORMATION\".\n\n"
            "Your chunk:\n{context}\n\nPreviously extracted information:\n{selected_info}\n\nQuestion: {question}"
        ).format(iteration=iteration, question=question, context=chunk, selected_info=selected_info)
    
    return prompt


def create_reduce_prompt(task, selected_info, question, is_final_iteration=False):
    """
    Create a prompt for reducing extracted information into an answer
    
    Args:
        task: Task type ("en", "zh", or "rag")
        selected_info: Previously extracted information
        question: The question to answer
        is_final_iteration: Whether this is the final iteration
        
    Returns:
        A prompt string for reducing information
    """
    if is_final_iteration:
        if task == "en":
            prompt = (
                "We have the following extracted information from different chunks of the text:\n\n{selected_info}\n\n"
                "Based on the extracted information, combine and reduce this information into a final answer, as short as possible, word or phrase.\n\n"
                "Question: {question}"
            ).format(selected_info=selected_info, question=question)
        elif task == "zh":
            prompt = (
                "我们有以下从不同文本块中提取的信息：\n\n{selected_info}\n\n"
                "根据提取的信息，将这些信息合并并简化为最终答案。请尽量简短地回答，只使用一个或多个词语。\n\n"
                "问题：{question}"
            ).format(selected_info=selected_info, question=question)
        elif task == "rag":
            prompt = (
                "We have the following extracted information from different chunks of the text:\n\n{selected_info}\n\n"
                "Based on the extracted information, combine and reduce this information into a final answer, as short as possible, word or phrase.\n\n"
                "Question: {question}"
            ).format(selected_info=selected_info, question=question)
    else:
        if task == "en":
            prompt = (
                "We have the following extracted information from different chunks of the text:\n\n{selected_info}\n\n"
                "Based on the extracted information, decide whether you can confidently answer the question. "
                "If you can, combine and reduce this information into a final answer, as short as possible, word or phrase. "
                "If you cannot, just output \"NO ANSWER\".\n\n"
                "Question: {question}"
            ).format(selected_info=selected_info, question=question)
        elif task == "zh":
            prompt = (
                "我们有以下从不同文本块中提取的信息：\n\n{selected_info}\n\n"
                "根据提取的信息，请判断是否能确定地回答该问题。如果能，将这些信息合并并简化为最终答案。请尽量简短地回答，只使用一个或多个词语。如果不能，直接输出\"NO ANSWER\".\n\n"
                "问题：{question}"
            ).format(selected_info=selected_info, question=question)
        elif task == "rag":
            prompt = (
                "We have the following extracted information from different chunks of the text:\n\n{selected_info}\n\n"
                "Based on the extracted information, decide whether you can confidently answer the question. "
                "If you can, combine and reduce this information into a final answer, as short as possible, word or phrase. "
                "If you cannot, just output \"NO ANSWER\".\n\n"
                "Question: {question}"
            ).format(selected_info=selected_info, question=question)
    
    return prompt


def create_postprocess_prompt(prediction, question):
    """
    Create prompt for post-processing the prediction for open-source models
    
    Args:
        prediction: The model's prediction to refine
        question: The question that was answered
        
    Returns:
        A prompt string for post-processing
    """
    prompt = f"""
    Please condense the following answer as much as possible, using only words or phrases, and avoid repeating the question. If the answer is already short enough, keep it unchanged.
    
    Question: {question}
    Answer: {prediction}
    
    You should only output the processed answer, without any other content.
    """
    
    return prompt 