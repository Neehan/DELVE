"""Locate solver-generated content inside the critic's quoted conversation, excluding prompts and references."""


def conversation_spans(messages: list[dict[str, str]]) -> tuple[str, list[tuple[int, int]]]:
    """Return quoted messages and character spans for assistant content; messages retain their saved order."""
    if not messages or messages[-1]["role"] != "assistant" or not messages[-1]["content"].strip():
        raise ValueError("critic history must end with a nonempty solver response")
    parts, spans = [], []
    position = 0
    for message in messages:
        heading = ("\n\n" if parts else "") + f"{message['role']}:\n"
        start = position + len(heading)
        text = heading + message["content"]
        if message["role"] == "assistant" and message["content"]:
            spans.append((start, start + len(message["content"])))
        parts.append(text)
        position += len(text)
    return "".join(parts), spans


def solver_token_mask(rendered: str, conversation: str, spans: list[tuple[int, int]], offsets: list[tuple[int, int]]) -> list[bool]:
    """Return a token mask from template text, conversation-relative spans and fast-tokenizer character offsets.

    Only tokens entirely inside assistant content are supervised. Boundary tokens spanning instructions or delimiters are excluded. The quoted conversation must occur exactly once in the rendered template.
    """
    if rendered.count(conversation) != 1:
        raise ValueError("quoted conversation must occur exactly once in the critic input")
    origin = rendered.index(conversation)
    mask = [False] * len(offsets)
    for start, end in spans:
        for index, (left, right) in enumerate(offsets):
            if origin + start <= left < right <= origin + end:
                mask[index] = True
    final_start = origin + spans[-1][0]
    if not any(mask[index] for index, (left, _) in enumerate(offsets) if left >= final_start):
        raise ValueError("final solver response has no complete tokens to supervise")
    return mask
