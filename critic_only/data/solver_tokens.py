"""Locate solver-generated content inside the critic's quoted conversation, excluding prompts and references."""


def conversation_spans(messages: list[dict[str, str]]) -> tuple[str, list[tuple[int, int]]]:
    """Return quoted messages and character spans for assistant content; messages retain their saved order."""
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
    if not spans:
        raise ValueError("critic history contains no solver tokens")
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
    if not any(mask):
        raise ValueError("solver content has no complete tokens to supervise")
    return mask
