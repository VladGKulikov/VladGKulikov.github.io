import numpy as np

IGNORE_INDEX = -100


def build_sft_example(messages, special_ids):
    """Build input_ids and labels; supervise assistant content and EOT."""
    role_id = {
        "system": special_ids["role_system"],
        "user": special_ids["role_user"],
        "assistant": special_ids["role_assistant"],
        "tool": special_ids["role_tool"],
    }
    input_ids = [special_ids["bos"]]
    labels = [IGNORE_INDEX]

    for message in messages:
        header = [
            special_ids["start_header"],
            role_id[message["role"]],
            special_ids["end_header"],
            special_ids["newline"],
        ]
        body = [*message["token_ids"], special_ids["eot"]]

        input_ids.extend(header + body)
        labels.extend([IGNORE_INDEX] * len(header))
        labels.extend(
            body
            if message["role"] == "assistant"
            else [IGNORE_INDEX] * len(body)
        )

    return np.asarray(input_ids), np.asarray(labels)


def first_fit_decreasing(lengths, max_length=2048):
    """Keep original example indices in every First-Fit Decreasing bin."""
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    if any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) for n in lengths):
        raise ValueError("lengths must be integers")
    bins = []
    for index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        length = lengths[index]
        if length <= 0 or length > max_length:
            raise ValueError("invalid sequence length")

        for bucket in bins:
            if bucket["used"] + length <= max_length:
                bucket["indices"].append(index)
                bucket["used"] += length
                break
        else:
            bins.append({"indices": [index], "used": length})

    return bins


def lora_params(d=4096, d_kv=1024, layers=32, rank=16):
    """Count q/k/v/o LoRA parameters for a GQA transformer."""
    per_layer = rank * (
        (d + d) + (d + d_kv) + (d + d_kv) + (d + d)
    )
    return per_layer * layers
