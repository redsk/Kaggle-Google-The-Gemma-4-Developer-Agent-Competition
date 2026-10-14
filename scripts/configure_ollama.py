"""Create a local model alias with a bounded context, then test tool calling."""

import argparse

import requests

URL = "http://127.0.0.1:11434"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=("31b", "12b", "e4b", "e2b"), default="12b")
    args = parser.parse_args()
    alias = f"gemma4-{args.size}-smoke"
    response = requests.post(
        f"{URL}/api/create",
        json={"model": alias, "from": f"gemma4:{args.size}",
              "parameters": {"num_ctx": 32768}, "stream": False},
        timeout=(10, 300),
    )
    response.raise_for_status()
    if "error" in response.json():
        raise RuntimeError(response.json()["error"])
    response = requests.post(
        f"{URL}/v1/chat/completions",
        json={
            "model": alias,
            "messages": [{"role": "user", "content": "Call get_status now to inspect the task budget."}],
            "tools": [{"type": "function", "function": {
                "name": "get_status", "description": "Return the task budget remaining.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }}],
            "temperature": 0,
            "max_tokens": 256,
            "stream": False,
        },
        timeout=(10, 300),
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    names = [call["function"]["name"] for call in message.get("tool_calls", [])]
    if "get_status" not in names:
        raise RuntimeError("The local model did not produce the expected structured tool call.")
    print(f"Local model ready: {alias}, context 32768, structured tool call verified.")


if __name__ == "__main__":
    main()
