import requests


class LlamaCppAdapter:
    """Adapter for communicating with a running llama.cpp server."""

    def __init__(self, port=8080, timeout=120):
        self.base_url = f"http://127.0.0.1:{port}"
        self.timeout = timeout
        self.session = requests.Session()

    def is_ready(self):
        """Check whether the llama.cpp server is responding."""
        try:
            response = self.session.get(
                f"{self.base_url}/v1/models",
                timeout=3,
            )
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def generate(self, messages, temperature=0.2, max_tokens=256):
        """Send chat messages to llama.cpp and return the answer."""
        if not self.is_ready():
            raise RuntimeError(
                f"llama.cpp server is not responding at {self.base_url}. "
                "Start the server first."
            )

        response = self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json={
    		"messages": messages,
    		"temperature": temperature,
   		"max_tokens": max_tokens,
            	"chat_template_kwargs": {"enable_thinking": False},
	    },
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices", [])

        if not choices:
            raise RuntimeError(
                f"The server returned no choices: {data}"
            )

        choice = choices[0]
        message = choice.get("message", {})
        answer = (message.get("content") or "").strip()

        if not answer:
            raise RuntimeError(
                "The server returned an empty final answer. "
                f"Finish reason: {choice.get('finish_reason')}. "
                "The model may be returning reasoning instead of "
                "a final answer."
            )

        if choice.get("finish_reason") == "length":
            print(
                "Warning: the response reached the token limit "
                "and may be incomplete."
            )

        return answer
