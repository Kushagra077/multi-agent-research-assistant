"""A small client for the research API, used by the Streamlit review page.

The page only talks to the API, never to the graph, so it works against a server running
anywhere and needs no keys of its own.
"""

import httpx


class ApiError(Exception):
    """The API refused a request, or could not be reached. The message is for a person."""


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            return httpx.request(
                method, self.base_url + path, timeout=self.timeout, **kwargs
            )
        except httpx.HTTPError as e:
            raise ApiError(f"Cannot reach the API at {self.base_url}: {e}") from e

    @staticmethod
    def _detail(response: httpx.Response) -> str:
        try:
            detail = response.json()["detail"]
        except ValueError, KeyError, TypeError:
            return f"HTTP {response.status_code}"
        if isinstance(detail, list):  # a 422: one entry per bad field
            return "; ".join(str(item.get("msg", item)) for item in detail)
        return str(detail)

    def _checked(self, response: httpx.Response) -> dict:
        if response.is_error:
            raise ApiError(self._detail(response))
        return response.json()

    def health(self) -> dict | None:
        """The API's health report, or None if it cannot be reached."""
        try:
            response = self._request("GET", "/health")
        except ApiError:
            return None
        return None if response.is_error else response.json()

    def start(self, question: str) -> dict:
        return self._checked(
            self._request("POST", "/runs", json={"question": question})
        )

    def get(self, thread_id: str) -> dict | None:
        """The run, or None if there is no run with that id."""
        response = self._request("GET", f"/runs/{thread_id}")
        if response.status_code == 404:
            return None
        return self._checked(response)

    def review(
        self,
        thread_id: str,
        action: str,
        feedback: str | None = None,
        text: str | None = None,
    ) -> dict:
        body = {"action": action, "feedback": feedback, "text": text}
        return self._checked(
            self._request("POST", f"/runs/{thread_id}/review", json=body)
        )

    def resume(self, thread_id: str) -> dict:
        return self._checked(self._request("POST", f"/runs/{thread_id}/resume"))
