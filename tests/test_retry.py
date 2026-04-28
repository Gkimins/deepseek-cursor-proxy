from __future__ import annotations

import json
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from urllib.error import HTTPError, URLError
from urllib.request import Request

from deepseek_cursor_proxy.config import ProxyConfig
from deepseek_cursor_proxy.reasoning_store import ReasoningStore
from deepseek_cursor_proxy.server import DeepSeekProxyHandler


def _handler_with_config(config: ProxyConfig | None = None) -> DeepSeekProxyHandler:
    handler = object.__new__(DeepSeekProxyHandler)
    handler.server = SimpleNamespace(
        config=config or ProxyConfig(),
        reasoning_store=ReasoningStore(":memory:"),
    )
    handler.wfile = BytesIO()
    handler.close_connection = False
    handler.send_response = lambda status: None
    handler.send_header = lambda name, value: None
    handler.end_headers = lambda: None
    return handler


def _fake_response(status: int = 200, body: bytes = b"{}") -> object:
    resp = BytesIO(body)
    resp.status = status  # type: ignore[attr-defined]
    resp.headers = {"Content-Type": "application/json"}  # type: ignore[attr-defined]
    return resp


class RetryLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = _handler_with_config()

    def tearDown(self) -> None:
        self.handler.server.reasoning_store.close()

    def test_successful_request_no_retry(self) -> None:
        request = Request("http://example.com/api")

        with patch("deepseek_cursor_proxy.server.urlopen") as mock_urlopen, \
                patch("time.sleep") as mock_sleep:
            mock_urlopen.return_value = _fake_response(200, b'{"ok":true}')
            response = self.handler._request_with_retry(request, label="test")

            self.assertEqual(response.status, 200)
            mock_urlopen.assert_called_once()
            mock_sleep.assert_not_called()

    def test_retries_on_429_then_succeeds(self) -> None:
        request = Request("http://example.com/api")
        http_error_429 = HTTPError(
            "http://example.com/api", 429, "Too Many Requests",
            {"Content-Type": "application/json"},
            BytesIO(b'{"error":"rate_limited"}'),
        )

        with patch("deepseek_cursor_proxy.server.urlopen") as mock_urlopen, \
                patch("time.sleep") as mock_sleep, \
                patch("deepseek_cursor_proxy.server._clone_request") as mock_clone:
            mock_clone.side_effect = lambda req: Request(req.full_url)
            mock_urlopen.side_effect = [
                http_error_429,
                _fake_response(200, b'{"ok":true}'),
            ]
            response = self.handler._request_with_retry(request, label="test")

            self.assertEqual(response.status, 200)
            self.assertEqual(mock_urlopen.call_count, 2)
            self.assertEqual(mock_sleep.call_count, 1)
            mock_sleep.assert_called_with(1)

    def test_retries_on_500_then_succeeds(self) -> None:
        request = Request("http://example.com/api")
        http_error_500 = HTTPError(
            "http://example.com/api", 500, "Internal Server Error",
            {"Content-Type": "application/json"},
            BytesIO(b'{"error":"server_error"}'),
        )

        with patch("deepseek_cursor_proxy.server.urlopen") as mock_urlopen, \
                patch("time.sleep") as mock_sleep, \
                patch("deepseek_cursor_proxy.server._clone_request") as mock_clone:
            mock_clone.side_effect = lambda req: Request(req.full_url)
            mock_urlopen.side_effect = [
                http_error_500,
                http_error_500,
                _fake_response(200, b'{"ok":true}'),
            ]
            response = self.handler._request_with_retry(request, label="test")

            self.assertEqual(response.status, 200)
            self.assertEqual(mock_urlopen.call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)
            mock_sleep.assert_any_call(1)
            mock_sleep.assert_any_call(2)

    def test_no_retry_on_400_client_error(self) -> None:
        request = Request("http://example.com/api")
        http_error_400 = HTTPError(
            "http://example.com/api", 400, "Bad Request",
            {"Content-Type": "application/json"},
            BytesIO(b'{"error":"bad_request"}'),
        )

        with patch("deepseek_cursor_proxy.server.urlopen") as mock_urlopen, \
                patch("time.sleep") as mock_sleep:
            mock_urlopen.side_effect = http_error_400

            with self.assertRaises(HTTPError) as ctx:
                self.handler._request_with_retry(request, label="test")

            self.assertEqual(ctx.exception.code, 400)
            mock_urlopen.assert_called_once()
            mock_sleep.assert_not_called()

    def test_no_retry_on_404(self) -> None:
        request = Request("http://example.com/api")
        http_error_404 = HTTPError(
            "http://example.com/api", 404, "Not Found",
            {"Content-Type": "application/json"},
            BytesIO(b'{"error":"not_found"}'),
        )

        with patch("deepseek_cursor_proxy.server.urlopen") as mock_urlopen, \
                patch("time.sleep") as mock_sleep:
            mock_urlopen.side_effect = http_error_404

            with self.assertRaises(HTTPError):
                self.handler._request_with_retry(request, label="test")

            mock_urlopen.assert_called_once()
            mock_sleep.assert_not_called()

    def test_retries_on_urlerror_then_succeeds(self) -> None:
        request = Request("http://example.com/api")

        with patch("deepseek_cursor_proxy.server.urlopen") as mock_urlopen, \
                patch("time.sleep") as mock_sleep, \
                patch("deepseek_cursor_proxy.server._clone_request") as mock_clone:
            mock_clone.side_effect = lambda req: Request(req.full_url)
            mock_urlopen.side_effect = [
                URLError("connection refused"),
                _fake_response(200, b'{"ok":true}'),
            ]
            response = self.handler._request_with_retry(request, label="test")

            self.assertEqual(response.status, 200)
            self.assertEqual(mock_urlopen.call_count, 2)
            mock_sleep.assert_called_once_with(1)

    def test_max_retries_exhausted_raises_last_exception(self) -> None:
        request = Request("http://example.com/api")
        http_error_500 = HTTPError(
            "http://example.com/api", 500, "Server Error",
            {"Content-Type": "application/json"},
            BytesIO(b'{"error":"persistent"}'),
        )

        with patch("deepseek_cursor_proxy.server.urlopen") as mock_urlopen, \
                patch("time.sleep") as mock_sleep, \
                patch("deepseek_cursor_proxy.server._clone_request") as mock_clone:
            mock_clone.side_effect = lambda req: Request(req.full_url)
            mock_urlopen.side_effect = http_error_500

            with self.assertRaises(HTTPError) as ctx:
                self.handler._request_with_retry(request, label="test")

            self.assertEqual(ctx.exception.code, 500)
            self.assertEqual(mock_urlopen.call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)

    def test_all_retries_exhausted_returns_last_error(self) -> None:
        request = Request("http://example.com/api")
        last_error = HTTPError(
            "http://example.com/api", 503, "Service Unavailable",
            {"Content-Type": "application/json"},
            BytesIO(b'{"error":"unavailable"}'),
        )

        with patch("deepseek_cursor_proxy.server.urlopen") as mock_urlopen, \
                patch("time.sleep") as mock_sleep, \
                patch("deepseek_cursor_proxy.server._clone_request") as mock_clone:
            mock_clone.side_effect = lambda req: Request(req.full_url)
            mock_urlopen.side_effect = [
                HTTPError("url", 429, "Rate Limited", {}, BytesIO(b"")),
                HTTPError("url", 502, "Bad Gateway", {}, BytesIO(b"")),
                last_error,
            ]

            with self.assertRaises(HTTPError) as ctx:
                self.handler._request_with_retry(request, label="test")
            self.assertEqual(ctx.exception.code, 503)

    def test_urlerror_all_attempts_fail(self) -> None:
        request = Request("http://example.com/api")

        with patch("deepseek_cursor_proxy.server.urlopen") as mock_urlopen, \
                patch("time.sleep") as mock_sleep, \
                patch("deepseek_cursor_proxy.server._clone_request") as mock_clone:
            mock_clone.side_effect = lambda req: Request(req.full_url)
            mock_urlopen.side_effect = URLError("persistent failure")

            with self.assertRaises(URLError) as ctx:
                self.handler._request_with_retry(request, label="test")

            self.assertIn("persistent failure", str(ctx.exception.reason))
            self.assertEqual(mock_urlopen.call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)

    def test_backoff_delays_increase(self) -> None:
        request = Request("http://example.com/api")
        http_error_429 = HTTPError(
            "http://example.com/api", 429, "Rate Limited",
            {"Content-Type": "application/json"},
            BytesIO(b"{}"),
        )

        with patch("deepseek_cursor_proxy.server.urlopen") as mock_urlopen, \
                patch("time.sleep") as mock_sleep, \
                patch("deepseek_cursor_proxy.server._clone_request") as mock_clone:
            mock_clone.side_effect = lambda req: Request(req.full_url)
            mock_urlopen.side_effect = [
                http_error_429,
                http_error_429,
                _fake_response(200),
            ]
            self.handler._request_with_retry(request, label="test")

            delays = [call.args[0] for call in mock_sleep.call_args_list]
            self.assertEqual(delays, [1, 2])

    def test_mixed_errors_retries_correctly(self) -> None:
        """First URLError, then HTTP 429, then success."""
        request = Request("http://example.com/api")

        with patch("deepseek_cursor_proxy.server.urlopen") as mock_urlopen, \
                patch("time.sleep") as mock_sleep, \
                patch("deepseek_cursor_proxy.server._clone_request") as mock_clone:
            mock_clone.side_effect = lambda req: Request(req.full_url)
            mock_urlopen.side_effect = [
                URLError("timeout"),
                HTTPError("url", 429, "Rate Limited", {}, BytesIO(b"")),
                _fake_response(200, b'{"ok":true}'),
            ]
            response = self.handler._request_with_retry(request, label="test")

            self.assertEqual(response.status, 200)
            self.assertEqual(mock_urlopen.call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
