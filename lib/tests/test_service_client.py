import unittest
from unittest.mock import Mock, patch

from sedna.service.client import http_request


class HTTPRequestTest(unittest.TestCase):

    @patch("sedna.service.client.requests.request")
    def test_forwards_explicit_timeout(self, request):
        response = Mock(status_code=200)
        request.return_value = response

        self.assertIs(http_request(
            "http://example.test", timeout=7, no_decode=True), response)

        request.assert_called_once_with(
            method="GET", url="http://example.test", timeout=7)

    @patch("sedna.service.client.requests.request")
    def test_forwards_default_timeout(self, request):
        response = Mock(status_code=200)
        request.return_value = response

        self.assertIs(http_request(
            "http://example.test", no_decode=True), response)

        request.assert_called_once_with(
            method="GET", url="http://example.test", timeout=300)


if __name__ == "__main__":
    unittest.main()
