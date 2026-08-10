# Copyright 2026 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import tenacity
from websockets.exceptions import ConnectionClosedError

from sedna.service.client import AggregationClient


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _new_client():
    # Build an AggregationClient without going through __init__, since
    # __init__ opens a real websocket connection.
    client = object.__new__(AggregationClient)
    client.uri = "ws://fake-aggregator"
    client.client_id = "test-client"
    client._ws_timeout = 5
    return client


def test_recv_reconnects_after_connection_closed():
    """A dropped websocket must be re-established instead of leaving
    recv() to fail forever on the stale connection."""
    client = _new_client()

    closed_ws = MagicMock()
    closed_ws.closed = True
    closed_ws.recv = AsyncMock(side_effect=ConnectionClosedError(1006, ""))
    client.ws = closed_ws

    reconnected_ws = MagicMock()
    reconnected_ws.closed = False
    reconnected_ws.recv = AsyncMock(return_value='{"type": "recv_weight"}')

    async def fake_connect():
        client.ws = reconnected_ws
        return "ok"

    client.connect = AsyncMock(side_effect=fake_connect)

    result = _run(client._recv())

    assert result == '{"type": "recv_weight"}'
    client.connect.assert_called_once()


def test_recv_does_not_reconnect_when_connection_is_open():
    client = _new_client()

    open_ws = MagicMock()
    open_ws.closed = False
    open_ws.recv = AsyncMock(return_value='{"type": "recv_weight"}')
    client.ws = open_ws
    client.connect = AsyncMock()

    result = _run(client._recv())

    assert result == '{"type": "recv_weight"}'
    client.connect.assert_not_called()


def test_recv_fails_fast_when_reconnect_is_exhausted():
    """connect() already retries internally; once it gives up, _recv()
    must surface the error immediately instead of retrying it again."""
    client = _new_client()

    closed_ws = MagicMock()
    closed_ws.closed = True
    client.ws = closed_ws
    client.connect = AsyncMock(side_effect=tenacity.RetryError(None))

    with pytest.raises(tenacity.RetryError):
        _run(client._recv())

    client.connect.assert_called_once()
