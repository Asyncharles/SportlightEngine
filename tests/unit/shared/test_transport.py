from __future__ import annotations

import asyncio

from websockets.asyncio.server import serve

from intensity.schema import ClockTick, Control, Event, make_envelope
from intensity.transport import IngestClient, SendQueue


def _tick(i: int):
    return make_envelope(ClockTick(feed_id="a", t_video=float(i), loop_epoch=0, t_emit_ns=i))


def _event(source: str, i: int, envelope_fixtures):
    data = dict(envelope_fixtures[source]["data"], t_emit_ns=i)
    return make_envelope(Event.model_validate(data))


def test_drops_oldest_clock_ticks_first(envelope_fixtures):
    q = SendQueue(maxsize=3)
    q.put(_tick(1))
    q.put(_event("crowd", 2, envelope_fixtures))
    q.put(_tick(3))
    q.put(_event("match", 4, envelope_fixtures))
    kinds = [q.get_nowait().data.t_emit_ns for _ in range(3)]
    assert kinds == [2, 3, 4]  # tick 1 went first
    assert q.dropped["clock"] == 1


def test_never_drops_match_or_control(envelope_fixtures):
    q = SendQueue(maxsize=2)
    for i in range(5):
        q.put(_event("match", i, envelope_fixtures))
    q.put(make_envelope(Control(feed_id="a", kind="loop", loop_epoch=1, t_emit_ns=9)))
    assert len(q) == 6  # grows past the bound rather than lose one
    q.put(_event("video", 10, envelope_fixtures))
    q.put(_tick(11))
    assert len(q) == 6
    assert q.dropped == {"clock": 1, "event": 1}


def test_client_sends_in_order_and_reconnects(envelope_fixtures):
    async def scenario() -> list[str]:
        received: list[str] = []
        connections = 0

        async def handler(ws):
            nonlocal connections
            connections += 1
            async for msg in ws:
                received.append(msg)
                if len(received) == 2 and connections == 1:
                    await ws.close()  # force a reconnect mid-stream
                    return

        async with serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            client = IngestClient(f"ws://127.0.0.1:{port}", backoff_s=(0.05, 0.2))
            task = asyncio.create_task(client.run())
            for i in range(5):
                client.put(_tick(i))
                await asyncio.sleep(0.05)
            for _ in range(100):
                if len(received) >= 5:
                    break
                await asyncio.sleep(0.02)
            client.stop()
            await asyncio.wait_for(task, 2)
            assert connections >= 2
        return received

    received = asyncio.run(scenario())
    stamps = [int(m.split('"t_emit_ns":')[1].split("}")[0]) for m in received]
    assert stamps[:2] == [0, 1] and set(range(5)) <= set(stamps)
