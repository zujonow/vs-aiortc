import asyncio
from unittest import TestCase

import aioice.stun
from aioice import ConnectionClosed
from vsaiortc.exceptions import InvalidStateError
from vsaiortc.rtcconfiguration import RTCIceServer
from vsaiortc.rtcicetransport import (
    RTCIceCandidate,
    RTCIceGatherer,
    RTCIceParameters,
    RTCIceTransport,
    connection_kwargs,
    parse_stun_turn_uri,
)

from .utils import asynctest


async def mock_connect():
    pass


async def mock_get_event():
    await asyncio.sleep(0.5)
    return ConnectionClosed()


class ConnectionKwargsTest(TestCase):
    def test_empty(self):
        self.assertEqual(connection_kwargs([]), {})

    def test_stun(self):
        self.assertEqual(
            connection_kwargs([RTCIceServer("stun:stun.l.google.com:19302")]),
            {"stun_server": ("stun.l.google.com", 19302)},
        )

    def test_stun_with_suffix(self):
        self.assertEqual(
            connection_kwargs(
                [RTCIceServer("stun:global.stun.twilio.com:3478?transport=udp")]
            ),
            {"stun_server": ("global.stun.twilio.com", 3478)},
        )

    def test_stun_multiple_servers(self):
        self.assertEqual(
            connection_kwargs(
                [
                    RTCIceServer("stun:stun.l.google.com:19302"),
                    RTCIceServer("stun:stun.example.com"),
                ]
            ),
            {"stun_server": ("stun.l.google.com", 19302)},
        )

    def test_stun_multiple_urls(self):
        self.assertEqual(
            connection_kwargs(
                [
                    RTCIceServer(
                        [
                            "stun:stun1.l.google.com:19302",
                            "stun:stun2.l.google.com:19302",
                        ]
                    )
                ]
            ),
            {"stun_server": ("stun1.l.google.com", 19302)},
        )

    def test_turn(self):
        self.assertEqual(
            connection_kwargs([RTCIceServer("turn:turn.example.com")]),
            {
                "turn_password": None,
                "turn_server": ("turn.example.com", 3478),
                "turn_ssl": False,
                "turn_transport": "udp",
                "turn_username": None,
            },
        )

    def test_turn_multiple_servers(self):
        self.assertEqual(
            connection_kwargs(
                [
                    RTCIceServer("turn:turn.example.com"),
                    RTCIceServer("turn:turn.example.net"),
                ]
            ),
            {
                "turn_password": None,
                "turn_server": ("turn.example.com", 3478),
                "turn_ssl": False,
                "turn_transport": "udp",
                "turn_username": None,
            },
        )

    def test_turn_multiple_urls(self):
        self.assertEqual(
            connection_kwargs(
                [RTCIceServer(["turn:turn1.example.com", "turn:turn2.example.com"])]
            ),
            {
                "turn_password": None,
                "turn_server": ("turn1.example.com", 3478),
                "turn_ssl": False,
                "turn_transport": "udp",
                "turn_username": None,
            },
        )

    def test_turn_over_bogus(self):
        self.assertEqual(
            connection_kwargs([RTCIceServer("turn:turn.example.com?transport=bogus")]),
            {},
        )

    def test_turn_over_tcp(self):
        self.assertEqual(
            connection_kwargs([RTCIceServer("turn:turn.example.com?transport=tcp")]),
            {
                "turn_password": None,
                "turn_server": ("turn.example.com", 3478),
                "turn_ssl": False,
                "turn_transport": "tcp",
                "turn_username": None,
            },
        )

    def test_turn_with_password(self):
        self.assertEqual(
            connection_kwargs(
                [
                    RTCIceServer(
                        urls="turn:turn.example.com", username="foo", credential="bar"
                    )
                ]
            ),
            {
                "turn_password": "bar",
                "turn_server": ("turn.example.com", 3478),
                "turn_ssl": False,
                "turn_transport": "udp",
                "turn_username": "foo",
            },
        )

    def test_turn_with_token(self):
        self.assertEqual(
            connection_kwargs(
                [
                    RTCIceServer(
                        urls="turn:turn.example.com",
                        username="foo",
                        credential="bar",
                        credentialType="token",
                    )
                ]
            ),
            {},
        )

    def test_turns(self):
        self.assertEqual(
            connection_kwargs([RTCIceServer("turns:turn.example.com")]),
            {
                "turn_password": None,
                "turn_server": ("turn.example.com", 5349),
                "turn_ssl": True,
                "turn_transport": "tcp",
                "turn_username": None,
            },
        )

    def test_turns_over_udp(self):
        self.assertEqual(
            connection_kwargs([RTCIceServer("turns:turn.example.com?transport=udp")]),
            {},
        )


class ParseStunTurnUriTest(TestCase):
    def test_invalid_scheme(self):
        with self.assertRaises(ValueError) as cm:
            parse_stun_turn_uri("foo")
        self.assertEqual(str(cm.exception), "malformed uri: invalid scheme")

    def test_invalid_uri(self):
        with self.assertRaises(ValueError) as cm:
            parse_stun_turn_uri("stun")
        self.assertEqual(str(cm.exception), "malformed uri")

    def test_stun(self):
        uri = parse_stun_turn_uri("stun:stun.services.mozilla.com")
        self.assertEqual(
            uri, {"host": "stun.services.mozilla.com", "port": 3478, "scheme": "stun"}
        )

    def test_stuns(self):
        uri = parse_stun_turn_uri("stuns:stun.services.mozilla.com")
        self.assertEqual(
            uri, {"host": "stun.services.mozilla.com", "port": 5349, "scheme": "stuns"}
        )

    def test_stun_with_port(self):
        uri = parse_stun_turn_uri("stun:stun.l.google.com:19302")
        self.assertEqual(
            uri, {"host": "stun.l.google.com", "port": 19302, "scheme": "stun"}
        )

    def test_turn(self):
        uri = parse_stun_turn_uri("turn:1.2.3.4")
        self.assertEqual(
            uri, {"host": "1.2.3.4", "port": 3478, "scheme": "turn", "transport": "udp"}
        )

    def test_turn_with_port_and_transport(self):
        uri = parse_stun_turn_uri("turn:1.2.3.4:3478?transport=tcp")
        self.assertEqual(
            uri, {"host": "1.2.3.4", "port": 3478, "scheme": "turn", "transport": "tcp"}
        )

    def test_turns(self):
        uri = parse_stun_turn_uri("turns:1.2.3.4")
        self.assertEqual(
            uri,
            {"host": "1.2.3.4", "port": 5349, "scheme": "turns", "transport": "tcp"},
        )

    def test_turns_with_port_and_transport(self):
        uri = parse_stun_turn_uri("turns:1.2.3.4:1234?transport=tcp")
        self.assertEqual(
            uri,
            {"host": "1.2.3.4", "port": 1234, "scheme": "turns", "transport": "tcp"},
        )


class RTCIceGathererTest(TestCase):
    @asynctest
    async def test_gather(self):
        gatherer = RTCIceGatherer()
        self.assertEqual(gatherer.state, "new")
        self.assertEqual(gatherer.getLocalCandidates(), [])
        await gatherer.gather()
        self.assertEqual(gatherer.state, "completed")
        self.assertTrue(len(gatherer.getLocalCandidates()) > 0)

        # close
        await gatherer._connection.close()

    def test_default_ice_servers(self):
        self.assertEqual(
            RTCIceGatherer.getDefaultIceServers(),
            [RTCIceServer(urls="stun:stun.l.google.com:19302")],
        )


class RTCIceTransportTest(TestCase):
    def setUp(self):
        # save timers
        self.consent_failures = aioice.ice.CONSENT_FAILURES
        self.consent_interval = aioice.ice.CONSENT_INTERVAL
        self.retry_max = aioice.stun.RETRY_MAX
        self.retry_rto = aioice.stun.RETRY_RTO

        # shorten timers to run tests faster
        aioice.ice.CONSENT_FAILURES = 1
        aioice.ice.CONSENT_INTERVAL = 1
        aioice.stun.RETRY_MAX = 1
        aioice.stun.RETRY_RTO = 0.1

    def tearDown(self):
        # restore timers
        aioice.ice.CONSENT_FAILURES = self.consent_failures
        aioice.ice.CONSENT_INTERVAL = self.consent_interval
        aioice.stun.RETRY_MAX = self.retry_max
        aioice.stun.RETRY_RTO = self.retry_rto

    @asynctest
    async def test_construct(self):
        gatherer = RTCIceGatherer()
        connection = RTCIceTransport(gatherer)
        self.assertEqual(connection.state, "new")
        self.assertEqual(connection.getRemoteCandidates(), [])

        candidate = RTCIceCandidate(
            component=1,
            foundation="0",
            ip="192.168.99.7",
            port=33543,
            priority=2122252543,
            protocol="UDP",
            type="host",
        )

        # add candidate
        await connection.addRemoteCandidate(candidate)
        self.assertEqual(connection.getRemoteCandidates(), [candidate])

        # end-of-candidates
        await connection.addRemoteCandidate(None)
        self.assertEqual(connection.getRemoteCandidates(), [candidate])

    @asynctest
    async def test_connect(self):
        gatherer_1 = RTCIceGatherer()
        transport_1 = RTCIceTransport(gatherer_1)

        gatherer_2 = RTCIceGatherer()
        transport_2 = RTCIceTransport(gatherer_2)

        # gather candidates
        await asyncio.gather(gatherer_1.gather(), gatherer_2.gather())
        for candidate in gatherer_2.getLocalCandidates():
            await transport_1.addRemoteCandidate(candidate)
        for candidate in gatherer_1.getLocalCandidates():
            await transport_2.addRemoteCandidate(candidate)
        self.assertEqual(transport_1.state, "new")
        self.assertEqual(transport_2.state, "new")

        # connect
        await asyncio.gather(
            transport_1.start(gatherer_2.getLocalParameters()),
            transport_2.start(gatherer_1.getLocalParameters()),
        )
        self.assertEqual(transport_1.state, "completed")
        self.assertEqual(transport_2.state, "completed")

        # cleanup
        await asyncio.gather(transport_1.stop(), transport_2.stop())
        self.assertEqual(transport_1.state, "closed")
        self.assertEqual(transport_2.state, "closed")

    @asynctest
    async def test_connect_fail(self):
        gatherer_1 = RTCIceGatherer()
        transport_1 = RTCIceTransport(gatherer_1)

        gatherer_2 = RTCIceGatherer()
        transport_2 = RTCIceTransport(gatherer_2)

        # gather candidates
        await asyncio.gather(gatherer_1.gather(), gatherer_2.gather())
        for candidate in gatherer_2.getLocalCandidates():
            await transport_1.addRemoteCandidate(candidate)
        for candidate in gatherer_1.getLocalCandidates():
            await transport_2.addRemoteCandidate(candidate)
        self.assertEqual(transport_1.state, "new")
        self.assertEqual(transport_2.state, "new")

        # connect
        await transport_2.stop()
        await transport_1.start(gatherer_2.getLocalParameters())
        self.assertEqual(transport_1.state, "failed")
        self.assertEqual(transport_2.state, "closed")

        # cleanup
        await asyncio.gather(transport_1.stop(), transport_2.stop())
        self.assertEqual(transport_1.state, "closed")
        self.assertEqual(transport_2.state, "closed")

    @asynctest
    async def test_connect_then_consent_expires(self):
        gatherer_1 = RTCIceGatherer()
        transport_1 = RTCIceTransport(gatherer_1)

        gatherer_2 = RTCIceGatherer()
        transport_2 = RTCIceTransport(gatherer_2)

        # gather candidates
        await asyncio.gather(gatherer_1.gather(), gatherer_2.gather())
        for candidate in gatherer_2.getLocalCandidates():
            await transport_1.addRemoteCandidate(candidate)
        for candidate in gatherer_1.getLocalCandidates():
            await transport_2.addRemoteCandidate(candidate)
        self.assertEqual(transport_1.state, "new")
        self.assertEqual(transport_2.state, "new")

        # connect
        await asyncio.gather(
            transport_1.start(gatherer_2.getLocalParameters()),
            transport_2.start(gatherer_1.getLocalParameters()),
        )
        self.assertEqual(transport_1.state, "completed")
        self.assertEqual(transport_2.state, "completed")

        # close one side
        await transport_1.stop()
        self.assertEqual(transport_1.state, "closed")

        # wait for consent to expire
        await asyncio.sleep(2)

        # close other side
        await transport_2.stop()
        self.assertEqual(transport_2.state, "closed")

    @asynctest
    async def test_connect_when_closed(self):
        gatherer = RTCIceGatherer()
        transport = RTCIceTransport(gatherer)

        # stop transport
        await transport.stop()
        self.assertEqual(transport.state, "closed")

        # try to start it
        with self.assertRaises(InvalidStateError) as cm:
            await transport.start(
                RTCIceParameters(usernameFragment="foo", password="bar")
            )
        self.assertEqual(str(cm.exception), "RTCIceTransport is closed")

    @asynctest
    async def test_connection_closed(self):
        gatherer = RTCIceGatherer()

        # mock out methods
        gatherer._connection.connect = mock_connect
        gatherer._connection.get_event = mock_get_event

        transport = RTCIceTransport(gatherer)
        self.assertEqual(transport.state, "new")

        await transport.start(RTCIceParameters(usernameFragment="foo", password="bar"))
        self.assertEqual(transport.state, "completed")

        await asyncio.sleep(1)
        self.assertEqual(transport.state, "failed")

        await transport.stop()
        self.assertEqual(transport.state, "closed")
