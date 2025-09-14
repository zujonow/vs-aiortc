import fractions
from typing import Optional, cast

from av import AudioFrame, AudioResampler, CodecContext
from av.frame import Frame
from av.packet import Packet

from ..jitterbuffer import JitterFrame
from ..mediastreams import convert_timebase
from .base import Decoder, Encoder

SAMPLE_RATE = 48000
SAMPLES_PER_FRAME = 960
TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)


class OpusDecoder(Decoder):
    def __init__(
        self, audio_ptime: Optional[float] = None, sample_rate: Optional[int] = None
    ) -> None:
        # keep old constructor logic
        self.audio_ptime = (
            self._build_ptime(audio_ptime * 1000) / 1000 if audio_ptime else 0.020
        )
        self.sample_rate = sample_rate if sample_rate else SAMPLE_RATE
        self.samples_per_frame = int(self.audio_ptime * self.sample_rate)
        self.time_base = fractions.Fraction(1, self.sample_rate)

        # use PyAV codec instead of ffi/libopus
        self.codec = CodecContext.create("libopus", "r")
        self.codec.format = "s16"
        self.codec.layout = "stereo"
        self.codec.sample_rate = self.sample_rate

    def _build_ptime(self, audio_ptime: float) -> float:
        # round ptime to multiple of 20ms (old logic preserved)
        if (audio_ptime / 10) % 2 == 0:
            return audio_ptime
        else:
            return audio_ptime + 10

    def decode(self, encoded_frame: JitterFrame) -> list[Frame]:
        packet = Packet(encoded_frame.data)
        packet.pts = encoded_frame.timestamp
        packet.time_base = self.time_base
        return cast(list[Frame], self.codec.decode(packet))


class OpusEncoder(Encoder):
    def __init__(self) -> None:
        self.codec = CodecContext.create("libopus", "w")
        self.codec.bit_rate = 96000
        self.codec.format = "s16"
        self.codec.layout = "stereo"
        self.codec.options = {"application": "voip"}
        self.codec.sample_rate = SAMPLE_RATE
        self.codec.time_base = TIME_BASE

        # Create our own resampler to control the frame size.
        self.resampler = AudioResampler(
            format="s16",
            layout="stereo",
            rate=SAMPLE_RATE,
            frame_size=SAMPLES_PER_FRAME,
        )

        self.first_packet_pts: Optional[int] = None

    def encode(
        self, frame: Frame, force_keyframe: bool = False
    ) -> tuple[list[bytes], int]:
        assert isinstance(frame, AudioFrame)
        assert frame.format.name == "s16"
        assert frame.layout.name in ["mono", "stereo"]

        packets = []
        for frame in self.resampler.resample(frame):
            packets += self.codec.encode(frame)

        if self.first_packet_pts is None and packets:
            self.first_packet_pts = packets[0].pts

        if packets:
            return [bytes(p) for p in packets], packets[0].pts - self.first_packet_pts
        else:
            return [], None

    def pack(self, packet: Packet) -> tuple[list[bytes], int]:
        timestamp = convert_timebase(packet.pts, packet.time_base, TIME_BASE)
        return [bytes(packet)], timestamp
