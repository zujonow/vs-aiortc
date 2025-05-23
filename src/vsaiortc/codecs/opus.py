import fractions
from typing import List, Tuple

from av import AudioFrame, AudioResampler
from av.frame import Frame
from av.packet import Packet

from ..jitterbuffer import JitterFrame
from ..mediastreams import convert_timebase
from ._opus import ffi, lib
from .base import Decoder, Encoder

CHANNELS = 2
SAMPLE_RATE = 48000
SAMPLE_WIDTH = 2
SAMPLES_PER_FRAME = 960
TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)


class OpusDecoder(Decoder):
    def __init__(self, audio_ptime, sample_rate) -> None:
        self.audio_ptime = self.build_ptime(audio_ptime*1000)/1000 if audio_ptime!=None else 0.020
        self.sample_rate = sample_rate if sample_rate!=None else SAMPLE_RATE
        self.samples_per_frame = int(self.audio_ptime*self.sample_rate)
        self.time_base = fractions.Fraction(1, self.sample_rate)        
        
        error = ffi.new("int *")
        self.decoder = lib.opus_decoder_create(SAMPLE_RATE, CHANNELS, error)
        assert error[0] == lib.OPUS_OK

    def build_ptime(self, audio_ptime)->int:
        if((audio_ptime/10)%2==0):
            return audio_ptime
        else:
            return audio_ptime+10
        
    def __del__(self) -> None:
        lib.opus_decoder_destroy(self.decoder)

    # original
    def decode(self, encoded_frame: JitterFrame) -> List[Frame]:
        frame = AudioFrame(format="s16", layout="stereo", samples=self.samples_per_frame)
        frame.pts = encoded_frame.timestamp
        frame.sample_rate = SAMPLE_RATE
        frame.time_base = self.time_base
        
        length = lib.opus_decode(
            self.decoder,
            encoded_frame.data,
            len(encoded_frame.data),
            ffi.cast("int16_t *", frame.planes[0].buffer_ptr),
            self.samples_per_frame,
            0,
        )
        assert length == self.samples_per_frame
        return [frame]


class OpusEncoder(Encoder):
    def __init__(self) -> None:
        error = ffi.new("int *")
        self.encoder = lib.opus_encoder_create(
            SAMPLE_RATE, CHANNELS, lib.OPUS_APPLICATION_VOIP, error
        )
        assert error[0] == lib.OPUS_OK

        self.cdata = ffi.new(
            "unsigned char []", SAMPLES_PER_FRAME * CHANNELS * SAMPLE_WIDTH
        )
        self.buffer = ffi.buffer(self.cdata)

        # Create our own resampler to control the frame size.
        self.resampler = AudioResampler(
            format="s16",
            layout="stereo",
            rate=SAMPLE_RATE,
            frame_size=SAMPLES_PER_FRAME,
        )

    def __del__(self) -> None:
        lib.opus_encoder_destroy(self.encoder)

    def encode(
        self, frame: Frame, force_keyframe: bool = False
    ) -> Tuple[List[bytes], int]:
        assert isinstance(frame, AudioFrame)
        assert frame.format.name == "s16"
        assert frame.layout.name in ["mono", "stereo"]

        # Send frame through resampler and encoder.
        payloads = []
        timestamp = None
        for frame in self.resampler.resample(frame):
            data = bytes(frame.planes[0])
            length = lib.opus_encode(
                self.encoder,
                ffi.cast("int16_t*", ffi.from_buffer(data)),
                SAMPLES_PER_FRAME,
                self.cdata,
                len(self.cdata),
            )
            assert length > 0

            payloads.append(self.buffer[0:length])
            if timestamp is None:
                timestamp = frame.pts

        return payloads, timestamp

    def pack(self, packet: Packet) -> Tuple[List[bytes], int]:
        timestamp = convert_timebase(packet.pts, packet.time_base, TIME_BASE)
        return [bytes(packet)], timestamp
