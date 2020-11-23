import argparse
from concurrent import futures
import queue
import time

from google.cloud import speech as speechclient
import numpy as np
from pynput import keyboard
from scipy.io import wavfile
import sounddevice as sd
import webrtcvad

DEFAULT_SAMPLE_RATE = 32000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--credentials')

    return parser.parse_args()


class SpeechRecognizer:
    def __init__(self, speech_client_creds=None, speech_client: speechclient.SpeechClient=None,
                 max_silence_ms=1000, verbose=False):
        if not speech_client_creds and not speechclient:
            raise RuntimeError('Please provide either speech_client or speech_client_creds')
        if speech_client_creds:
            if speech_client_creds == 'default':
                self.speech = speechclient.SpeechClient()
            else:
                self.speech = speechclient.SpeechClient.from_service_account_file(speech_client_creds)
        else:
            self.speech = speech_client
        self.outdata = None
        self.queue = queue.Queue(maxsize=100)
        self.threadpool = futures.ThreadPoolExecutor(max_workers=8)
        self.stopped = False
        self.blocksize = 20  # 20ms
        self.vad = webrtcvad.Vad(2)
        self.max_silence = max_silence_ms
        self.verbose = verbose

    def detect_longest_pause(self, data):
        np_data = np.frombuffer(data, dtype=np.int16).reshape([-1, 2])
        np_1_channel = np_data[:, 0]
        np_2_channel = np_data[:, 1]

        blocksize = int(DEFAULT_SAMPLE_RATE / 1000 * self.blocksize)
        count = 0
        max_count = 0
        for i in range(0, len(np_1_channel), blocksize):
            subarr = np_1_channel[i:i + blocksize]
            subarr2 = np_2_channel[i:i + blocksize]
            is_speech = self.vad.is_speech(subarr.tobytes(), DEFAULT_SAMPLE_RATE)
            is_speech2 = self.vad.is_speech(subarr2.tobytes(), DEFAULT_SAMPLE_RATE)
            # print('!' if is_speech or is_speech2 else '.', end='')
            if not is_speech or not is_speech2:
                count += 1
                if count > max_count:
                    max_count = count
            else:
                count = 0
        # print()

        return max_count * self.blocksize

    def get_callback(self):
        def callback(indata, frames: int, time, status):
            self.queue.put(indata)

            if self.outdata is None:
                self.outdata = bytearray(indata)
            else:
                self.outdata += indata

            # Check for pause
            # wavfile.write(
            #     'test.wav',
            #     DEFAULT_SAMPLE_RATE,
            #     np.frombuffer(self.outdata, dtype=np.int16).reshape([-1, 2])
            # )
            longest_pause = self.detect_longest_pause(self.outdata)
            # print(longest_pause)
            if longest_pause >= self.max_silence:
                if self.stopped:
                    return
                if self.verbose:
                    print('[WebrtcVAD] Detected pause, stopping stream.')
                self.stopped = True

        return callback

    def generate_data(self):
        while True:
            if self.stopped:
                break
            try:
                data = self.queue.get(timeout=1)

                yield bytes(data)
            except queue.Empty:
                time.sleep(0.002)

    @staticmethod
    def wait_for_code(code):
        with keyboard.Events() as events:
            while True:
                event = events.get(0.01)
                if event is not None:
                    if event.key == code:
                        return

    def get_stream_config(self):
        cfg = speechclient.RecognitionConfig(dict(
            sample_rate_hertz=DEFAULT_SAMPLE_RATE,
            encoding=speechclient.RecognitionConfig.AudioEncoding.LINEAR16,
            audio_channel_count=2,
            language_code='en-US',
            model='default',
            use_enhanced=False,
        ))
        return speechclient.StreamingRecognitionConfig(dict(
            config=cfg,
            single_utterance=False,
            interim_results=True,
        ))

    def _record_until_stop(self, stream, max_duration):
        if self.verbose:
            print('Recording...')
        started = time.time()
        while True:
            if time.time() - started > max_duration:
                if self.verbose:
                    print(f'Max duration of {max_duration} seconds exceeded, stop.')
                self.stopped = True
                break
            if self.stopped:
                break
            time.sleep(0.002)

        stream.stop()
        stream.close()

    def _reset(self):
        self.stopped = False
        self.outdata = None

    def record_and_recognize(self, max_duration: int, need_wait_key=False, verbose=False):
        self._reset()

        stream = sd.RawInputStream(
            samplerate=DEFAULT_SAMPLE_RATE,
            blocksize=DEFAULT_SAMPLE_RATE // 2,
            channels=2,
            callback=self.get_callback(),
            dtype='int16',
        )
        if need_wait_key:
            self.wait_for_code(keyboard.Key.space)

        stream.start()
        # Start watch key pressed
        self.threadpool.submit(self._record_until_stop, stream, max_duration)

        # Start receive data and send to recognize
        stream_reqs = (
            speechclient.StreamingRecognizeRequest(audio_content=content)
            for content in self.generate_data()
        )
        cfg = self.get_stream_config()
        responses = self.speech.streaming_recognize(config=cfg, requests=stream_reqs)

        last_result = None
        for resp in responses:
            if resp.results:
                if verbose:
                    print(resp.results[0].alternatives[0].transcript)
                last_result = resp.results[0].alternatives

        if last_result:
            if verbose:
                print(f'confidence: {last_result[0].confidence}')
            return last_result[0].transcript

        return None


def main():
    args = parse_args()
    duration = 15  # seconds
    if args.credentials:
        speech = speechclient.SpeechClient.from_service_account_file(args.credentials)
    else:
        speech = speechclient.SpeechClient()

    recognizer = SpeechRecognizer(speech, verbose=True)
    print('==============================')
    print('Press Space to start recording')
    print('==============================')

    text = recognizer.record_and_recognize(duration, need_wait_key=True, verbose=True)
    print(f'Your speech: {text}')


if __name__ == '__main__':
    main()
