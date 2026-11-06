from barkmodel import BarkSpeech

speech = BarkSpeech()

speech.generate(
    text="Hello! This is a test of Bark running on an NVIDIA GPU.",
    voice="v2/en_speaker_9",
    output_file="output/test.wav"
)
