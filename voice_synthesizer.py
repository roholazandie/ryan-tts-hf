import time
import json
import numpy as np
import re
import os
import torch
from espnet2.bin.tts_inference import Text2Speech
from scipy.io.wavfile import write

class TTSConfig:

    def __init__(self,
                 model_file=None,
                 model_tag=None,
                 vocoder_file=None,
                 vocoder_tag=None,
                 voice_dir="",
                 fs="",
                 port="",
                 device="cuda"
                 ):
        self.model_file = model_file
        self.model_tag = model_tag
        self.vocoder_file = vocoder_file
        self.vocoder_tag = vocoder_tag
        self.voice_dir = voice_dir
        self.fs = fs
        self.port = port
        self.device = device

    @classmethod
    def from_dict(cls, json_object):
        config = TTSConfig()
        for key in json_object:
            config.__dict__[key] = json_object[key]
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with open(json_file) as f:
            config_json = f.read()

        return cls.from_dict(json.loads(config_json))

    def __str__(self):
        return str(self.__dict__)


class VoiceSynthesizer:
    def __init__(self, config_file):

        self.config = TTSConfig.from_json_file(config_file)

        self.cmu_phonemes = ["F", "M", "N", "L", "D", "B", "HH", "P", "T", "S", "R", "AE", "W", "Z", "V", "G", "NG",
                             "DH", "AX",
                             "AA", "AH", "AO", "AW", "AXR", "AY", "CH", "EH", "ER", "EY", "IH", "IX", "IY", "JH", "OW",
                             "OY", "SH",
                             "TH", "UH", "UW", "Y", "TS", "R", "R", "AH", "AA", "SIL", "IY", "L", "L", "R", "IH", ]

        # going from the default (having files for both model and vocoder ) to model tags and vocoder tags (downloadable)
        if self.config.model_file and self.config.vocoder_file:
            self.text2speech = Text2Speech.from_pretrained(
                model_file=self.config.model_file,
                vocoder_file=self.config.vocoder_file,
                device=self.config.device,
            )
        elif self.config.model_file and self.config.vocoder_tag:
            self.text2speech = Text2Speech.from_pretrained(
                model_file=self.config.model_file,
                vocoder_tag=self.config.vocoder_tag,
                device=self.config.device,
            )
        elif self.config.model_tag and self.config.vocoder_tag:
            self.text2speech = Text2Speech.from_pretrained(
                model_tag=self.config.model_tag,
                vocoder_tag=self.config.vocoder_tag,
                device=self.config.device,
            )
        elif self.config.model_file:
            self.text2speech = Text2Speech.from_pretrained(
                model_file=self.config.model_file,
                device=self.config.device,
            )
        elif self.config.model_tag:
            self.text2speech = Text2Speech.from_pretrained(
                model_tag=self.config.model_tag,
                device=self.config.device,
            )
        else:
            raise Exception("The model_file or model_tag is not provided")

    def extract_phonemes(self, text, cmu_syle=True):
        phonemes = self.text2speech.preprocess_fn.tokenizer.text2tokens(text)

        if cmu_syle:
            cleaned_phonemes = []
            for phone in phonemes:
                cleaned_phone = re.sub(r'\d+', '', phone)
                cleaned_phonemes.append(cleaned_phone)
            return cleaned_phonemes
        return phonemes

    def regulate_phoneme_duration(self, phoneme, start, end):
        for char in ['0', '1', '2', '3']:
            if char in phoneme:
                phoneme = phoneme.replace(char, '')
        if phoneme not in self.cmu_phonemes:
            phoneme = "SIL"

        start = int(float(start) / 10) + 10
        end = int(float(end) / 10) + 10
        return phoneme, start, end

    def tts(self, input_text):
        # synthesis
        with torch.no_grad():
            start = time.time()
            output = self.text2speech(input_text)

        # extract phonemes
        phonemes = self.extract_phonemes(input_text)

        wav = output['wav']
        # print("Created phonemes successfully")
        y = wav.view(-1).cpu().tolist()
        durations = output['duration'].cpu().tolist()
        durations = durations[1:]
        # print("Created durations")
        rtf = (time.time() - start) / (len(wav) / self.config.fs)
        print(f"RTF = {rtf:5f}")

        audio_duration = (len(y) / self.config.fs) * 1000

        unit_duration = audio_duration / sum(durations)

        ends = np.cumsum(durations) * unit_duration
        starts = [0] + ends[:-1].tolist()

        lines = []
        for phoneme, start, end in zip(phonemes, starts, ends):
            phoneme, start, end = self.regulate_phoneme_duration(phoneme, start, end)
            line = "{:4d} 0    0    0    0  {:4d} {:4s} 0.0000 ".format(start, end, phoneme) + '\n'
            # file_writer.write(line)
            lines.append(line)
            # phoneme_out["phonemes"].append(phoneme)
            # phoneme_out["start"].append(start)
            # phoneme_out["end"].append(end)

        # let us listen to generated samples
        # NOTE: Wav file data: wav.view(-1).cpu().numpy()
        wav_file = os.path.join(self.config.voice_dir, "x.wav")

        # wav_data = wav.view(-1).cpu().numpy()
        # print("wav_data type: {}".format(type(wav_data[0])))
        # print("wav_data type: {}".format(type(wav_data[0])))

        write(wav_file, self.text2speech.fs, wav.view(-1).cpu().numpy())
        # print("writing to file complete")

        # return {"phonemes": " ".join(lines)}
        # print("returning phonemes")
        print(f"wav type: {type(wav)}")
        return {"phonemes": " ".join(lines), "wav": wav.view(-1).cpu().numpy()}


if __name__ == '__main__':
    string = "Hey How are you doing today? I am happy."
    vs = VoiceSynthesizer(config_file="configs/female_voice_config.json")
    response = vs.tts(string)
    print(response)
