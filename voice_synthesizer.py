import time
import json
import numpy as np
import re
import os
import torch
from espnet2.bin.tts_inference import Text2Speech
from scipy.io.wavfile import write
import ast
from bs4 import BeautifulSoup


character_to_phonemes = {"A": ['EY2'],
                         "B": ['B', 'IY2'],
                         "C": ['S', 'IY1'],
                         "D": ['D', 'IY1'],
                         "E": ['IY1'],
                         "F": ['EH1', 'F'],
                         "G": ['JH', 'IY1'],
                         "H": ['EY1', 'CH'],
                         "I": ['AY1'],
                         "J": ['JH', 'EY1'],
                         "K": ['K', 'EY1'],
                         "L": ['EH1', 'L'],
                         "M": ['EH1', 'M'],
                         "N": ['EH1', 'N'],
                         "O": ['OW1'],
                         "P": ['P', 'IY1'],
                         "Q": ['K', 'Y', 'UW1'],
                         "R": ['AA1', 'R'],
                         "S": ['EH1', 'S'],
                         "T": ['T', 'IY1'],
                         "U": ['Y', 'UW1'],
                         "V": ['V', 'IY1'],
                         "W": ['D', 'AH1', 'B', 'AH0', 'Y', 'UW0'],
                         "X": ['EH1', 'K', 'S'],
                         "Y": ['W', 'AY1'],
                         "Z": ['Z', 'IY1'],
                         }

class TTSConfig:

    def __init__(self,
                 model_file=None,
                 model_tag=None,
                 vocoder_file=None,
                 vocoder_tag=None,
                 voice_dir=None,
                 abbreviations_file=None,
                 fs="",
                 port="",
                 device="cuda"
                 ):
        self.model_file = model_file
        self.model_tag = model_tag
        self.vocoder_file = vocoder_file
        self.vocoder_tag = vocoder_tag
        self.voice_dir = voice_dir
        self.abbreviations_file = abbreviations_file
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

        self.add_abreviations_to_phonemes()

    def add_abreviations_to_phonemes(self):
        abbreviations_dict = json.loads(open(self.config.abbreviations_file).read())
        try:
            abbreviations_dict = {k.lower(): [ast.literal_eval(v)] for k, v in abbreviations_dict.items()}
        except Exception as e:
            print(e)
            raise Exception("The abbreviations file is not in the correct format")
        # initialize g2p instance
        self.text2speech.preprocess_fn.tokenizer.g2p("initialize")
        # update dict
        for word in abbreviations_dict:
            if word in self.text2speech.preprocess_fn.tokenizer.g2p.g2p.cmu:
                self.text2speech.preprocess_fn.tokenizer.g2p.g2p.cmu[word].insert(0, abbreviations_dict[word][0])
            else:
                self.text2speech.preprocess_fn.tokenizer.g2p.g2p.cmu[word] = abbreviations_dict[word]

    def add_acronyms_to_phonemes_dict(self, acronym, phonemes):
        # initialize g2p instance
        self.text2speech.preprocess_fn.tokenizer.g2p("initialize")
        if acronym in self.text2speech.preprocess_fn.tokenizer.g2p.g2p.cmu:
            self.text2speech.preprocess_fn.tokenizer.g2p.g2p.cmu[acronym].insert(0, phonemes)
        else:
            self.text2speech.preprocess_fn.tokenizer.g2p.g2p.cmu[acronym] = [phonemes]

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

    def get_text(self, input_text):
        soup = BeautifulSoup(input_text, "html.parser")
        return soup.get_text()

    def _convert_time_to_miliseconds(self, break_time):
        if break_time.endswith('ms'):
            break_time = float(break_time[:-2])
        elif break_time.endswith('s'):
            break_time = int(break_time[:-1]) * 1000
        else:
            raise ValueError("Unsupported break time format")

        return break_time

    def _flatten(self, t):
        return [item for sublist in t for item in sublist]


    def tts(self, input_text):
        # extract acronyms and add it to the cmu dictionary
        soup = BeautifulSoup(input_text, "html.parser")
        for s in soup.select('say-as'):
            if s.attrs['interpret-as'] == 'acronym':
                characters = list(filter(lambda item: item.strip(), s.text))
                acronym = ''.join(characters).lower()
                phonemes = self._flatten([character_to_phonemes[character.upper()] for character in characters])
                self.add_acronyms_to_phonemes_dict(acronym, phonemes)
                s.replace_with(s.text)

        # concatenate all consecutive strings
        soup = BeautifulSoup(str(soup), "html.parser")
        outputs = []
        all_y = []
        all_durations = []
        for s in soup.children:
            if s.name == 'break':
                break_time = self._convert_time_to_miliseconds(s['time'])
                outputs.append({"break": break_time})
            else:
                text = s.get_text()
                text = re.sub(' +', ' ', text) # remove multiple spaces
                with torch.no_grad():
                    output = self.text2speech(text)
                    y = output['wav'].view(-1).cpu().tolist()
                all_y.append(y)
                durations = output['duration'].cpu().tolist()[1:]
                all_durations.append(durations)
                outputs.append({"speech": output})

        # extract phonemes
        phonemes = self.extract_phonemes(self.get_text(input_text))

        final_y = []
        for output in outputs:
            if 'break' in output:
                break_time = output['break']
                final_y.extend([0.0] * int(np.floor(break_time * self.config.fs / 1000)))
            elif 'speech' in output:
                y = output['speech']['wav'].view(-1).cpu().numpy()
                final_y.extend(y)
        final_y = np.array(final_y)

        audio_duration = (len(self._flatten(all_y)) / self.config.fs) * 1000
        unit_duration = audio_duration / sum([sum(durations) for durations in all_durations])

        all_ends = []
        for durations in all_durations:
            ends = np.cumsum(durations) * unit_duration
            all_ends.append(ends)
        breaks = [output['break'] for output in outputs if 'break' in output]
        assert len(breaks) == len(all_ends) - 1

        for i in range(len(all_ends) - 1):
            all_ends[i + 1] += all_ends[i][-1] + breaks[i]

        ends = np.concatenate(all_ends)

        starts = [0] + ends[:-1].tolist()
        lines = []
        for phoneme, start, end in zip(phonemes, starts, ends):
            phoneme, start, end = self.regulate_phoneme_duration(phoneme, start, end)
            line = "{:4d} 0    0    0    0  {:4d} {:4s} 0.0000 ".format(start, end, phoneme) + '\n'
            lines.append(line)
        wav_file = os.path.join(self.config.voice_dir, "x.wav")
        write(wav_file, self.text2speech.fs, final_y)
        return {"phonemes": " ".join(lines), "wav": final_y}


if __name__ == '__main__':
    #s = "This is the first sentence. <break time='2000ms'/> This is the second sentence. <break time='5s'/> This is the third sentence."
    #s = "The state of CO has a lot of ski resorts."
    #s = "These materials are NSFW."
    #s = "The format is JPEG."
    s = "I love all things food. Let me rephrase that, I love all things good food. I’m picky about what I like and I’d like to say I know great food. My favorite foods include all things Italian or Mexican, anything with chocolate, caramel, fresh fruit or browned butter. I don’t know how I’d live without the comfort food classics like pizza and enchiladas, all those indulgent dishes loaded with an abundance of gooey melted cheese. I’m a meat eater. Cookie dough is my weakness. The one food I think should be illegal is canned beets, they terrify me. Sugar is my addiction. I could live in the kitchen. I like to make food look pretty, because when it looks good doesn’t it just taste better?"
    #s = "This is just a sentence. I don't know how to answer this question. <break time='1s'/> This is the second sentence. The state of CO has a lot of ski resorts."
    #s = "CBS Sports has the latest NFL Football news."
    #s = "۷Chapter VI of the book is about the history of the United States. <break time='5s'/> The book is about the history of the United States."
    #s = "The city of hamedan is located in Iran."
    #s = "The NASDAK told me to go away."
    #s = "this is <say-as interpret-as='acronym'>AMD </say-as> stands for Advanced Micro Devices. Also <say-as interpret-as='acronym'>XMR</say-as> is irrelevant. <break time='1s'/> This is the second sentence."
    #s = "The program <say-as interpret-as='acronym'>XWD</say-as> captures the content of a screen or of a window."
    #s = "The program XWD captures the content of a screen or of a window."
    vs = VoiceSynthesizer(config_file="configs/ryan_hq_voice_config.json")
    response = vs.tts(s)
    print(response)
