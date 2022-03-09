[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc/4.0/)

## Ryan-TTS-HF
The code to run Ryan Text-to-Speech using the new API supported by HuggingFace. This code can be used with off the shelf vocoder or with a custom vocoder.

## RyanSpeech dataset
Get the dataset from [here](http://mohammadmahoor.com/ryanspeech-request-form/)

## Installation

```
mkdir -p outputs/phoneme_files
mkdir outputs/wav_files
```

```
pip install -r requirements.txt
```
and install the latest torch

download the models and change the config.json to point to them.

Note: if you want to move the model files to a different location, you should change the paths in config.yaml for each model.

## Usage
Models and Vocoders can be trained and used in config as ```model_file``` and ```vocoder_file``` respectively. But if you don't have
any models or vocoders, you can use the list of models for ```model_tag``` as in the following list.

```
Models:
- kan-bayashi/ljspeech_tacotron2
- kan-bayashi/ljspeech_fastspeech
- kan-bayashi/ljspeech_fastspeech2
- kan-bayashi/ljspeech_conformer_fastspeech2
- kan-bayashi/ljspeech_joint_finetune_conformer_fastspeech2_hifigan
- kan-bayashi/ljspeech_joint_train_conformer_fastspeech2_hifigan
- kan-bayashi/ljspeech_vits"
```
and the list of vocoders for ```vocoder_tag``` in the following list.
```
- none
- parallel_wavegan/ljspeech_parallel_wavegan.v1
- parallel_wavegan/ljspeech_full_band_melgan.v2
- parallel_wavegan/ljspeech_multi_band_melgan.v2
- parallel_wavegan/ljspeech_hifigan.v1
- parallel_wavegan/ljspeech_style_melgan.v1
```
Note that when you use ```model_tag``` the value for ```model_file``` should be empty. The same applies to ```vocoder_tag```.
But ```model_file``` can be used with ```vocoder_tag``` or you can just leave vocoder specification empty.
If you use ```none``` the algorithm will be the ```griffin-lim``` vocoder.


### Abbreviations and acronyms
The code supports adding abbreviations and acronyms to the default cmu dictionary in runtime. 
- For abbreviations like IDC which should be read as "I don't care" you should first use the following code to get the phonemes.
```
>> from g2p_en import G2p
>> g2p = G2p()
>> g2p("I don't care")
['AY1', ' ', 'D', 'OW1', 'N', 'T', ' ', 'K', 'EH1', 'R']
```
and add it to abbreviations.json file. 
- For acronyms like NFL just use the tag as follows:
```
The <say-as interpret-as='acronym'> AMD </say-as> stands for Advanced Micro Devices.
```

- For Roman numerals and other characters you can follow the rules of phonemes. For example "IV" can be extracted 
by feeding "four" to the g2p and then adding it to the abbreviations.json file.
- For other words with no corresponding phonemes, first lookup IPA pronunciation in wikipedia then you can use the following code to get the phonemes.
```
>> from arpabetandipaconvertor.phoneticarphabet2arpabet import PhoneticAlphabet2ARPAbetConvertor
>> ipa_convertor = PhoneticAlphabet2ARPAbetConvertor()
>> ipa_convertor.convert('hæmedɑ:n')
HH AE0 M EY0 D AA0 N
```
This doesn't work perfectly in all cases. So for correction see the table below.

- Vowels

1-letter	|2-letter		|  IPA	| Example	                    |   ARPAbet
----------|------------|------|-----------------------------------------------------|-----------
a	   		|    AA		|   ɑ 	|  balm 	      |   B AA1 M
@	   		|    AE	  	|   æ 	|  bat   	              |   B AE1 T
A	   		|    AH	  	|   ʌ 	|  butt  	                |   B AH1 T
c    		|    AO	  	|   ɔ	|  bought	           |   B AO1 T
W	   		|    AW	  	|   aʊ	|   bout             |   B AW1 T
x	   		|    AX	  	|   ə	|   about          |   AH0 B AW1 T
N/A	   		|    AXR	|   ɚ 	|   letter            |   L EH1 T ER0
Y	   		|    AY	 	|   aɪ	|   bite              |   B AY1 T
E	   		|    EH	 	|   ɛ	|   bet              |   B EH1 T
R    		|    ER	 	|   ɝ	|   bird           |   B ER1 D
e	    	|    EY		|   eɪ	|   bait              |   B EH1 T
I	   		|    IH	  	|   ɪ	|   bit           |   B IH1 T
X	   		|    IX	  	|   ɨ	|   roses           |   R OW1 Z
i	    	|    IY	 	|   i	|   beat             |   B IY1 T
o	    	|    OW	  	|   oʊ	|   boat         |   B OW1 T
O	    	|    OY	 	|   ɔɪ	|   boy                 |   B OY1
U       	|	  UH    	|   ʊ	|   book              |   B UH1 K
u	    	|    UW		|   u	|   boot            |   B UW1 T
N/A	    	|    UX  	|   ʉ	|   dude          |  D UW1 D


- Consonants

1-letter	|  2-letter    	| IPA	|  Example                   |  ARPABET
----------|----------------|-----|--------------------------------------------------|---------
b	  		|   B	       	|  b  	|  buy            |  B AY1
C	    	|   CH	       	|  tʃ 	|  China    |  CH AY1 N AH0
d	    	|   D	       	|  d	|  die           |  D AY1
D	    	|   DH	       	|  ð	|  thy              |  DH AY1
F	    	|   DX	       	|  ɾ	|  butter       |  B AH1 T ER0
L	    	|   EL	       	|  l̩	|  bottle         |  B AA1 T AH0 L
M	    	|   EM	       	|  m̩	|  rhythm       |  R IH1 DH AH0 M
N	    	|   EN	       	|  n̩	|  button         |  B AH1 T AH0 N
f	    	|   F	       	|  f	|  fight       |  F AY1 T
g	    	|   G	       	|  ɡ	|  guy        |  G AY1
h	    	|   HH or H 	|  h  	|	 high            |  HH AY1
J	    	|   JH	       	|  dʒ 	|  jive           |  JH AY1 V
k	    	|   K	       	|  k	|  kite             |  K AY1 T
l	    	|   L	       	|  l	|  lie             |  L AY1
m	    	|   M	       	|  m	|  my              |  M AY1
n       	|	N	       	|  n	|  nigh           |  N AY1
G	    	|   NX or NG|  ŋ  	|	 sing          |  S IH1 NG
N/A     	|	NX   		|  ɾ̃	|  winner            |  W IH1 N ER0
p	    	|   P	       	|  p	|  pie            |  P AY1
Q	    	|   Q	       	|  ʔ	|  uh-oh            |
r	    	|   R	       	|  ɹ	|  rye       |  R AY1
s	    	|   S          	|  s	|  sigh         |  S AY1
S	    	|   SH	       	|  ʃ  	|  shy           |  SH AY1
t	    	|   T	       	|  t	|  tie            |  T AY1
T	    	|   TH	       	|  θ	|  thigh           |  TH AY1
v       	|	V	       	|  v  	|  vie         |  V AY1
w	    	|   W	       	|  w	|  wise         |  W AY1 Z
H	    	|   WH	       	|  ʍ	|  why      |  HH W AY1, W AY1
y	    	|   Y	       	|  j	|  yachting       |  Y AA1 T IH0 NG
z	    	|   Z	       	|  z	|  zoo              |  Z UW1
Z	    	|   ZH	       	|  ʒ	|  pleasure     |  P L EH1 ZH ER0

### Breaks (Pauses)
The code supports adding pauses (breaks) to the speech. It can be specified in seconds or milliseconds.

```
"This is the first sentence. <break time='2000ms'/> This is the second sentence. <break time='1s'/> This is the third sentence."
```

(no breaks allowed in the beginning or end of the sentence)

### TODO:
- We will add ryan models as downloadable tags soon.


- Detect pronounceable sequences of characters [this](https://stackoverflow.com/questions/40209592/arranging-letters-in-the-most-pronounceable-way) 
and generating unprounacable gibberish for dataset [here]()

- Add '.' at the end of acronyms for a better pronunciation