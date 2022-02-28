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

### TODO:
We will add ryan models as downloadable tags soon.