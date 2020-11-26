## Interaction with GPT-2 network with ranking and speech recognition

### Downloading model files

As a base for this instruction, microsoft-dialoGPT is used.

1. Download [config.json](https://storage.googleapis.com/kuberlab/gpt2/config.json), [vocab.json](https://storage.googleapis.com/kuberlab/gpt2/vocab.json) and [merges.txt](https://storage.googleapis.com/kuberlab/gpt2/merges.txt) files.
2. Download model state dict file, currently there a couple of them available:
   * [rick_morty.bin](https://storage.googleapis.com/kuberlab/gpt2/rick_morty.bin) - trained on Rick and Morty series dialogs
   * [dialoGPT-medium](https://storage.googleapis.com/kuberlab/gpt2/dialoGPT-medium.bin) - microsoft DialoGPT-medium
   * You can search more models at [https://github.com/microsoft/DialoGPT](https://github.com/microsoft/DialoGPT), 
   but don't forget to download another config.json and vocab/merges files (if applicable).
3. Save all these files into one directory and run:

```bash
python gpt_script.py --tokenizer <dir> --config_name <dir>/config.json --mode interact --state_dict <dir>/model_file.bin
```

### Usage with ranker models

Ranker models at [https://github.com/golsun/DialogRPT](https://github.com/golsun/DialogRPT)

There are several ranking transformer models, each one does its own task (see the DialogRPT repo for the details):

- updown ([link](https://xiagnlp2.blob.core.windows.net/dialogrpt/updown.pth))
- width ([link](https://xiagnlp2.blob.core.windows.net/dialogrpt/width.pth))
- depth ([link](https://xiagnlp2.blob.core.windows.net/dialogrpt/depth.pth))
- human_vs_rand ([link](https://xiagnlp2.blob.core.windows.net/dialogrpt/human_vs_rand.pth))
- human_vs_machine ([link](https://xiagnlp2.blob.core.windows.net/dialogrpt/human_vs_machine.pth))

* Download any of those models, or a couple;

You could run the script with argument `--ranker <path-to-model.pth>` in order
to use a single ranking model, or run with `--ranker <dir>` to use "ensemble" model
where it uses several models and computes weighted average on all available models.
To achieve that, follow the next additional steps below:

* Save the model files into separate directory
* Copy ensemble.yml model config: `cp ranking/ensemble.yml <dir>/`
* Then you can run the script with `--ranker <dir>`, it will use all available
models here (regardless if some of them just don't exist in the directory).

### Usage speech API

 * Setup authentication for google - [https://googleapis.dev/python/google-api-core/latest/auth.html](https://googleapis.dev/python/google-api-core/latest/auth.html)
 * Run `pip install sounddevice google-cloud-speech webrtcvad pynput`
 * If it is failed at installing `sounddevice`, it may be required to install `libportaudio2` system package. On Ubuntu:
   * `sudo apt install libportaudio2`
 * Run with options `--speech_creds default`

Or, alternatively, you may want to use credentials .json file so use `--speech_creds <credentials.json>` 
