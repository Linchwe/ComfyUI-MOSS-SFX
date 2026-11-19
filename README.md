# ComfyUI-MOSS-SFX (version 1.0)

Small ComfyUI nodes that make a sound effect from an English prompt with **MOSS-SoundEffect v2.0** by OpenMOSS. Runs locally, works on 8 GB cards.

The model code comes from OpenMOSS's `moss_soundeffect_v2` package (Apache-2.0). These nodes only call that package and convert the result to ComfyUI `AUDIO`.

A Korean guide is available in `README_KO.txt`.

## Install

Run every command in a cmd window opened in the `ComfyUI_windows_portable` folder (the one that contains `run_nvidia_gpu.bat`), one line at a time. Git is required for the install (check with `git --version`).

1. Get the node into `ComfyUI\custom_nodes`, either with git:

```
git clone https://github.com/Linchwe/ComfyUI-MOSS-SFX ComfyUI\custom_nodes\ComfyUI-MOSS-SFX
```

   or by downloading the ZIP (Code > Download ZIP) and extracting it so that the result is `ComfyUI\custom_nodes\ComfyUI-MOSS-SFX\nodes.py`.

2. Record the current state of your Python packages. Skip this line if `pip_before.txt` already exists.

```
python_embeded\python.exe -m pip list --format=freeze > pip_before.txt
```

Keep this file. The `-c pip_before.txt` at the end of the install lines tells pip not to change versions of packages you already have. You can also roll back with `python_embeded\python.exe -m pip install -r pip_before.txt`.

3. Install the required packages. Run the 4 lines in order, one at a time, and keep `-c pip_before.txt`.

```
python_embeded\python.exe -m pip install --no-deps git+https://github.com/OpenMOSS/MOSS-TTS.git#subdirectory=moss_soundeffect_v2 -c pip_before.txt
python_embeded\python.exe -m pip install ftfy einops diffusers soundfile imageio regex -c pip_before.txt
python_embeded\python.exe -m pip install --no-deps descript-audiotools==0.7.2 -c pip_before.txt
python_embeded\python.exe -m pip install argbind pyloudnorm importlib-resources julius ffmpy ipython rich matplotlib librosa pystoi torch_stoi flatten-dict markdown2 randomname tensorboard -c pip_before.txt
```

Red version-mismatch warnings during install are expected as long as the last line says `Successfully installed`. Verify with `python_embeded\python.exe -m pip show moss-soundeffect-v2`.

4. Download the model (about 11 GB).

```
python_embeded\python.exe -c "import sys, huggingface_hub as h; h.snapshot_download(sys.argv[1], local_dir=sys.argv[2])" OpenMOSS-Team/MOSS-SoundEffect-v2.0 ComfyUI\models\moss_soundeffect_v2\MOSS-SoundEffect-v2.0
```

`ComfyUI\models\moss_soundeffect_v2\MOSS-SoundEffect-v2.0\model_index.json` must exist.

5. Fully restart ComfyUI.

## Nodes

Double-click the canvas and search for "MOSS SFX".

`MOSS SFX Loader` -> `MOSS SFX Generate` -> `Save Audio` (built-in)

- **memory_mode**: use `low_vram` for 8 GB cards. `standard` also works with 12 GB or more. If unsure, use `low_vram`.
- **dit_dtype**: `bfloat16` (default) for RTX 30 series or newer, `float16` for RTX 20 series. Both use the same memory. `float32` uses about twice the memory; avoid it on 8 GB cards.
- **keep_loaded**: if off, the model is unloaded from VRAM after each generation (slower next run).
- **unload_other_models**: on by default. Unloads models ComfyUI keeps in VRAM (e.g. an image model used earlier) before generating. Keep it on for 8 GB cards.
- Prompts: English or Chinese only.

## Notes

- The first generation takes about 3 minutes while the model loads. Later runs reuse the loaded model.
- Peak VRAM in `low_vram` mode was about 4-5 GB (measured on an RTX 3060 12GB). The 8 GB / RTX 20 series settings above are not yet fully verified.
- If the console shows `[MOSS-SFX] This GPU has no bfloat16 hardware`, set dit_dtype to float16.
- 16 GB or more of system RAM is recommended.

## License

These nodes are MIT licensed (see `LICENSE`). The `moss_soundeffect_v2` package and the MOSS-SoundEffect v2.0 model are Apache-2.0 (see `NOTICE`).
