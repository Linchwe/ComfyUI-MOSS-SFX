ComfyUI-MOSS-SFX (버전 1.0)

MOSS-SoundEffect v2.0 을 ComfyUI 에서 쓰는 작은 노드입니다.
모델을 돌리는 코드는 OpenMOSS 의 moss_soundeffect_v2 패키지(Apache-2.0)가 맡고,
이 노드는 그 패키지를 불러서 결과를 ComfyUI 의 AUDIO 로 바꿔 주는 역할만 합니다.


[설치]
아래 명령은 모두 ComfyUI_windows_portable 폴더(run_nvidia_gpu.bat 이 있는 곳)에서 연 cmd 창에서 한 줄씩 실행합니다.
(탐색기로 그 폴더를 열고, 위쪽 주소창에 cmd 라고 입력한 뒤 엔터를 누르면 됩니다)
이 노드는 설치할 때 git 이 필요합니다. git --version 을 입력해서 버전이 나오는지 확인하세요.

1. 이 노드를 ComfyUI\custom_nodes 안에 넣습니다. git 으로 받으려면 아래 줄을 실행합니다.

git clone https://github.com/Linchwe/ComfyUI-MOSS-SFX ComfyUI\custom_nodes\ComfyUI-MOSS-SFX

   압축 파일(Code > Download ZIP)로 받았다면 풀어서 넣습니다.
   (ComfyUI\custom_nodes\ComfyUI-MOSS-SFX\nodes.py 가 되어야 합니다)

2. 설치 전 현재 상태를 기록합니다. (이미 pip_before.txt 가 있으면 이 줄은 실행하지 마세요)

python_embeded\python.exe -m pip list --format=freeze > pip_before.txt

   이 파일은 지우지 마세요. 아래 설치 줄 끝의 -c pip_before.txt 가 "이미 설치된 프로그램의 버전은 바꾸지 말 것"을 지키게 하고,
   문제가 생겼을 때 python_embeded\python.exe -m pip install -r pip_before.txt 로 되돌리는 데도 씁니다.

3. 필요한 프로그램을 설치합니다. 4줄을 순서대로, 한 줄씩 실행합니다. 끝의 -c pip_before.txt 를 빼지 마세요.

python_embeded\python.exe -m pip install --no-deps git+https://github.com/OpenMOSS/MOSS-TTS.git#subdirectory=moss_soundeffect_v2 -c pip_before.txt

python_embeded\python.exe -m pip install ftfy einops diffusers soundfile imageio regex -c pip_before.txt

python_embeded\python.exe -m pip install --no-deps descript-audiotools==0.7.2 -c pip_before.txt

python_embeded\python.exe -m pip install argbind pyloudnorm importlib-resources julius ffmpy ipython rich matplotlib librosa pystoi torch_stoi flatten-dict markdown2 randomname tensorboard -c pip_before.txt

   설치 중 빨간색 ERROR 와 함께 버전 불일치 문구가 나와도 마지막에 Successfully installed 가 나오면 정상입니다.
   설치 확인: python_embeded\python.exe -m pip show moss-soundeffect-v2  (Name: moss-soundeffect-v2 가 나오면 성공)

4. 모델을 다운로드합니다. (약 11GB)

python_embeded\python.exe -c "import sys, huggingface_hub as h; h.snapshot_download(sys.argv[1], local_dir=sys.argv[2])" OpenMOSS-Team/MOSS-SoundEffect-v2.0 ComfyUI\models\moss_soundeffect_v2\MOSS-SoundEffect-v2.0

   ComfyUI\models\moss_soundeffect_v2\MOSS-SoundEffect-v2.0 안에 model_index.json 이 있으면 정상입니다.

5. ComfyUI 를 완전히 껐다가 다시 켭니다.

자세한 설명과 오류 해결법은 [Audio] Text To SFX 튜토리얼을 보세요.


[노드]  (빈 곳 더블클릭 > MOSS SFX 검색)
MOSS SFX Loader -> MOSS SFX Generate -> Save Audio(기본 노드)

- memory_mode : 8GB 카드는 low_vram, 12GB 이상은 standard 도 됩니다. 잘 모르겠으면 low_vram 으로 두세요.
- dit_dtype : RTX 30 시리즈 이상은 bfloat16(기본값), RTX 20 시리즈는 float16 으로 바꾸세요. 둘 다 메모리 사용량은 같습니다.
  float32 는 메모리를 두 배 가까이 써서 8GB 카드에서는 쓰지 마세요.
- keep_loaded : 끄면 생성이 끝날 때 모델을 VRAM 에서 내립니다. 다음 생성 때 다시 불러오느라 시간이 걸립니다.
- unload_other_models : 켜 두면(기본값) 생성 전에 ComfyUI 가 들고 있는 다른 모델(앞에서 쓴 이미지 생성 모델 등)을 VRAM 에서 내립니다. 8GB 카드는 꼭 켜 두세요.
- 프롬프트는 영어와 중국어만 됩니다. 한국어는 인식되지 않습니다.


[참고]
- 첫 생성은 모델을 불러오느라 약 3분 걸립니다. 두 번째부터는 모델이 이미 올라가 있어서 다시 읽지 않습니다.
- low_vram 모드에서 생성 중 VRAM 사용량은 약 4~5GB 였습니다. (RTX 3060 12GB 에서 측정)
- CPU RAM 은 16GB 이상을 권장합니다.
- 8GB 카드와 RTX 20 시리즈 설정은 아직 충분히 검증되지 않았습니다.
- 콘솔에 [MOSS-SFX] This GPU has no bfloat16 hardware 가 나오면 dit_dtype 을 float16 으로 바꾸세요.


[라이선스]
이 노드는 MIT 라이선스입니다. (LICENSE 파일 참고)
OpenMOSS 의 moss_soundeffect_v2 패키지와 MOSS-SoundEffect v2.0 모델은 Apache-2.0 입니다. (NOTICE 파일 참고)
