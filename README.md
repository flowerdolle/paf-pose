# PAF-Pose

**Part-Aware Fusion Whole-Body 3D Pose Estimator**

RGB 영상 한 편을 입력하면 body / hand / face 파트별로 선택한 3D 자세 추정 모델을
각각의 Docker 컨테이너에서 실행하고, 그 결과를 공통 관절 레이아웃으로 변환한 뒤
파트 인식 융합(part-aware fusion)으로 하나의 전신 3D 골격 시퀀스를 출력하는 프로그램입니다.

논문: *Part-Aware Algorithm Fusion for Whole-Body 3D Pose Estimation* (ICCAS 2026)

```bash
pafpose run --video input.mp4 --body pear --hand wilor --face pear --out result/
pafpose run --video videos/ --preset accuracy --out result/
```

## 목차

1. [개요](#1-개요)
2. [요구 환경](#2-요구-환경)
3. [설치](#3-설치)
4. [가중치 준비](#4-가중치-준비)
5. [실행](#5-실행)
6. [입력 규격](#6-입력-규격)
7. [출력 규격](#7-출력-규격)
8. [지원 백엔드](#8-지원-백엔드)
9. [구조](#9-구조)
10. [라이선스와 외부 모델](#10-라이선스와-외부-모델)

## 1. 개요

RGB 영상에서 전신 3D 자세를 뽑을 때, body·hand·face는 각각 정확도, 프레임 커버리지, 속도의 균형이 다릅니다.
PAF-Pose는 파트마다 다른 모델을 고를 수 있게 하고, 그 결과를 하나의 전신 골격으로 합칩니다. 처리 순서는 세 단계입니다.

1. **백엔드 실행.** 선택한 body / hand / face 모델을 각각의 Docker 컨테이너에서 실행합니다. 모델마다 실행 환경이 달라도
   컨테이너로 격리되므로 한 머신에서 함께 쓸 수 있습니다. 같은 모델이 여러 파트에 선택되면 한 번만 실행됩니다.
2. **공통 레이아웃 변환.** 각 모델의 고유 관절 순서와 좌표계를 공통 레이아웃(body8 + eye2, hands42, face70)과
   공통 좌표계로 바꿔 저장합니다. 추정에 실패한 프레임은 NaN과 유효 마스크로 표시합니다.
3. **파트 인식 융합.** 손은 손목 기준으로, 얼굴은 두 눈 중점 기준으로 body 소스에 붙입니다. 스케일은 영상당 한 번,
   body 소스의 손 뼈 길이(손)와 눈 사이 거리(얼굴)의 중앙값 비율로 맞춥니다. 회전 정렬, 학습 기반 보정, 시간축 평활은
   하지 않습니다. 결과는 120관절 전신 시퀀스와 프레임별 유효 마스크입니다.

지원 백엔드는 SAM 3D Body, PEAR, WiLoR, TEASER, MediaPipe이며, 정확도·균형·속도 프리셋 세 가지가 준비되어 있습니다.

## 2. 요구 환경

- Linux x86_64, NVIDIA GPU (검증 환경: A40, driver 535)
- Docker 24+ 와 NVIDIA Container Toolkit
- Python 3.10+ (호스트 CLI 전용, GPU 라이브러리 불필요)

## 3. 설치

```bash
git clone https://github.com/flowerdolle/paf-pose.git
cd paf-pose
pip install -e .
docker compose build            # 백엔드 이미지 5개 빌드
pafpose doctor                  # docker / GPU / 가중치 점검
```

첫 빌드는 외부 저장소 clone과 PyTorch 설치가 포함되어 이미지당 수십 분이 걸립니다. 특정 백엔드만 빌드하려면
`docker compose build mediapipe pear`처럼 이름을 지정합니다. 모든 Dockerfile은 저장소 루트를 빌드 컨텍스트로 사용합니다.

### 3.1 이미지 하나로 빌드하기

백엔드 5개를 이미지 하나(`pafpose/all:0.1`)에 담을 수도 있습니다. 저장소 루트의 `Dockerfile`이
백엔드마다 별도 stage에서 자기 Python·torch 버전의 가상환경을 만든 뒤 마지막 stage에 모두 모읍니다.
버전과 커밋은 `backends/*/Dockerfile`과 동일합니다.

```bash
docker compose -f docker-compose.single.yml build      # 또는: docker build -t pafpose/all:0.1 .
export PAFPOSE_REGISTRY=backends/backends-single.yaml   # 호스트 CLI가 이 이미지를 쓰도록
pafpose doctor
pafpose run --video clip.mp4 --preset balanced --out results/
```

컨테이너 안에서는 `pafpose-backend <backend> ...`가 해당 가상환경으로 어댑터를 실행합니다.

```bash
docker run --rm --gpus all -v $PWD/clip.mp4:/input/clip.mp4:ro -v $PWD/out:/output \
  -v $PAFPOSE_WEIGHTS/pear:/weights:ro pafpose/all:0.1 \
  pafpose-backend pear --video /input/clip.mp4 --out /output --weights /weights
```

이미지 하나는 빌드 명령이 한 번이고 배포가 단순한 대신, 크기가 백엔드 5개의 합(약 15~20 GB)이고
한 백엔드의 의존성 설치가 실패하면 전체 빌드가 실패합니다. 백엔드를 처음 검증할 때는 개별 이미지가
실패를 찾기 쉽고, 검증이 끝난 뒤 배포용으로 단일 이미지를 쓰는 순서를 권합니다.

## 4. 가중치 준비

PAF-Pose는 모델 가중치를 배포하지 않습니다. Docker 이미지에는 코드만 들어 있고, 모든 체크포인트는 사용자의 머신에 있는 폴더를 컨테이너의 `/weights`로 읽기 전용 마운트해서 씁니다. 아래에 백엔드마다 필요한 파일, 받는 곳, 놓을 위치를 정리했습니다.

### 4.1 가중치 폴더 정하기

기본 위치는 저장소 안의 `./weights` (git에서 제외됨)입니다. 다른 곳에 두려면 환경 변수를 설정합니다.

```bash
export PAFPOSE_WEIGHTS=/data/pafpose-weights     # 선택 사항
```

아래 경로는 모두 이 폴더 기준입니다. 최종 구조는 다음과 같습니다.

```text
weights/
├── sam3dbody/
│   ├── sam-3d-body-dinov3/model.ckpt
│   ├── sam-3d-body-dinov3/model_config.yaml
│   ├── sam-3d-body-dinov3/assets/mhr_model.pt
│   └── moge-2-vitl-normal/model.pt
├── pear/
│   ├── pear/pear_model.pt
│   ├── smplx/SMPLX_NEUTRAL_2020.npz
│   └── flame/generic_model.pkl
├── wilor/
│   └── pretrained_models/
│       ├── wilor_final.ckpt
│       ├── detector.pt
│       ├── mano_mean_params.npz
│       └── MANO_RIGHT.pkl
└── teaser/
    ├── TEASER.pt
    └── FLAME2020/generic_model.pkl
```

`mediapipe`는 가중치가 필요 없습니다. `.task` 모델은 이미지 빌드 때 포함됩니다.

### 4.2 공개 파일 자동 다운로드

```bash
pip install huggingface_hub gdown        # 스크립트가 쓰는 도구
scripts/download_weights.sh weights      # 또는 scripts/download_weights.sh "$PAFPOSE_WEIGHTS"
```

공개된 파일은 모두 내려받고, 직접 받아야 하는 파일은 `MISSING`으로 표시합니다. 백엔드 하나만 받으려면 `backends/pear/download_weights.sh weights`처럼 실행합니다.

### 4.3 라이선스 동의가 필요한 파일

아래 모델들은 각자의 라이선스(대부분 연구용)로 배포되어 재배포할 수 없습니다. 각 사이트에 한 번 가입한 뒤 압축 파일을 받아, 표시된 파일 하나를 지정 경로에 복사하면 됩니다.

**SAM 3D Body** (백엔드 `sam3dbody`)

1. Hugging Face에 로그인한 뒤 https://huggingface.co/facebook/sam-3d-body-dinov3 를 엽니다.
2. **Agree and access repository**를 눌러 Meta 라이선스에 동의합니다.
3. 사용 중인 머신에서 `huggingface-cli login`을 실행하거나 `export HF_TOKEN=hf_...`를 설정합니다.
4. `backends/sam3dbody/download_weights.sh weights`를 다시 실행하면 `model.ckpt`, `model_config.yaml`, `assets/mhr_model.pt`가 `weights/sam3dbody/sam-3d-body-dinov3/`에 받아집니다. MoGe-2 파일(`moge-2-vitl-normal/model.pt`)은 공개 파일이라 4.2에서 이미 받아졌습니다.

수동으로 받을 때는 저장소의 **Files** 탭에서 세 파일을 받아 위 구조대로 놓습니다.

**SMPL-X** (백엔드 `pear`)

1. https://smpl-x.is.tue.mpg.de/ 에 가입하고 라이선스에 동의합니다.
2. **Download** → **SMPL-X v1.1 (NPZ+PKL)** (`models_smplx_v1_1.zip`)을 받습니다.
3. 압축 안의 `models/smplx/SMPLX_NEUTRAL_2020.npz`를 `weights/pear/smplx/SMPLX_NEUTRAL_2020.npz`에 복사합니다.

**FLAME 2020** (백엔드 `pear`, `teaser`)

1. https://flame.is.tue.mpg.de/ 에 가입하고 라이선스에 동의합니다.
2. **Download** → **FLAME 2020** (`FLAME2020.zip`)을 받습니다.
3. 압축 안의 `generic_model.pkl`을 **두 곳** 모두에 복사합니다. `weights/pear/flame/generic_model.pkl`과 `weights/teaser/FLAME2020/generic_model.pkl`입니다. 같은 파일이므로 심볼릭 링크로 연결해도 됩니다.

**MANO** (백엔드 `wilor`)

1. https://mano.is.tue.mpg.de/ 에 가입하고 라이선스에 동의합니다.
2. **Download** → **Models & Code** (`mano_v1_2.zip`)를 받습니다.
3. 압축 안의 `mano_v1_2/models/MANO_RIGHT.pkl`을 `weights/wilor/pretrained_models/MANO_RIGHT.pkl`에 복사합니다.

MANO 라이선스에 동의했다면 `backends/wilor/download_weights.sh weights --with-mano-mirror`로 WiLoR-mini Hugging Face 미러에서 같은 파일을 받을 수도 있습니다.

**TEASER 체크포인트** (백엔드 `teaser`)

`TEASER.pt`는 Google Drive에 공개되어 있어 스크립트가 `gdown`으로 받습니다. Drive 용량 제한 등으로 실패하면 https://drive.google.com/drive/folders/1WhTjAZIQBCZqDRziu8_ZBtMC736K9T2A 에서 직접 받아 `weights/teaser/TEASER.pt`에 놓습니다.

**PEAR, WiLoR 체크포인트**

Hugging Face에 공개되어 있어(`BestWJH/PEAR_models`, `warmshao/WiLoR-mini`) 스크립트가 받습니다.

### 4.4 확인

```bash
pafpose doctor --weights weights
```

`weights <백엔드>` 행이 모두 `OK`여야 합니다. `pafpose run`은 파일이 빠진 백엔드를 실행하지 않고 어떤 파일이 없는지 알려줍니다.

### 4.5 라이선스

| 자산 | 라이선스 | 상업적 사용 |
| --- | --- | --- |
| SAM 3D Body (Meta) | SAM License | 라이선스 확인 필요 |
| SMPL-X, FLAME, MANO (MPI) | 연구 목적 전용 | 불가, 별도 상업 라이선스 필요 |
| PEAR, TEASER, WiLoR 체크포인트 | 각 저장소 참조 | 각 저장소 참조 |
| MoGe-2 | MIT | 가능 |
| MediaPipe 모델 | Apache 2.0 | 가능 |

PAF-Pose 자체는 이 모델들을 실행·융합하는 프로그램이며, 백엔드를 사용하는 것은 해당 모델의 라이선스를 따르는 것을 의미합니다.

## 5. 실행

```bash
# 프리셋으로 실행
pafpose run --video input.mp4 --preset balanced --out result/

# 파트별 백엔드를 직접 지정 (프리셋 위에 덮어쓰기 가능)
pafpose run --video videos/ --body pear --hand wilor --face teaser --out result/

# 실행하지 않고 docker 명령만 확인
pafpose run --video input.mp4 --preset speed --out result/ --dry-run
```

주요 옵션:

| 옵션 | 설명 |
| --- | --- |
| `--video` | `.mp4` 파일 하나 또는 `.mp4`가 들어 있는 폴더 |
| `--out` | 출력 루트. 영상마다 `<out>/<영상이름>/` 아래에 백엔드별 결과와 로그 생성 |
| `--preset` | `accuracy` (sam3dbody / sam3dbody / pear), `balanced` (pear / wilor / pear), `speed` (pear / pear / pear) 또는 yaml 경로. 순서는 body / hand / face |
| `--body`, `--hand`, `--face` | 파트별 백엔드 이름. 프리셋보다 우선 |
| `--weights` | 가중치 루트 (기본 `$PAFPOSE_WEIGHTS` 또는 `./weights`) |
| `--cpu` | 컨테이너에 GPU를 넘기지 않음 (mediapipe 전용 실행에 사용) |
| `--dry-run` | docker 명령만 출력 |
| `--keep-going` | 한 백엔드가 실패해도 나머지 영상을 계속 처리 |
| `--preview mp4` / `--preview gif` | 영상마다 3D 골격(왼쪽)과 원본 영상(오른쪽)을 나란히 그린 미리보기를 함께 생성 |

같은 백엔드가 여러 파트에 선택되면(예: `accuracy` 프리셋의 SAM 3D Body body+hand) 컨테이너는 한 번만 실행됩니다.

실행 전 점검:

```bash
pafpose doctor
```

docker, NVIDIA 런타임, 백엔드 이미지, 가중치 폴더, 레지스트리, 프리셋을 확인해 `OK` / `MISSING`으로 표시합니다.

## 6. 입력 규격

- 단일 인물이 촬영된 `.mp4` 파일 하나, 또는 `.mp4` 파일들이 들어 있는 폴더 (하위 폴더는 탐색하지 않음)
- 카메라 파라미터는 입력받지 않으며, 각 백엔드가 자체 기본값으로 추정
- 프레임 수, 해상도, fps 제한은 없지만 긴 영상은 그만큼 처리 시간이 늘어남

## 7. 출력 규격

`pafpose run`은 영상마다 `<out>/<영상이름>/` 아래에 다음을 만듭니다.

```text
<영상이름>/
├── <백엔드>/<영상이름>.npz         백엔드별 공통 레이아웃 결과
├── <백엔드>/<영상이름>.meta.json   백엔드, 고정 커밋, 프레임 수, 실행 시간
├── logs/                            컨테이너 stdout/stderr
├── selection.json                   사용한 body/hand/face 백엔드
├── fused.npz                        융합 결과
├── fusion.json                      프레임 통계, 부착 스케일, 설정
└── preview.mp4 / preview.gif        미리보기 (--preview 또는 pafpose visualize 로 생성)
```

`fused.npz` 키:

| 키 | 모양 | 의미 |
| --- | --- | --- |
| `wholebody120_xyz` | (T, 120, 3) | body8 + hands42 + face70 |
| `body8_xyz` | (T, 8, 3) | neck, R/L shoulder·elbow·wrist, mid hip |
| `eye2_xyz` | (T, 2, 3) | 부착에만 쓰인 눈 중심 2점 |
| `hands42_xyz` | (T, 42, 3) | left21 + right21, body 소스 손목에 부착 |
| `face70_xyz` | (T, 70, 3) | 68 landmark + 눈 중심 2점, body 소스 눈 중점에 부착 |
| `valid` | (T,) | body·양손·얼굴이 모두 유효한 프레임 |
| `*_valid` | (T,) | 파트별 유효 프레임 |

좌표계는 x 오른쪽, y 깊이(카메라에서 멀어지는 방향), z 위, 단위 m 입니다. 유효하지 않은 프레임은 NaN 입니다.

백엔드가 쓰는 공통 npz 키(`body8_eye2_xyz`, `hands42_xyz`, `face70_xyz`, `*_valid`)와 `meta.json` 필드는 `pafpose/schema.py`에 정의되어 있습니다.

융합 결과를 눈으로 확인하려면 미리보기를 만듭니다. 왼쪽에 3D 골격 애니메이션, 오른쪽에 원본 영상이 나란히 놓입니다.

```bash
pafpose visualize --result result/clip                       # result/clip/preview.mp4
pafpose visualize --result result/clip --format gif --stride 3 --gif-width 640
pafpose visualize --result result/clip --azim -60 --elev 20  # 시점 변경
```

컨테이너를 다시 돌리지 않고 기존 결과만 융합하려면:

```bash
pafpose fuse --body-npz result/clip/pear/clip.npz --hand-npz result/clip/wilor/clip.npz \
             --face-npz result/clip/pear/clip.npz --out result/clip
```

## 8. 지원 백엔드

| 이름 | 파트 | 출처 (고정 커밋) | GPU |
| --- | --- | --- | --- |
| `sam3dbody` | body, hand | SAM 3D Body, gaomingqi/sam-body4d 구현을 frame-wise 호출 (`21af102`) | 필요 |
| `pear` | body, hand, face | Pixel-Talk/PEAR (`e1aa1f7`) | 필요 |
| `wilor` | hand | warmshao/WiLoR-mini (`ebec42f`), 논문 실행과 동일한 패키징 | 필요 |
| `teaser` | face | Pixel-Talk/TEASER (`c235716`) | 필요 |
| `mediapipe` | body, hand, face | MediaPipe Tasks 0.10.35 | CPU |

레지스트리: `backends/backends.yaml`. 각 백엔드 폴더의 `README.md`에 어댑터 동작, 가중치 배치, 수동 실행 방법, 제한 사항이 있습니다.
백엔드 이미지는 `docker compose build`로 빌드하며, 외부 저장소는 Dockerfile 안에서 위 커밋으로 clone 됩니다.
단일 이미지 레지스트리는 `backends/backends-single.yaml`, 빌드는 3.1절을 참고하세요.

## 9. 구조

```text
pafpose/          호스트 CLI, 백엔드 레지스트리, 컨테이너 실행기, 출력 스키마, 융합, 평가 지표, 미리보기 렌더링
backends/         컨테이너 안에서 실행되는 백엔드별 어댑터, 관절 매핑, Dockerfile, 가중치 스크립트
backends/_common/ 모든 어댑터가 공유하는 영상 읽기·출력 쓰기 헬퍼
backends/_single/ 단일 이미지 안에서 백엔드를 고르는 dispatcher (`pafpose-backend`)
Dockerfile         백엔드 5개를 담은 단일 이미지 (3.1절)
configs/          프리셋
scripts/          가중치 일괄 다운로드
tools/repro/      논문 결과 재현 스크립트
tests/            단위 테스트
docs/             개발 계획
```

## 10. 라이선스와 외부 모델

본 저장소의 자체 코드는 `pafpose/`, `backends/` 아래의 어댑터·매핑·헬퍼·Dockerfile·가중치 스크립트,
`scripts/`, `tools/`, 설정 파일입니다. 외부 모델 코드는 저장소에 포함하지 않으며,
Docker 이미지 빌드 시 각 공식 저장소의 고정된 커밋을 받아옵니다.
각 외부 모델과 가중치는 해당 프로젝트의 라이선스를 따릅니다.
