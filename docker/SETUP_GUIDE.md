# WSL2 + Docker + vLLM 로컬 교사 LLM 서빙 설치 가이드

> graphrag_00_overview.md Phase 0.0 (로컬 교사 모델 서빙 환경 구축) 실행 가이드.
> 대상: RTX 3090 24GB, Windows 11 + WSL2, Docker로 vLLM(`Qwen/Qwen2.5-32B-Instruct-AWQ`) 서빙.
> 모든 명령은 **WSL Ubuntu 터미널에서 직접** 실행할 것 (sudo 비밀번호 입력이 필요해 에이전트가 대신 실행 불가).

---

## 0. 시작 전에 — 초기화가 정말 필요한가?

기존 WSL 환경(conda로 vLLM 네이티브 설치 시도했던 것)에 문제가 쌓였다면 두 가지 선택지가 있다.

| 선택지 | 언제 | 주의 |
|---|---|---|
| **A. 완전 초기화** (`wsl --unregister`) | Docker/네트워크/systemd 등 기반 자체가 의심될 때 | **이미 받아둔 ~19GB 모델 가중치(`~/.cache/huggingface`)가 통째로 삭제됨** — 재다운로드 필요 |
| **B. 부분 정리** (conda 환경만 삭제, distro는 유지) | Docker 설치만 다시 하면 될 것 같을 때 | 가중치 캐시 보존됨, 더 빠름 |

**권장: B부터 시도하고, 그래도 이상하면 A.** 아래 1장은 A를 선택했을 때만 실행.

---

## 1. (선택) WSL2 완전 초기화

Windows PowerShell(관리자 권한)에서:

```powershell
wsl --list --verbose
wsl --unregister Ubuntu          # 경고: 해당 distro의 모든 데이터 삭제됨
wsl --install -d Ubuntu          # 재설치, 완료 후 유저명/비번 설정 프롬프트 따라가기
wsl --shutdown                   # 초기 설정 마친 뒤 한 번 완전 재시작 권장
```

재설치 후 WSL Ubuntu 터미널을 새로 열고 2장부터 진행.

### 1-a. (선택, B를 골랐다면) conda 환경만 정리

```bash
rm -rf ~/miniconda3/envs/graphrag
```

가중치 캐시(`~/.cache/huggingface`)는 그대로 둔다 — Docker 컨테이너가 재사용함.

---

## 2. WSL2 기본 상태 확인

```bash
# systemd가 켜져 있어야 Docker 서비스가 정상 동작
cat /etc/wsl.conf
```

출력에 아래가 없으면 추가하고 `wsl --shutdown` 후 재시작(Windows PowerShell에서):

```ini
[boot]
systemd=true
```

GPU 확인:

```bash
nvidia-smi
```

`NVIDIA GeForce RTX 3090`이 보이고 프로세스 목록이 뜨면 정상. **여기서 안 뜨면 WSL 문제가 아니라 Windows 쪽 NVIDIA 드라이버 문제** — Windows에서 GeForce/Studio 드라이버를 최신으로 업데이트할 것 (WSL2는 Windows 호스트 드라이버를 그대로 통과시키는 구조라 별도 WSL용 드라이버 설치 불필요/금지).

---

## 3. Docker Engine 설치

`get.docker.com` 원클릭 스크립트는 이번 세션에서 **NVIDIA Container Toolkit만 설치되고 Docker Engine 자체는 조용히 실패**하는 문제를 겪었다(에러 로그도 안 남음). 아래는 Docker 공식 문서의 apt 저장소 수동 등록 방식 — 실패 지점이 명확히 보인다.

```bash
# 3-1. 충돌 가능 패키지 정리
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
  sudo apt-get remove -y "$pkg" 2>/dev/null || true
done

# 3-2. 필수 패키지
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# 3-3. Docker GPG 키
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# 3-4. 저장소 등록 (codename 자동 감지 + fallback)
UBUNTU_CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME}")"
if curl -fsS -o /dev/null "https://download.docker.com/linux/ubuntu/dists/${UBUNTU_CODENAME}/Release"; then
  DOCKER_CODENAME="$UBUNTU_CODENAME"
else
  echo "Docker 저장소에 ${UBUNTU_CODENAME} dist 없음 -> noble(24.04 LTS)로 대체"
  DOCKER_CODENAME="noble"
fi
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${DOCKER_CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 3-5. 설치
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 3-6. 검증 + 그룹 등록
sudo docker --version
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
newgrp docker   # 새 세션 없이 바로 docker 그룹 권한 반영
```

**여기까지 하고 `docker --version`이 sudo 없이도 나오면 성공.**

---

## 4. NVIDIA Container Toolkit 설치 (GPU 패스스루)

```bash
# 이미 설치돼 있으면 건너뛰어도 됨: dpkg -l | grep nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

검증:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

컨테이너 안에서 `NVIDIA GeForce RTX 3090`이 출력되면 GPU 패스스루 성공.

---

## 5. vLLM 서빙 컨테이너 기동

리포지토리 루트에서:

```bash
cd /mnt/c/chun/LLM/Graph_Rag   # Windows 경로가 WSL에서 보이는 위치
docker compose -f docker/docker-compose.yml up -d
docker compose -f docker/docker-compose.yml logs -f
```

`Uvicorn running on http://0.0.0.0:8000` 이 보이면 준비 완료. `Ctrl+C`로 로그 보기만 종료(컨테이너는 계속 실행됨).

확인:

```bash
curl http://localhost:8000/v1/models
```

`Qwen/Qwen2.5-32B-Instruct-AWQ`가 응답에 포함되면 완전히 끝난 것. `configs/eval.yaml`의 `teacher_endpoint: http://localhost:8000/v1`이 이걸 가리킨다 (Windows 쪽에서도 `localhost:8000`으로 접근 가능 — WSL2 localhost 포워딩 기본 활성화).

---

## 6. 트러블슈팅

### 6-1. 이번 세션에서 실제로 만난 문제 (Docker 이미지 사용 시 대부분 해당 없음, 참고용)

| 증상 | 원인 | 해결 |
|---|---|---|
| `RuntimeError: UVA is not available` | WSL2에서 pinned memory가 커널 조건 충족해도 기본 비활성화 | `docker-compose.yml`에 이미 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 반영됨. 네이티브(non-docker)로 돌릴 경우 커널 버전이 `4.19.121` 이상인지 `uname -r`로 확인 |
| `ValueError: ... KV cache 부족` | `--max-model-len`이 VRAM 예산보다 큼 | `docker-compose.yml`의 `--max-model-len 4096`이 RTX 3090 24GB + util 0.90 기준 실측 안전값. 필요시 더 줄이거나 `--gpu-memory-utilization`을 스펙 상한(0.90) 안에서 조정 |
| `Could not find nvcc` / `gcc 15` 비호환 / `curand.h` 없음 | 네이티브 conda 환경에 CUDA 툴킷/호환 gcc 미설치 | **공식 이미지(`vllm/vllm-openai`) 사용으로 원천 해결** — Dockerfile이 이미 이렇게 돼 있음 |
| `get.docker.com` 스크립트가 조용히 실패 | 원인 불명(로그 없음), Ubuntu 26.04가 너무 최신이라는 가설도 배제됨(저장소 실존 확인함) | 3장의 수동 설치로 대체 |

### 6-2. 앞으로 만날 수 있는 문제 (예방적 대응)

| 증상 | 원인 | 해결 |
|---|---|---|
| `docker: permission denied ... /var/run/docker.sock` | 방금 그룹에 추가돼 세션에 반영 안 됨 | `newgrp docker` 또는 WSL 터미널 재시작 |
| `docker compose up` 후 컨테이너가 바로 죽음 (`docker ps -a`로 확인 시 `Exited`) | 대부분 OOM 또는 모델 로딩 실패 | `docker compose -f docker/docker-compose.yml logs` 로 원인 확인. GPU 메모리가 이미 다른 프로세스(Windows 쪽 게임/브라우저 GPU 가속 등)에 물려 있으면 `nvidia-smi`로 확인 후 정리 |
| `Bind for 0.0.0.0:8000 failed: port is already allocated` | 8000번 포트를 이미 다른 프로세스가 사용 중 (예: 예전에 native로 띄운 vllm serve가 안 죽어있음) | `docker ps` / Windows에서 `netstat -ano \| findstr :8000` 으로 점유 프로세스 확인 후 종료, 또는 compose의 포트 매핑을 `"8001:8000"`처럼 바꾸고 `configs/eval.yaml`의 `teacher_endpoint`도 같이 변경 |
| `docker pull` / `docker compose up`이 아주 느리거나 멈춤 | WSL2 DNS 문제 (흔한 케이스) | `/etc/resolv.conf`에 `nameserver`가 있는지 확인. 없거나 이상하면 `/etc/wsl.conf`에 `[network]\ngenerateResolvConf = false` 추가 후 `/etc/resolv.conf`를 `nameserver 8.8.8.8`로 수동 작성, `wsl --shutdown` 후 재시작 |
| 모델 다운로드가 매번 처음부터 다시 됨 | 볼륨 마운트 경로가 실제 캐시 위치와 다름 | `docker-compose.yml`의 `${HF_CACHE_DIR:-~/.cache/huggingface}`가 실제 WSL 홈(`/home/<사용자명>/.cache/huggingface`)을 가리키는지 확인. `echo ~/.cache/huggingface`로 실제 展開 경로 확인 |
| Windows 쪽에서 `localhost:8000` 접속 안 됨 | WSL2 localhost 포워딩이 꺼져 있거나 방화벽 | 최신 WSL2는 기본 활성화. 안 되면 `.wslconfig`(`%UserProfile%\.wslconfig`)에 `[wsl2]\nlocalhostForwarding=true` 추가 후 `wsl --shutdown` |
| `nvidia-container-toolkit` 설치 후에도 6장 GPU 테스트 실패 | Docker 데몬이 재시작 전이거나 `nvidia-ctk` 설정이 `daemon.json`에 반영 안 됨 | `cat /etc/docker/daemon.json`에 `"nvidia"` 런타임이 있는지 확인, 없으면 `sudo nvidia-ctk runtime configure --runtime=docker` 재실행 후 `sudo systemctl restart docker` |
| WSL이 메모리를 너무 많이/적게 씀 | `.wslconfig` 미설정 | `%UserProfile%\.wslconfig`에 아래 추가 후 `wsl --shutdown`:<br>`[wsl2]`<br>`memory=24GB`<br>`processors=16` (RAM 31GB/24코어 기준 여유 두고 설정, 필요시 조정) |

---

## 7. 완료 체크리스트

- [ ] `docker --version` sudo 없이 실행됨
- [ ] `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` 에서 RTX 3090 출력됨
- [ ] `docker compose -f docker/docker-compose.yml up -d` 후 `Uvicorn running` 로그 확인
- [ ] `curl http://localhost:8000/v1/models` 정상 응답
- [ ] `configs/eval.yaml`의 `teacher_endpoint`/`teacher_model`과 실제 기동값 일치 확인

모두 체크되면 TODO.md Milestone 0의 "Phase 0.0 vLLM 엔드포인트 확인" 항목을 완료로 표시하면 된다.
