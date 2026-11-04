# syntax=docker/dockerfile:1
# RFD4-Proteina application image: the public build.
#
# One file covers every supported accelerator; pick one with a build argument:
#   docker build --build-arg RFD4_ACCELERATOR=cu13 -t rfd4-proteina:cu13 .   # default
#   docker build --build-arg RFD4_ACCELERATOR=cu12 -t rfd4-proteina:cu12 .
#   docker build --build-arg RFD4_ACCELERATOR=cpu  -t rfd4-proteina:cpu  .
#
# RFD4_ACCELERATOR selects the extra and the Torch wheel index, nothing else:
#   cu13 -> .[all-cuda13], Torch cu130, tmol-ops-cu13
#   cu12 -> .[all-cuda12], Torch cu126, tmol-ops-cu12
#   cpu  -> no metrics extra,  Torch cpu,   no CUDA operators
#
# Everything installs from public PyPI and the public PyTorch index. `--no-sources`
# ignores [tool.uv.sources] in pyproject.toml, so no Git or SSH dependency is reachable
# from this build and it needs no submodule and no internal credentials. TMol and its
# operator packages come from published wheels only, never a source build.
#
# Network sources used by this build (all must be whitelisted on the runner):
#   deb.debian.org (apt), pypi.org + files.pythonhosted.org (pip, uv, all wheels),
#   download.pytorch.org (Torch wheels via `uv --torch-backend`). uv itself is installed from
#   PyPI, so it needs no separate whitelist entry; GitHub is never contacted.
#
# CUDA runtime libraries come from the PyTorch wheels, so no CUDA base image is needed;
# the host only needs an NVIDIA driver and the NVIDIA container toolkit.
# Refolding oracle runtimes (AF3/OpenFold3 and AF2), built natively for the image architecture.
# Their sources are NOT in this repository: pass them as named build contexts, e.g.
#   docker build --build-context af3src=/path/to/alphafold3-open --build-context af2src=/path/to/alphafold2 ...
# where af3src holds the alphafold3-open tree (pyproject.toml, CMakeLists.txt, src/) and af2src holds the
# Pallas-enabled `alphafold/` package. Without them the stages are empty and the image keeps the default
# behaviour (oracles then need AF3_IMAGE / AF2_IMAGE runtimes as documented).
FROM scratch AS af3src
FROM scratch AS af2src

FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS oracle-builder
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates build-essential git zlib1g-dev zstd && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --no-cache-dir "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" && \
    python -m pip install --no-cache-dir "uv>=0.13.4"
RUN mkdir -p /alphafold3_venv /alphafold2_venv /opt/alphafold2
COPY --from=af3src . /src/af3
COPY --from=af2src . /src/af2
# AF3 / OpenFold3: jax[cuda13] on both x86_64 and aarch64 (the alphafold3-open lockfile covers both).
RUN if [ -f /src/af3/pyproject.toml ]; then \
        cd /src/af3 && \
        UV_PROJECT_ENVIRONMENT=/alphafold3_venv uv venv /alphafold3_venv --python /usr/local/bin/python3.12 --allow-existing && \
        UV_PROJECT_ENVIRONMENT=/alphafold3_venv SETUPTOOLS_SCM_PRETEND_VERSION=3.1.2 uv sync --frozen --no-editable --no-dev && \
        uv pip install --python /alphafold3_venv/bin/python zstandard && \
        /alphafold3_venv/bin/build_data && \
        /alphafold3_venv/bin/python -c "import alphafold3.cpp, jax, haiku, tokamax"; \
    fi
# AF2 (Pallas): same JAX release; the overlay package shadows alphafold-colabfold through PYTHONPATH.
RUN if [ -d /src/af2/alphafold ]; then \
        uv venv /alphafold2_venv --python /usr/local/bin/python3.12 --allow-existing && \
        uv pip install --python /alphafold2_venv/bin/python "jax[cuda13]==0.10.2" dm-haiku==0.0.17 absl-py ml-collections \
            immutabledict biopython dm-tree optax==0.2.2 flax==0.8.4 chex jmp scipy pandas numpy packaging && \
        uv pip install --python /alphafold2_venv/bin/python --no-deps alphafold-colabfold==2.3.18 && \
        cp -r /src/af2/alphafold /opt/alphafold2/alphafold && \
        PYTHONPATH=/opt/alphafold2 /alphafold2_venv/bin/python -c "import jax, haiku, alphafold.model.model"; \
    fi

FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

ARG RFD4_ACCELERATOR=cu13
# Minimum uv: 0.13.4 is the first release with the emSign CA fix (python.org announcement, Oct 20 2026).
ARG UV_MIN_VERSION=0.13.4

ENV DATA_PATH=/data \
    HF_HOME=/opt/rfd4/cache/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONNOUSERSITE=1 \
    TMOL_DISABLE_WHEEL_FETCH=1 \
    TMOL_JIT_FALLBACK=0 \
    TMOL_USE_JIT=0

# build-essential: torch.compile (Inductor/Triton) compiles C++ and CUDA glue at run time.
# The lib* packages are runtime dependencies of OpenBabel.
# ca-certificates: makes sure the system CA bundle is the newest the distro offers.
#
# Bootstrap trust (eMudhra/emSign CA incident, Oct 2026): the pip shipped in the base image bundles a
# certifi that still trusts the compromised CA, and pip always trusts its bundled certifi in addition
# to the system store. The base image is pinned by digest, so its pip does not change on rebuild.
# Install the fixed pip from a hash-pinned URL so this download is verified by checksum rather than
# by the old trust store. Remove this step once the base image ships pip >= 26.2.2.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates build-essential libxrender1 libx11-6 libxext6 && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --no-cache-dir "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70" && \
    python -m pip install --no-cache-dir "uv>=${UV_MIN_VERSION}"

WORKDIR /opt/rfd4

# Dependencies first, from pyproject.toml alone, so source edits do not invalidate this layer.
# TMol and its operators must come from published wheels (never a source build).
# The certifi constraint is a floor so no resolution can install a copy that trusts the compromised CA;
# it only applies if certifi is pulled in and does not add the package.
COPY pyproject.toml ./
RUN printf 'certifi>=2026.10.19\n' > /tmp/rfd4-constraints.txt && \
    case "${RFD4_ACCELERATOR}" in \
        cu13) set -- --extra all-cuda13 --torch-backend cu130 ;; \
        cu12) set -- --extra all-cuda12 --torch-backend cu126 ;; \
        cpu) set -- --torch-backend cpu ;; \
        *) echo "RFD4_ACCELERATOR must be one of: cu13, cu12, cpu" >&2; exit 2 ;; \
    esac && \
    uv pip install --system --no-cache --no-sources -c /tmp/rfd4-constraints.txt \
        --only-binary tmol,tmol-ops-cu12,tmol-ops-cu13 "$@" -r pyproject.toml && \
    rm -f /tmp/rfd4-constraints.txt

# The application itself.
COPY . .
# pip check: on aarch64, nvidia-cusparselt-cu13 ships a wheel tagged manylinux2014_sbsa, which pip reports as
# "not supported on this platform" although it installs and works. Only that message is ignored.
RUN uv pip install --system --no-cache --no-sources --no-deps . && \
    { python -m pip check 2>&1 | grep -v 'is not supported on this platform' | { ! grep .; }; } && \
    uv --version && \
    python -m pip --version && \
    (python -m pip list 2>/dev/null | grep -i '^certifi ' || true)

# BioIR (inference_backend=bioir) comes with the all-cuda13 extra; cu12 and cpu stay on the native runtime.
RUN if [ "${RFD4_ACCELERATOR}" = "cu13" ]; then \
        python -c "from bionemo_ir._torch.layers.transformers.blocks import ConditionedTransformerBlock, PairUpdateBlock, with_cuda_graph"; \
    fi

# The cpu profile has no CUDA extra; install the physics metrics it needs directly.
RUN if [ "${RFD4_ACCELERATOR}" = "cpu" ]; then \
        printf 'certifi>=2026.10.19\n' > /tmp/rfd4-constraints.txt && \
        uv pip install --system --no-cache -c /tmp/rfd4-constraints.txt --only-binary tmol "tmol>=0.2.0,<0.3" posebusters && \
        rm -f /tmp/rfd4-constraints.txt; \
    fi

# JupyterLab, from PyPI only (same source as uv and the dependencies above).
RUN printf 'certifi>=2026.10.19\n' > /tmp/rfd4-constraints.txt && \
    uv pip install --system --no-cache -c /tmp/rfd4-constraints.txt jupyterlab && \
    rm -f /tmp/rfd4-constraints.txt && \
    { python -m pip check 2>&1 | grep -v 'is not supported on this platform' | { ! grep .; }; }

# Model weights: the default hf_ckpt_path bundle (nvidia/RFD4-Proteina) is baked into HF_HOME so the image
# runs offline. Build with --build-arg RFD4_PRELOAD_WEIGHTS=0 to skip it; weights then download on first use
# (mount a volume at /opt/rfd4/cache/huggingface to keep them across containers). Pass a token for private
# repos with: --secret id=hf_token,env=HF_TOKEN
ARG RFD4_PRELOAD_WEIGHTS=1
ARG RFD4_HF_REPO=nvidia/RFD4-Proteina
RUN --mount=type=secret,id=hf_token,required=false \
    if [ "${RFD4_PRELOAD_WEIGHTS}" = "1" ]; then \
        if [ -f /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi && \
        hf download "${RFD4_HF_REPO}" config.json model.safetensors; \
    fi

# Optional refolding-oracle weights for fully offline use: OpenFold3 (nvidia/RFD4-Proteina, pinned revision) and
# the AlphaFold 2 parameters (CC BY 4.0, all models, about 5.3 GB). Weights only: the prebuilt oracle runtimes
# (.sif) are amd64 + Apptainer images and are not embedded. Build with --build-arg RFD4_PRELOAD_ORACLE_WEIGHTS=0
# to skip. Keep the revision in sync with scripts/oracle_artifacts.sh.
ARG RFD4_PRELOAD_ORACLE_WEIGHTS=1
ARG RFD4_ORACLE_REVISION=598acab93d5d0aeacc4362f6e88dc20e7db85ef6
ARG AF2_PARAMS_URL=https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
ENV AF3_MODEL_DIR=/opt/rfd4/oracles/openfold3/weights \
    AF3_JAX_CACHE_DIR=/opt/rfd4/oracles/openfold3/jax_cache \
    AF2_MODEL_DIR=/opt/rfd4/oracles/af2/model_dir \
    AF2_JAX_CACHE_DIR=/opt/rfd4/oracles/af2/jax_cache
RUN --mount=type=secret,id=hf_token,required=false \
    mkdir -p "${AF3_MODEL_DIR}" "${AF3_JAX_CACHE_DIR}" "${AF2_MODEL_DIR}/params" "${AF2_JAX_CACHE_DIR}" && \
    if [ "${RFD4_PRELOAD_ORACLE_WEIGHTS}" = "1" ]; then \
        if [ -f /run/secrets/hf_token ]; then export HF_TOKEN="$(cat /run/secrets/hf_token)"; fi && \
        hf download "${RFD4_HF_REPO}" openfold3/weights/of3_ported_weights.bin.zst openfold3/weights/of3_ported_weights.provenance.json \
            --revision "${RFD4_ORACLE_REVISION}" --local-dir /opt/rfd4/oracles && \
        AF2_PARAMS_URL="${AF2_PARAMS_URL}" python -c 'import os, tarfile, urllib.request; r = urllib.request.urlopen(os.environ["AF2_PARAMS_URL"]); t = tarfile.open(fileobj=r, mode="r|"); t.extractall(os.environ["AF2_MODEL_DIR"] + "/params", filter="data")' && \
        test -f "${AF2_MODEL_DIR}/params/params_model_1_ptm.npz" && \
        test -f "${AF2_MODEL_DIR}/params/params_model_1_multimer_v3.npz" && \
        test -f "${AF3_MODEL_DIR}/of3_ported_weights.bin.zst"; \
    fi

# Native oracle runtimes from the oracle-builder stage (empty unless the af3src/af2src contexts were supplied).
# af3_apptainer / af2_apptainer detect them and run the workers directly instead of through Apptainer.
COPY --from=oracle-builder /alphafold3_venv /alphafold3_venv
COPY --from=oracle-builder /alphafold2_venv /alphafold2_venv
COPY --from=oracle-builder /opt/alphafold2 /opt/alphafold2

# Interactive shell for the JupyterLab terminal: bash (Jupyter uses $SHELL) with programmable tab completion.
RUN apt-get update && \
    apt-get install -y --no-install-recommends bash-completion less nano git && \
    rm -rf /var/lib/apt/lists/* && \
    printf '\n# RFD4: tab completion\nif ! shopt -oq posix && [ -f /usr/share/bash-completion/bash_completion ]; then\n    . /usr/share/bash-completion/bash_completion\nfi\nalias ll="ls -alF --color=auto"\nalias ls="ls --color=auto"\n' >> /etc/bash.bashrc && \
    printf '[ -f /etc/bash.bashrc ] && . /etc/bash.bashrc\n' >> /root/.bash_profile && \
    printf '[ -f /etc/bash.bashrc ] && . /etc/bash.bashrc\n' >> /root/.bashrc
ENV SHELL=/bin/bash

RUN mkdir -p /data /opt/rfd4/cache/huggingface /workspace
WORKDIR /workspace

# JupyterLab listens on 8080. Token authentication stays on (the token is printed in the container logs).
EXPOSE 8080

# Smoke test: imports, TMol operators for the selected profile, Inductor, and the CLI.
RUN TORCHINDUCTOR_CACHE_DIR=/tmp/rfd4-build-compile RFD4_ACCELERATOR="${RFD4_ACCELERATOR}" python -c \
        'import os; from importlib.metadata import distributions, version; import torch, atomworks, rfproteina; from openbabel import pybel; from tmol.score.ljlk.potentials._compiled import ljlk_pose_scores; accelerator = os.environ["RFD4_ACCELERATOR"]; assert torch.__version__.endswith("+" + {"cu13": "cu130", "cu12": "cu126", "cpu": "cpu"}[accelerator]); assert pybel.readstring("smi", "CCO").OBMol.NumAtoms() == 3; assert callable(ljlk_pose_scores); ops = {d.metadata["Name"].lower().replace("_", "-") for d in distributions() if d.metadata["Name"].lower().replace("_", "-").startswith("tmol-ops-")}; assert ops == (set() if accelerator == "cpu" else {"tmol-ops-" + accelerator}), ops; x = torch.ones(8); torch.testing.assert_close(torch.compile(lambda t: t.sin() + 1, backend="inductor", fullgraph=True)(x), x.sin() + 1)' && \
    rm -rf /tmp/rfd4-build-compile && \
    rfd4-generate --help

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8080", "--no-browser", "--allow-root", "--notebook-dir=/workspace"]
