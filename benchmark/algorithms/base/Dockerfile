FROM ubuntu:22.04

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    python3-numpy python3-scipy \
    build-essential git curl libtbb-dev \
    pkg-config libssl-dev libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python virtual environment
# -----------------------------
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:/root/.cargo/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip maturin

# -----------------------------
# Rust toolchain
# -----------------------------
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN rustup install 1.86.0
RUN rustup default 1.86.0
ENV RUSTUP_TOOLCHAIN=1.86.0

# -----------------------------
# Build HSSL Rust extension
# -----------------------------
WORKDIR /opt/build

RUN rm -rf HSSL GraphIndexAPI && \
    git clone https://github.com/CamillaOkkels/HSSL && \
    git clone --branch cabi --single-branch https://github.com/eth42/GraphIndexAPI HSSL/GraphIndexAPI && \
    rustup default 1.86.0  && \
    rustc --version && \
    /bin/bash -c "source /opt/venv/bin/activate && cd HSSL/HNSWhsslRust && maturin develop -r"
    

# -----------------------------
# Python application
# -----------------------------
WORKDIR /home/app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/opt/build/HSSL

ENTRYPOINT ["/opt/venv/bin/python", "-u"]
CMD ["run.py"]