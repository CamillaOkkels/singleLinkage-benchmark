FROM singleLinkage-benchmark

RUN apt update -y && apt install -y libtbb-dev
RUN git clone https://github.com/CamillaOkkels/HSSL && cd HSSL/HNSWhsslRust && maturin develop -r
RUN python3 -c 'import hnswhsslrust as hrr'