# bh-rerd: ProDifEvo-Refinement (AF2.3M backend) for SageMaker processing jobs.
# Mirrors the bonobo Dockerfile pattern: nvidia/cuda runtime base + miniconda +
# RERD conda env + pip deps + baked AF2 / NBB2 weights.
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# System packages. python3.11 via deadsnakes for parity with the conda env.
RUN apt-get update && apt-get install -y software-properties-common && \
    DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        python3.11 python3-pip vim make wget git build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Miniconda copied in from the official image.
COPY --from=continuumio/miniconda3:24.5.0-0 /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /home

# Build the conda env first (rarely changes) so it's cached across code edits.
COPY environment.yml /home/environment.yml
RUN conda env create -f environment.yml

# Install pip deps inside the RERD env. Order matters: torch (cu128) before
# requirements.txt so deepspeed/fair-esm/torch_geometric resolve against a
# known torch version. Then jax with cuda extras.
COPY requirements.txt /home/requirements.txt
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate RERD && \
    pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "jax[cuda12]==0.5.2"

# Bake AF2 (~3.5GB) + NanoBodyBuilder2 weights into the image so containers
# launch instantly on SageMaker. ESM2 / ESMFold are skipped — unused at runtime.
COPY mber-open/download_weights.sh /home/mber-open/download_weights.sh
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate RERD && \
    yes "y" | bash /home/mber-open/download_weights.sh /root/.mber --skip-esm

# Copy the rest of the repo. Heavy stuff above is already cached.
COPY . /home/

ENV PYTHONPATH=/home
ENTRYPOINT ["/home/scripts/entrypoint.sh"]
CMD ["python", "ab_refinement.py", "--help"]
