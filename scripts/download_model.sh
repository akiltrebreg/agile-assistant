#!/usr/bin/env bash
# Download avibe-gptq-8bit model from Yandex Cloud S3 to local directory.
#
# Usage:
#   ./scripts/download_model.sh [target_dir]
#
# Requires: aws cli configured with Yandex Cloud credentials.
# Environment variables (from .env):
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_ENDPOINT, S3_BUCKET, S3_MODEL_PATH

set -euo pipefail

TARGET_DIR="${1:-/models/avibe-gptq-8bit}"

# Source .env if present
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

S3_ENDPOINT="${S3_ENDPOINT:-https://storage.yandexcloud.net}"
S3_BUCKET="${S3_BUCKET:-quant-models-agile}"
S3_MODEL_PATH="${S3_MODEL_PATH:-models/avibe-gptq-8bit}"

echo "Downloading model from s3://${S3_BUCKET}/${S3_MODEL_PATH} → ${TARGET_DIR}"

mkdir -p "${TARGET_DIR}"

aws s3 sync \
    "s3://${S3_BUCKET}/${S3_MODEL_PATH}/" \
    "${TARGET_DIR}/" \
    --endpoint-url "${S3_ENDPOINT}" \
    --no-sign-request=false

echo "Model downloaded to ${TARGET_DIR}"
echo "Files:"
ls -lh "${TARGET_DIR}/"
