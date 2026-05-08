#!/bin/bash
# Pull rerd output.csv files for each target into ~/Downloads/rerd_<target>.csv.
# Reports gracefully when a run hasn't completed yet.
#
# Edit JOBS below if you want to point at a different set of SageMaker jobs.
set -uo pipefail

BUCKET="sagemaker-us-east-1-332120041740"
DEST_DIR="${HOME}/Downloads"
mkdir -p "$DEST_DIR"

declare -A JOBS=(
    [pdl1]=rerd-antibody-container-run-2026-05-07-19-21-48-238
    [bhrf1]=rerd-antibody-container-run-2026-05-07-19-22-03-696
    [il3]=rerd-antibody-container-run-2026-05-07-19-22-17-831
    [il20]=rerd-antibody-container-run-2026-05-07-19-46-25-763
)

for TARGET in pdl1 bhrf1 il3 il20; do
    JOB="${JOBS[$TARGET]}"

    # Find the S3 key for output.csv. The timestamped sub-prefix has spaces and
    # colons in it, so we resolve it dynamically rather than hard-coding.
    KEY=$(aws s3 ls "s3://${BUCKET}/${JOB}/" --recursive 2>/dev/null \
        | grep "output.csv$" \
        | awk '{ for(i=4;i<=NF;i++) printf "%s%s", $i, (i==NF?"":" ") }')

    if [ -z "$KEY" ]; then
        # Check job status to give a useful "not done" message.
        STATUS=$(aws sagemaker describe-processing-job \
            --processing-job-name "$JOB" --region us-east-1 \
            --query 'ProcessingJobStatus' --output text 2>/dev/null)
        REASON=$(aws sagemaker describe-processing-job \
            --processing-job-name "$JOB" --region us-east-1 \
            --query 'FailureReason' --output text 2>/dev/null)
        echo "[$TARGET] ⏳ no output.csv yet (job status: $STATUS${REASON:+, reason: $REASON})"
        continue
    fi

    DEST="${DEST_DIR}/rerd_${TARGET}.csv"
    if aws s3 cp "s3://${BUCKET}/${KEY}" "$DEST" --quiet 2>/dev/null; then
        ROWS=$(($(wc -l < "$DEST") - 1))
        echo "[$TARGET] ✅ $DEST ($ROWS binders)"
    else
        echo "[$TARGET] ❌ download failed for s3://${BUCKET}/${KEY}"
    fi
done
