#!/usr/bin/env bash
# k8s/entrypoint.sh -- DDP environment bootstrap for IndexedJob pods.
#
# Reads JOB_COMPLETION_INDEX (set by the IndexedJob controller) and derives
# the standard DDP env vars that setup_ddp() expects:
#   RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
#
# WORLD_SIZE and MASTER_PORT are static env vars set in the Job spec.
# MASTER_ADDR is resolved dynamically:
#   - rank 0 uses its own POD_IP (injected via the downward API)
#   - non-zero ranks query the Kubernetes API for the rank-0 pod's IP
#
# Requires: curl, jq (installed in the Docker image)

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Derive RANK and LOCAL_RANK from JOB_COMPLETION_INDEX
# ---------------------------------------------------------------------------
if [ -z "${JOB_COMPLETION_INDEX:-}" ]; then
    echo "ERROR: JOB_COMPLETION_INDEX is not set. Is this running as an IndexedJob?" >&2
    exit 1
fi

export RANK="$JOB_COMPLETION_INDEX"
export LOCAL_RANK="$JOB_COMPLETION_INDEX"

echo "entrypoint: RANK=$RANK  LOCAL_RANK=$LOCAL_RANK  WORLD_SIZE=${WORLD_SIZE:-?}  MASTER_PORT=${MASTER_PORT:-?}"

# ---------------------------------------------------------------------------
# 2. Resolve MASTER_ADDR
# ---------------------------------------------------------------------------
if [ "$RANK" -eq 0 ]; then
    # Rank 0 advertises its own pod IP.
    if [ -z "${POD_IP:-}" ]; then
        echo "ERROR: POD_IP is not set (expected via downward API)." >&2
        exit 1
    fi
    export MASTER_ADDR="$POD_IP"
    echo "entrypoint: rank 0 — MASTER_ADDR=$MASTER_ADDR (own pod IP)"
else
    # Non-zero ranks discover the rank-0 pod IP via the Kubernetes API.
    SA_DIR="/var/run/secrets/kubernetes.io/serviceaccount"
    TOKEN="$(cat "$SA_DIR/token")"
    CA_CERT="$SA_DIR/ca.crt"
    NAMESPACE="$(cat "$SA_DIR/namespace")"
    API_SERVER="https://kubernetes.default.svc"

    # Label selector: completion-index 0 belonging to the same job.
    JOB_NAME="${JOB_NAME:?JOB_NAME env var must be set}"
    SELECTOR="batch.kubernetes.io/job-completion-index=0,job-name=${JOB_NAME}"
    URL="${API_SERVER}/api/v1/namespaces/${NAMESPACE}/pods?labelSelector=${SELECTOR}"

    MAX_RETRIES=30
    RETRY_INTERVAL=2
    MASTER_IP=""

    echo "entrypoint: rank $RANK — discovering rank-0 pod IP (job=$JOB_NAME) ..."

    for attempt in $(seq 1 "$MAX_RETRIES"); do
        RESPONSE=$(curl -s --cacert "$CA_CERT" \
            -H "Authorization: Bearer ${TOKEN}" \
            "$URL") || true

        # Extract podIP from the first (and only) matching pod.
        MASTER_IP=$(echo "$RESPONSE" | jq -r '.items[0].status.podIP // empty') || true

        if [ -n "$MASTER_IP" ] && [ "$MASTER_IP" != "null" ]; then
            break
        fi

        echo "entrypoint: attempt $attempt/$MAX_RETRIES — rank-0 pod IP not available yet, retrying in ${RETRY_INTERVAL}s ..."
        sleep "$RETRY_INTERVAL"
    done

    if [ -z "$MASTER_IP" ] || [ "$MASTER_IP" = "null" ]; then
        echo "ERROR: could not resolve rank-0 pod IP after $MAX_RETRIES attempts." >&2
        exit 1
    fi

    export MASTER_ADDR="$MASTER_IP"
    echo "entrypoint: rank $RANK — MASTER_ADDR=$MASTER_ADDR (discovered via K8s API)"
fi

# ---------------------------------------------------------------------------
# 3. Launch training
# ---------------------------------------------------------------------------
echo "entrypoint: exec python -m llmtrain train --config /config/train.yaml"
exec python -m llmtrain train --config /config/train.yaml
