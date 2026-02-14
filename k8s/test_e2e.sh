#!/usr/bin/env bash
# k8s/test_e2e.sh -- End-to-end validation of distributed training on kind.
#
# Creates a kind cluster, builds the Docker image, deploys the IndexedJob,
# waits for completion, and asserts correctness of the training output.
#
# Usage:
#   bash k8s/test_e2e.sh              # run E2E test (cleans up on success)
#   bash k8s/test_e2e.sh --no-cleanup # keep cluster and resources for debugging
#
# Prerequisites: docker, kind, kubectl

set -euo pipefail

CLUSTER_NAME="llmtrain"
IMAGE_NAME="llmtrain:dev"
JOB_TIMEOUT="300s"
CLEANUP=true

for arg in "$@"; do
    case "$arg" in
        --no-cleanup) CLEANUP=false ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "==> $*"; }
error() { echo "ERROR: $*" >&2; }

cleanup() {
    if [ "$CLEANUP" = true ]; then
        info "Cleaning up ..."
        kubectl delete -f k8s/job.yaml -f k8s/service.yaml \
            -f k8s/configmap.yaml -f k8s/rbac.yaml --ignore-not-found 2>/dev/null || true
        kind delete cluster --name "$CLUSTER_NAME" 2>/dev/null || true
        info "Cleanup complete."
    else
        info "Skipping cleanup (--no-cleanup). Cluster '$CLUSTER_NAME' is still running."
    fi
}

# ---------------------------------------------------------------------------
# 1. Create kind cluster (reuse if it already exists)
# ---------------------------------------------------------------------------
info "Step 1: Ensuring kind cluster '$CLUSTER_NAME' exists ..."
if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    info "Cluster '$CLUSTER_NAME' already exists — reusing."
else
    kind create cluster --name "$CLUSTER_NAME" --config k8s/kind-config.yaml
    info "Cluster '$CLUSTER_NAME' created."
fi

# From this point, trap cleanup on unexpected exit.
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 2. Build and load the Docker image
# ---------------------------------------------------------------------------
info "Step 2: Building Docker image '$IMAGE_NAME' ..."
docker build -t "$IMAGE_NAME" -f k8s/Dockerfile .

info "Loading image into kind cluster ..."
kind load docker-image "$IMAGE_NAME" --name "$CLUSTER_NAME"

# ---------------------------------------------------------------------------
# 3. Apply manifests and wait for Job completion
# ---------------------------------------------------------------------------
info "Step 3: Applying K8s manifests ..."

# Clean up any prior run to avoid conflicts.
kubectl delete -f k8s/job.yaml --ignore-not-found 2>/dev/null || true

kubectl apply -f k8s/rbac.yaml \
              -f k8s/configmap.yaml \
              -f k8s/service.yaml \
              -f k8s/job.yaml

info "Waiting for Job 'llmtrain' to complete (timeout: $JOB_TIMEOUT) ..."
kubectl wait --for=condition=complete --timeout="$JOB_TIMEOUT" job/llmtrain

# ---------------------------------------------------------------------------
# 4. Fetch logs from both pods
# ---------------------------------------------------------------------------
info "Step 4: Fetching logs from training pods ..."

# Identify the rank-0 pod (completion index 0).
RANK0_POD=$(kubectl get pods -l app=llmtrain,batch.kubernetes.io/job-completion-index=0 \
    -o jsonpath='{.items[0].metadata.name}')

echo ""
echo "--- All pod logs (prefixed) ---"
kubectl logs -l app=llmtrain --all-containers --prefix
echo "--- End of logs ---"
echo ""

RANK0_LOGS=$(kubectl logs "$RANK0_POD")

# ---------------------------------------------------------------------------
# 5. Assert training output appears in rank-0 logs
# ---------------------------------------------------------------------------
info "Step 5: Asserting training output in rank-0 logs ..."

ASSERT_PASS=true

# The training summary printed by the CLI includes "final_step=" when training
# completes successfully (text-mode summary from format_run_summary).
if echo "$RANK0_LOGS" | grep -q "final_step="; then
    info "  PASS: rank-0 logs contain 'final_step=' (training summary present)."
else
    error "  FAIL: rank-0 logs do NOT contain 'final_step=' — training may not have completed."
    ASSERT_PASS=false
fi

# The entrypoint logs the exec line before launching training.
if echo "$RANK0_LOGS" | grep -q "entrypoint: exec python"; then
    info "  PASS: rank-0 logs contain entrypoint launch message."
else
    error "  FAIL: rank-0 logs do NOT contain entrypoint launch message."
    ASSERT_PASS=false
fi

# ---------------------------------------------------------------------------
# 6. Assert both pods exited with code 0
# ---------------------------------------------------------------------------
info "Step 6: Asserting all pods exited with code 0 ..."

# Get exit codes for all containers in training pods.
EXIT_CODES=$(kubectl get pods -l app=llmtrain \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{range .status.containerStatuses[*]}{.state.terminated.exitCode}{end}{"\n"}{end}')

echo "$EXIT_CODES" | while IFS=$'\t' read -r pod_name exit_code; do
    if [ -z "$pod_name" ]; then
        continue
    fi
    if [ "$exit_code" = "0" ]; then
        info "  PASS: Pod '$pod_name' exited with code 0."
    else
        error "  FAIL: Pod '$pod_name' exited with code $exit_code (expected 0)."
        # Signal failure to the outer script via a temp file since this runs
        # in a subshell (piped while-read).
        touch /tmp/llmtrain_e2e_fail
    fi
done

if [ -f /tmp/llmtrain_e2e_fail ]; then
    rm -f /tmp/llmtrain_e2e_fail
    ASSERT_PASS=false
fi

# ---------------------------------------------------------------------------
# 7. Final result
# ---------------------------------------------------------------------------
echo ""
if [ "$ASSERT_PASS" = true ]; then
    info "All assertions passed. E2E test SUCCEEDED."
    # Cleanup runs via EXIT trap.
    exit 0
else
    error "One or more assertions failed. E2E test FAILED."
    error "Re-run with --no-cleanup to inspect the cluster."
    # Cleanup runs via EXIT trap.
    exit 1
fi
