# Kubernetes Operations Guide

This document contains the Kubernetes operational details that are intentionally
kept out of the top-level `README.md`.

## IndexedJob pattern

The training workload runs as a Kubernetes
[IndexedJob](https://kubernetes.io/docs/concepts/workloads/controllers/job/#completion-mode)
with `completions: 2` and `parallelism: 2`. Each pod receives a unique
`JOB_COMPLETION_INDEX` (0, 1, ...).

`k8s/entrypoint.sh` bootstraps DDP:

1. Sets `RANK` and `LOCAL_RANK` from `JOB_COMPLETION_INDEX`.
2. Uses static `WORLD_SIZE` and `MASTER_PORT` from the Job spec.
3. Resolves `MASTER_ADDR`:
   - Rank 0 uses its own pod IP (from downward API as `POD_IP`).
   - Other ranks query the Kubernetes API to locate rank-0 pod IP and retry
     with backoff until available.
4. Execs `python -m llmtrain train --config /config/train.yaml`.

The application code remains Kubernetes-agnostic and consumes the same env vars
that `torchrun` would provide.

## Manifest overview

| File | Purpose |
|------|---------|
| `k8s/Dockerfile` | Container image: `python:3.11-slim` with package + `curl`/`jq` |
| `k8s/kind-config.yaml` | Kind cluster spec + host `extraMounts` |
| `k8s/storage.yaml` | PV/PVC resources for host-backed run/MLflow persistence |
| `k8s/rbac.yaml` | ServiceAccount + Role + RoleBinding for master discovery |
| `k8s/configmap.yaml` | Embedded training config (`train.yaml`) |
| `k8s/service.yaml` | Headless Service resource (not used for `MASTER_ADDR` resolution) |
| `k8s/job.yaml` | IndexedJob manifest (`parallelism=2`, `completions=2`) |
| `k8s/entrypoint.sh` | DDP bootstrap and trainer launch |
| `k8s/dashboard-admin.yaml` | Admin account for local dashboard access |
| `k8s/test_e2e.sh` | End-to-end Kubernetes test script |

## Debugging commands

```bash
kubectl get pods -l app=llmtrain
kubectl describe pod <pod-name>
kubectl logs <pod-name>
kubectl logs -l app=llmtrain --prefix
kubectl get endpoints llmtrain-headless
```

## Common failure modes

- **MASTER_ADDR resolution failure**: RBAC misconfiguration or rank-0 pod not
  ready yet. Inspect `k8s/entrypoint.sh` log retries.
- **WORLD_SIZE mismatch**: Job `completions`/`parallelism` diverges from
  `WORLD_SIZE`.
- **RBAC 403 Forbidden**: ServiceAccount lacks pod get/list permissions.
- **API lookup failure**: rank>0 cannot find rank-0 pod via Kubernetes API
  (selector mismatch, auth issue, or rank-0 not yet assigned an IP).

## Kubernetes Dashboard (optional)

Launch:

```bash
make k8s-dashboard
```

This deploys dashboard resources, prints a Bearer token, and starts
`kubectl proxy` on `:8001`. To get the token or restart the proxy
without redeploying, use `make k8s-dashboard-token` or
`make k8s-dashboard-proxy`.

Dashboard URL:

```text
http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
```

Teardown:

```bash
make k8s-dashboard-delete
```
