.PHONY: format lint test train-ddp train-gpt-ddp mlflow \
       k8s-cluster k8s-cluster-delete \
       k8s-build k8s-train k8s-logs k8s-clean k8s-full

format:
	uv run ruff format src tests

lint:
	uv run ruff check src tests
	uv run mypy --config-file=pyproject.toml src/llmtrain tests

test:
	uv run pytest

train-ddp:
	uv run torchrun --nproc_per_node=2 -m llmtrain train --config configs/presets/ddp_smoke.yaml

train-gpt-ddp:
	uv run torchrun --nproc_per_node=2 -m llmtrain train --config configs/presets/gpt_wikitext_ddp.yaml

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///./mlflow.db

# ---------------------------------------------------------------------------
# Kubernetes (kind) targets
# ---------------------------------------------------------------------------

k8s-cluster:
	kind create cluster --name llmtrain --config k8s/kind-config.yaml

k8s-cluster-delete:
	kind delete cluster --name llmtrain

k8s-build:
	docker build -t llmtrain:dev -f k8s/Dockerfile .
	kind load docker-image llmtrain:dev --name llmtrain

k8s-train:
	kubectl apply -f k8s/rbac.yaml -f k8s/configmap.yaml -f k8s/service.yaml -f k8s/job.yaml
	kubectl wait --for=condition=complete --timeout=300s job/llmtrain

k8s-logs:
	kubectl logs -l app=llmtrain --all-containers --prefix

k8s-clean:
	kubectl delete -f k8s/job.yaml -f k8s/service.yaml -f k8s/configmap.yaml -f k8s/rbac.yaml --ignore-not-found

k8s-full: k8s-cluster k8s-build k8s-train k8s-logs
