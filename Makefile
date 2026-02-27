.PHONY: format lint test train-ddp train-gpt-ddp mlflow \
       k8s-cluster k8s-cluster-delete \
       k8s-build k8s-train k8s-logs k8s-clean k8s-full \
       k8s-mlflow \
       k8s-dashboard k8s-dashboard-delete k8s-dashboard-token k8s-dashboard-proxy

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
	mkdir -p runs mlflow-k8s
	kind create cluster --name llmtrain --config k8s/kind-config.yaml

k8s-cluster-delete:
	kind delete cluster --name llmtrain

k8s-build:
	docker build -t llmtrain:dev -f k8s/Dockerfile .
	kind load docker-image llmtrain:dev --name llmtrain

k8s-train:
	kubectl apply -f k8s/rbac.yaml -f k8s/storage.yaml -f k8s/configmap.yaml -f k8s/service.yaml -f k8s/job.yaml
	kubectl wait --for=condition=complete --timeout=300s job/llmtrain

k8s-logs:
	kubectl logs -l app=llmtrain --all-containers --prefix

k8s-clean:
	kubectl delete -f k8s/job.yaml -f k8s/service.yaml -f k8s/configmap.yaml -f k8s/storage.yaml -f k8s/rbac.yaml --ignore-not-found

k8s-full: k8s-cluster k8s-build k8s-train k8s-logs

k8s-mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow-k8s/mlflow.db

# ---------------------------------------------------------------------------
# Kubernetes Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_URL := https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml

k8s-dashboard:
	kubectl apply -f $(DASHBOARD_URL)
	kubectl apply -f k8s/dashboard-admin.yaml
	@echo ""
	@echo "Dashboard deployed. Next steps:"
	@echo "  make k8s-dashboard-token   # copy the bearer token"
	@echo "  make k8s-dashboard-proxy   # start kubectl proxy on :8001"
	@echo "  Open: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/"
	@$(MAKE) k8s-dashboard-token
	@echo "Starting kubectl proxy on http://localhost:8001 ..."
	@echo "Dashboard URL: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/"
	$(MAKE) k8s-dashboard-proxy

k8s-dashboard-token:
	@kubectl -n kubernetes-dashboard create token dashboard-admin

k8s-dashboard-proxy:
	kubectl proxy

k8s-dashboard-delete:
	kubectl delete -f k8s/dashboard-admin.yaml --ignore-not-found
	kubectl delete -f $(DASHBOARD_URL) --ignore-not-found
