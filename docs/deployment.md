# Deployment

No cloud credentials were available in the authoring environment, so this project ships a locally verifiable stack plus Kubernetes manifests. Do not commit kubeconfigs, tokens, or cloud keys.

## Docker

```bash
docker build -t pulseguard-api:1.0.0 .
docker run --rm -p 8000:8000 pulseguard-api:1.0.0
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data @examples/patient_elevated.json
```

The image is Python 3.9, matching the interpreter that pickles the pipeline, and it installs only `requirements-serve.txt`.

## Compose monitoring stack

```bash
docker compose up --build
```

- API: http://127.0.0.1:8000
- Prometheus: http://127.0.0.1:9090
- Grafana: http://127.0.0.1:3000 (`admin` / `pulseguard-local`)

The Grafana password is a local demo default. Change it before any shared deployment.

## Kubernetes

Build the image where the cluster can see it. For Docker Desktop or Minikube:

```bash
docker build -t pulseguard-api:1.0.0 .
kubectl apply -k k8s/
kubectl get pods,svc -n pulseguard
kubectl port-forward -n pulseguard svc/pulseguard-api 8000:80
```

`k8s/service.yaml` uses `type: LoadBalancer`. On Docker Desktop that exposes a local address. On Minikube, use `minikube service -n pulseguard pulseguard-api`.

`k8s/servicemonitor.yaml` is not part of the kustomization because it needs the Prometheus Operator CRD. Apply it only on a cluster that already has that operator. The compose file is the path that works without extra CRDs.

Helm equivalent:

```bash
helm upgrade --install pulseguard deploy/helm/pulseguard --namespace pulseguard --create-namespace
```

## Public cloud

The same image can be pushed to ECR, GAR, or ACR and the manifests pointed at that image. That push requires an account and is left as an operator step. There is no public URL until that step is done.
