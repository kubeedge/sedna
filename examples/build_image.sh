cd $(cd $(dirname ${BASH_SOURCE[0]}); pwd)

IMAGE_REPO=${IMAGE_REPO:-kubeedge}
IMAGE_TAG=${IMAGE_TAG:-v0.1.0}

EXAMPLE_REPO_PREFIX=${IMAGE_REPO}/sedna-example-

dockerfiles=(
federated-learning-surface-defect-detection-aggregation.Dockerfile
federated-learning-surface-defect-detection-train.Dockerfile
incremental-learning-helmet-detection.Dockerfile
joint-inference-helmet-detection-big.Dockerfile
joint-inference-helmet-detection-little.Dockerfile
)

for dockerfile in ${dockerfiles[@]}; do
  example_name=${dockerfile/.Dockerfile}
  docker build -f $dockerfile -t ${EXAMPLE_REPO_PREFIX}${example_name}:${IMAGE_TAG} --label sedna=examples ..
done
