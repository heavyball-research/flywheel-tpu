# The vLLM TPU backend is upstream tpu-inference pinned as a submodule plus one
# patch. The submodule's recorded commit is the pin; the patch holds every
# backend change on top of it and imports the kernels from flywheel_tpu.

tpu_inference := "third_party/tpu-inference"
tpu_inference_patch := "patches/tpu-inference.patch"

# Check out the pinned tpu-inference and apply the backend patch to it.
tpu-inference-apply:
    #!/usr/bin/env bash
    set -euo pipefail
    git submodule update --init {{tpu_inference}}
    if [ -n "$(git -C {{tpu_inference}} status --porcelain)" ]; then
      echo "{{tpu_inference}} has local changes: export or discard them first" >&2
      exit 1
    fi
    git -C {{tpu_inference}} apply --index "$PWD/{{tpu_inference_patch}}"
    echo "applied {{tpu_inference_patch}} onto $(git -C {{tpu_inference}} rev-parse --short HEAD)"

# Write the submodule's changes against the pinned commit into the patch.
tpu-inference-export:
    #!/usr/bin/env bash
    set -euo pipefail
    pinned="$(git ls-files --stage {{tpu_inference}} | awk '{print $2}')"
    head="$(git -C {{tpu_inference}} rev-parse HEAD)"
    if [ "$head" != "$pinned" ]; then
      echo "{{tpu_inference}} is at $head, not the pinned $pinned; the patch" \
        "must stay uncommitted on top of the pin" >&2
      exit 1
    fi
    git -C {{tpu_inference}} add -A
    mkdir -p "$(dirname {{tpu_inference_patch}})"
    git -C {{tpu_inference}} diff --cached --binary "$pinned" > {{tpu_inference_patch}}
    git -C {{tpu_inference}} diff --cached --stat "$pinned" | tail -1

# Move the pin to an upstream commit and re-apply the patch there (3-way).
tpu-inference-bump rev:
    #!/usr/bin/env bash
    set -euo pipefail
    pinned="$(git ls-files --stage {{tpu_inference}} | awk '{print $2}')"
    git -C {{tpu_inference}} add -A
    if ! git -C {{tpu_inference}} diff --cached --binary "$pinned" | cmp -s - {{tpu_inference_patch}}; then
      echo "{{tpu_inference}} differs from {{tpu_inference_patch}}: run just tpu-inference-export first" >&2
      exit 1
    fi
    git -C {{tpu_inference}} reset -q --hard "$pinned"
    git -C {{tpu_inference}} fetch -q origin
    git -C {{tpu_inference}} checkout -q --detach "{{rev}}"
    git add {{tpu_inference}}
    if ! git -C {{tpu_inference}} apply --3way "$PWD/{{tpu_inference_patch}}"; then
      echo "conflicts: resolve them in {{tpu_inference}}, then run just tpu-inference-export" >&2
      exit 1
    fi
    just tpu-inference-export
