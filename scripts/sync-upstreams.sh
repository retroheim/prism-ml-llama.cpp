#!/usr/bin/env bash
# Sync the prism-ml-llama.cpp fork with its three upstream sources:
#   ggml       — github.com/ggml-org/llama.cpp           (master)
#   prismml    — github.com/PrismML-Eng/llama.cpp        (prism)
#   turboquant — github.com/atomicmilkshake/llama-cpp-turboquant (feature/triattention)
#
# Usage:
#   scripts/sync-upstreams.sh           # fetch + show divergence (no merges)
#   scripts/sync-upstreams.sh status    # same as default
#   scripts/sync-upstreams.sh fetch     # just git fetch all three
#   scripts/sync-upstreams.sh merge     # interactive: prompt before each merge
#   scripts/sync-upstreams.sh merge --yes  # auto-merge all (dangerous if conflicts)
#
# Conflicts are NOT resolved automatically. After conflict, fix files,
# `git add`, `git commit`, then re-run `scripts/sync-upstreams.sh status`.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# upstream remote_name remote_url ref_to_track
UPSTREAMS=(
    "ggml|https://github.com/ggml-org/llama.cpp.git|master"
    "prismml|https://github.com/PrismML-Eng/llama.cpp.git|prism"
    "turboquant|https://github.com/atomicmilkshake/llama-cpp-turboquant.git|feature/triattention"
)

color_reset="$(printf '\033[0m')"
color_bold="$(printf '\033[1m')"
color_dim="$(printf '\033[2m')"
color_green="$(printf '\033[32m')"
color_yellow="$(printf '\033[33m')"
color_red="$(printf '\033[31m')"
color_cyan="$(printf '\033[36m')"

if [ ! -t 1 ]; then
    color_reset="" color_bold="" color_dim=""
    color_green="" color_yellow="" color_red="" color_cyan=""
fi

ensure_remote() {
    local name="$1" url="$2"
    if ! git remote get-url "$name" >/dev/null 2>&1; then
        echo "${color_yellow}+ adding remote $name → $url${color_reset}"
        git remote add "$name" "$url"
    fi
}

current_branch() {
    git rev-parse --abbrev-ref HEAD
}

cmd_fetch() {
    for entry in "${UPSTREAMS[@]}"; do
        IFS='|' read -r name url ref <<<"$entry"
        ensure_remote "$name" "$url"
        echo "${color_cyan}fetch $name $ref${color_reset}"
        git fetch --quiet "$name" "$ref"
    done
}

cmd_status() {
    cmd_fetch
    local branch
    branch="$(current_branch)"
    echo
    echo "${color_bold}Local branch:${color_reset} $branch ($(git rev-parse --short HEAD))"
    echo
    printf "%-12s %-30s %-12s %-12s %s\n" "REMOTE" "REF" "BEHIND" "AHEAD" "TIP"
    for entry in "${UPSTREAMS[@]}"; do
        IFS='|' read -r name url ref <<<"$entry"
        local remote_ref="$name/$ref"
        local behind ahead tip
        behind="$(git rev-list --count "${branch}..${remote_ref}" 2>/dev/null || echo "?")"
        ahead="$(git rev-list --count "${remote_ref}..${branch}" 2>/dev/null || echo "?")"
        tip="$(git rev-parse --short "$remote_ref" 2>/dev/null || echo "?")"
        local color="$color_green"
        if [ "$behind" != "0" ] && [ "$behind" != "?" ]; then color="$color_yellow"; fi
        printf "%-12s %-30s ${color}%-12s${color_reset} %-12s %s\n" \
            "$name" "$ref" "$behind" "$ahead" "$tip"
    done
    echo
    echo "${color_dim}'BEHIND' = commits in upstream not in local. 'AHEAD' = local commits not in upstream.${color_reset}"
}

merge_one() {
    local name="$1" ref="$2" auto="$3"
    local remote_ref="$name/$ref"
    local behind
    behind="$(git rev-list --count "HEAD..${remote_ref}")"
    if [ "$behind" = "0" ]; then
        echo "${color_green}✓ ${remote_ref} already merged${color_reset}"
        return 0
    fi
    echo "${color_bold}${color_yellow}→ ${remote_ref}: ${behind} commits to merge${color_reset}"
    if [ "$auto" != "--yes" ]; then
        read -r -p "  merge? [y/N/q] " ans
        case "$ans" in
            y|Y) ;;
            q|Q) echo "  quitting"; exit 0 ;;
            *)   echo "  skipping"; return 0 ;;
        esac
    fi
    if git merge --no-edit "$remote_ref"; then
        echo "${color_green}✓ merged $remote_ref${color_reset}"
    else
        echo "${color_red}✗ conflicts in $remote_ref — resolve, commit, then re-run${color_reset}"
        echo "  files:"
        git diff --name-only --diff-filter=U | sed 's/^/    /'
        exit 1
    fi
}

cmd_merge() {
    cmd_fetch
    if [ -n "$(git status --porcelain)" ]; then
        echo "${color_red}working tree not clean — commit or stash first${color_reset}"
        git status --short | head -10
        exit 1
    fi
    local auto="${1:-}"
    for entry in "${UPSTREAMS[@]}"; do
        IFS='|' read -r name url ref <<<"$entry"
        ensure_remote "$name" "$url"
        merge_one "$name" "$ref" "$auto"
    done
    echo
    echo "${color_bold}done. final state:${color_reset}"
    cmd_status
}

case "${1:-status}" in
    fetch)  cmd_fetch ;;
    status) cmd_status ;;
    merge)  cmd_merge "${2:-}" ;;
    -h|--help|help)
        sed -n '2,15p' "$0" | sed 's/^# \?//'
        ;;
    *)
        echo "unknown command: $1" >&2
        echo "usage: $0 [fetch|status|merge [--yes]]" >&2
        exit 1
        ;;
esac
