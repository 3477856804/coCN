#!/bin/bash
# 双向镜像同步：GitHub ⇄ Gitee
# 策略：仅快进（fast-forward）。谁领先就把落后的那一边快进到领先方；
# 若两边出现分叉（互不为祖先），绝不覆盖任何一边历史，只报错留给人工处理。
set -e
cd "$(dirname "$0")/.."

git config user.name "coCN"
git config user.email "dev@cocn.local"

GITEE_URL="https://${GITEE_OWNER}:${GITEE_TOKEN}@gitee.com/${GITEE_OWNER}/${GITEE_REPO}.git"
git remote add gitee "$GITEE_URL" 2>/dev/null || git remote set-url gitee "$GITEE_URL"

git fetch origin master
echo "[A] GitHub -> Gitee"
git push gitee origin/master:master 2>&1 \
    || echo "[A] 提示：Gitee 领先或存在分叉，未覆盖（安全，保留 Gitee 侧提交）"

git fetch gitee master 2>/dev/null || true
if git merge-base --is-ancestor origin/master gitee/master 2>/dev/null; then
    if [ "$(git rev-parse origin/master)" != "$(git rev-parse gitee/master)" ]; then
        echo "[B] Gitee -> GitHub（把 GitHub 快进到 Gitee）"
        git checkout -q master
        git reset -q --hard gitee/master
        git push origin master 2>&1 \
            || echo "[B] 回波到 GitHub 失败"
    else
        echo "[B] 无变化"
    fi
else
    echo "[B] GitHub 与 Gitee 存在分叉，跳过自动同步（请人工处理）"
    exit 1
fi