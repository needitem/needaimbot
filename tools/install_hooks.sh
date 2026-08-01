#!/bin/bash
# git hook 은 클론에 딸려오지 않는다(.git/hooks 는 추적 대상이 아니다). 추적되는
# tools/hooks 를 hooksPath 로 지정해, 클론한 곳에서 한 번만 실행하면 되게 한다.
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
git config core.hooksPath tools/hooks
chmod +x tools/hooks/* 2>/dev/null || true
echo "  core.hooksPath -> tools/hooks 설정 완료"
echo "  설치된 훅: $(ls tools/hooks | tr '\n' ' ')"
