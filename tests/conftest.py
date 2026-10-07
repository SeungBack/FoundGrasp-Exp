"""테스트 공통 — 프로젝트 루트를 import 경로에 추가 (하드웨어·GPU 불필요)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
