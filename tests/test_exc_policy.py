import ast
from pathlib import Path

BANNED_PATHS = [
    "earnings_edge/trade_approval.py",
    "earnings_edge/alpaca_bridge.py",
    "earnings_edge/fwd_factor_ladder.py",
    "earnings_edge/alpaca_trading.py",
    "framework/execution/",
    "framework/risk/",
    "framework/positions/exits.py",
    "framework/positions/guards.py",
    "framework/positions/book_actions.py",
]


def is_banned(path: Path) -> bool:
    rel_path = path.as_posix()
    for b in BANNED_PATHS:
        if (b.endswith("/") and rel_path.startswith(b)) or rel_path == b:
            return True
    return False


def contains_record_event(node):
    # Walk the exception handler body looking for a Call to record_event or a bare raise
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id == "record_event":
                return True
        if isinstance(child, ast.Raise):
            return True
    return False


def check_file(path: Path) -> list:
    errors = []
    with open(path, encoding="utf-8") as f:
        content = f.read()
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            # bare except: type is None
            # except Exception: type is Name(id='Exception')
            # except Exception as e: type is Name(id='Exception')
            is_broad = False
            if node.type is None or (isinstance(node.type, ast.Name) and node.type.id == "Exception"):
                is_broad = True

            if is_broad:
                if not contains_record_event(node):
                    errors.append(f"{path}:{node.lineno}")
    return errors


def test_exception_policy():
    errors = []
    root = Path(".")
    for path in root.rglob("*.py"):
        if is_banned(path):
            errors.extend(check_file(path))

    if errors:
        msg = "Found banned broad exceptions without record_event:\n" + "\n".join(errors)
        raise AssertionError(msg)


if __name__ == "__main__":
    test_exception_policy()
    print("Lint passed.")
