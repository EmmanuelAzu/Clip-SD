"""Fixes the one syntax error in src/bozeat_experiment.py: a backslash
escape used directly inside an f-string {} expression, which needs Python
3.12+ (PEP 701). Rewritten to avoid the backslash-in-braces pattern
entirely, so it's valid on any Python version that supports f-strings at
all (3.6+), regardless of which interpreter happens to run the syntax
check.

Run this once from the repo root: python3 patch_bozeat_fix.py
"""

TARGET = "src/bozeat_experiment.py"

OLD = '            title_text = f"\'{ret_concept}\' {\'\\u2713\' if is_correct else \'\\u2717\'}"\n'
NEW = (
    '            check_mark = "\\u2713" if is_correct else "\\u2717"\n'
    '            title_text = f"\'{ret_concept}\' {check_mark}"\n'
)

with open(TARGET, "r", encoding="utf-8") as f:
    content = f.read()

if NEW in content:
    print(f"[*] {TARGET} already patched -- nothing to do.")
elif OLD in content:
    content = content.replace(OLD, NEW)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[+] Patched {TARGET} successfully.")
else:
    print(f"[!] Expected old line not found in {TARGET} -- check the file manually.")
    print("    Look for a line containing: title_text = f\"'{ret_concept}' {'\\u2713' if is_correct else '\\u2717'}\"")
    raise SystemExit(1)

import py_compile
py_compile.compile(TARGET, doraise=True)
print(f"[+] {TARGET} compiles successfully.")
