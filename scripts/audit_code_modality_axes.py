"""Audit code-candidate rows for modality-specific coverage axes.

The fitz-gov V8 schema keeps `routing.expert_fired` as the semantic/domain
route and uses `meta.modality` for evidence representation. This script treats
code language, artifact type, question target, and failure mode as audit axes,
not schema fields. It does not modify candidate rows.

Run from project root:
    python scripts/audit_code_modality_axes.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

DEFAULT_INPUT = Path(
    "C:/Users/yanfi/PycharmProjects/fitz-gov/data/_workspaces/handoff/"
    "modality_code_v1_20260527/cases.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("outputs/code_modality_axis_audit/modality_code_v1_20260527")

PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_.@+-]+[/\\])+[A-Za-z0-9_.@+-]+(?:\.[A-Za-z0-9_+-]+)+)"
    r"|(?P<bare>\b(?:README|CHANGELOG|package|tsconfig|pyproject|Dockerfile)"
    r"(?:\.[A-Za-z0-9_+-]+)+\b)"
)

EXT_LANGUAGE = {
    ".go": "go",
    ".java": "java",
    ".java_kotlin": "java_kotlin",
    ".js": "typescript_javascript",
    ".json": "json",
    ".jsx": "typescript_javascript",
    ".kt": "kotlin",
    ".md": "markdown",
    ".py": "python",
    ".rs": "rust",
    ".sh": "shell_ci",
    ".shell_ci": "shell_ci",
    ".sql": "sql",
    ".toml": "toml",
    ".ts": "typescript_javascript",
    ".tsx": "typescript_javascript",
    ".yaml": "yaml",
    ".yml": "yaml",
}

MECHANISM_FAILURE_MODE = {
    "exact_function_answer": "exact_symbol_support",
    "control_flow_support": "control_flow_support",
    "decorator_guard_support": "control_flow_support",
    "transaction_order_support": "control_flow_support",
    "test_proves_behavior": "test_execution_support",
    "config_sets_behavior": "config_direct_support",
    "retry_limit_code_config_agreement": "config_direct_support",
    "stack_trace_resolves_line": "trace_root_line_support",
    "docs_and_impl_agree": "docs_impl_agreement",
    "api_route_direct": "route_direct_support",
    "type_iface_resolves": "type_contract_support",
    "missing_relevant_file": "missing_relevant_artifact",
    "wrong_version_api": "wrong_api_version",
    "wrong_api_version": "wrong_api_version",
    "incomplete_snippet": "incomplete_snippet",
    "impl_no_test": "missing_execution_result",
    "test_definition_without_run": "missing_execution_result",
    "docs_no_code": "docs_without_code",
    "ambiguous_name_collision": "ambiguous_or_wrong_symbol",
    "wrong_symbol": "wrong_symbol",
    "retry_limit_wrong_service": "wrong_symbol",
    "missing_specific_field": "missing_specific_field",
    "config_wrong_env": "wrong_environment",
    "trace_no_root_line": "missing_root_frame",
    "dep_docs_too_general": "dependency_docs_too_general",
    "client_present_server_missing": "generated_client_without_server",
    "code_vs_docs": "docs_code_conflict",
    "docs_code_conflict": "docs_code_conflict",
    "test_vs_impl": "test_impl_conflict",
    "test_impl_conflict": "test_impl_conflict",
    "two_files_defaults": "default_value_conflict",
    "retry_limit_code_config_conflict": "default_value_conflict",
    "constant_config_conflict": "default_value_conflict",
    "changelog_vs_code": "changelog_code_conflict",
    "type_vs_runtime": "type_runtime_conflict",
    "stale_client_vs_server": "stale_generated_client_conflict",
    "config_vs_runtime_guard": "config_runtime_guard_conflict",
    "config_runtime_guard_conflict": "config_runtime_guard_conflict",
    "security_vs_middleware": "security_policy_middleware_conflict",
    "ci_vs_pkg_script": "ci_package_script_conflict",
}

OOD_TARGET_ALIASES = {
    "exact_symbol_support": {"exact_symbol_support"},
    "control_flow_support": {
        "control_flow_support",
        "decorator_guard_support",
        "decorator_support",
        "transaction_order_support",
    },
    "config_code_conflict": {"default_value_conflict", "config_runtime_guard_conflict"},
    "security_metadata_conflict": {"security_policy_middleware_conflict"},
    "docs_code_conflict": {"docs_code_conflict", "changelog_code_conflict"},
    "wrong_symbol": {"ambiguous_or_wrong_symbol", "wrong_symbol"},
    "wrong_api_version": {"wrong_api_version"},
    "missing_specific_field": {"missing_specific_field"},
    "missing_execution_result": {"missing_execution_result", "missing_root_frame"},
    "test_impl_conflict": {"test_impl_conflict"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=None,
        help="Candidate cases JSONL. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def contexts(case: dict[str, Any]) -> list[dict[str, Any]]:
    values = case.get("input", {}).get("contexts") or []
    if not isinstance(values, list):
        return []
    return [ctx for ctx in values if isinstance(ctx, dict)]


def context_texts(case: dict[str, Any]) -> list[str]:
    return [str(ctx.get("text") or "") for ctx in contexts(case)]


def extract_paths(text: str) -> list[str]:
    found = []
    for match in PATH_RE.finditer(text):
        path = match.group("path") or match.group("bare")
        if not path:
            continue
        found.append(path.strip("`'\"<>"))
    return sorted(set(found))


def suffix_language(path: str) -> str | None:
    lower = path.casefold()
    for suffix, language in sorted(EXT_LANGUAGE.items(), key=lambda item: -len(item[0])):
        if lower.endswith(suffix):
            return language
    return None


def looks_like(text: str) -> set[str]:
    lower = text.casefold()
    signals: set[str] = set()
    if re.search(r"(^|\n)\s*def\s+\w+\(", text):
        signals.add("python")
    if "traceback (most recent call last)" in lower or "stack trace" in lower:
        signals.add("runtime_trace")
    if "runtime trace" in lower or "error class:" in lower:
        signals.add("runtime_trace")
    if re.search(r"\b(export\s+function|export\s+interface|const\s+\w+)", text):
        signals.add("typescript_javascript")
    if re.search(r"\bfunction\s+\w+\(", text):
        signals.add("typescript_javascript")
    if re.search(r"(^|\n)\s*[A-Za-z0-9_-]+:\s+.+", text):
        signals.add("yaml")
    if re.search(r"\b(SELECT|FROM|WHERE|JOIN)\b", text):
        signals.add("sql")
    if lower.lstrip().startswith("{") or '"scripts"' in lower:
        signals.add("json")
    if "<!--" in text or "readme" in lower or "changelog" in lower:
        signals.add("markdown")
    if "cargo " in lower or "fn " in lower:
        signals.add("rust")
    return signals


def infer_languages(texts: list[str], paths: list[str]) -> list[str]:
    languages = {lang for path in paths if (lang := suffix_language(path))}
    for text in texts:
        languages.update(looks_like(text))
    return sorted(languages or {"unknown"})


def infer_artifacts(texts: list[str], paths: list[str], authority_signals: list[str]) -> list[str]:
    joined = "\n".join(texts).casefold()
    path_blob = "\n".join(paths).casefold()
    artifacts: set[str] = set()

    if "traceback" in joined or "stack trace" in joined or "runtime trace" in joined:
        artifacts.add("trace_or_log")
    if re.search(r"(^|[/\\])tests?[/\\]|test_", path_blob) or "def test_" in joined:
        artifacts.add("test_file")
    if (
        "readme" in path_blob
        or "docs/" in path_blob
        or "changelog" in path_blob
        or "<!--" in joined
    ):
        artifacts.add("documentation")
    if "generated_client" in path_blob or "generated client" in joined:
        artifacts.add("generated_client")
    if (
        "config/" in path_blob
        or "settings" in path_blob
        or "production.yaml" in path_blob
        or "staging.yaml" in path_blob
        or "env:" in joined
    ):
        artifacts.add("config")
    if ".github/workflows" in path_blob or "package.json" in path_blob or "npm run" in joined:
        artifacts.add("build_or_ci")
    if "openapi" in path_blob or "api reference" in joined:
        artifacts.add("api_spec")
    if "repository search results" in joined or "symbol search" in joined:
        artifacts.add("search_results")
    if "source_code" in authority_signals or any(suffix_language(path) for path in paths):
        artifacts.add("source_code")

    return sorted(artifacts or {"unknown"})


def infer_question_targets(query: str, texts: list[str]) -> list[str]:
    lower = f"{query}\n" + "\n".join(texts)
    lower = lower.casefold()
    targets: set[str] = set()

    if "/" in query or "route" in lower or "endpoint" in lower or "middleware" in lower:
        targets.add("route_or_endpoint")
    if "`" in query or re.search(r"\b(function|method|class|symbol)\b", lower):
        targets.add("function_or_symbol")
    if re.search(r"\b(config|setting|timeout|retry|default|env|environment|flag|enabled)\b", lower):
        targets.add("config_or_default")
    if re.search(r"\b(test|passed|pass|execution result|result grid|migration)\b", lower):
        targets.add("test_or_execution_result")
    if re.search(r"\b(version|api v[0-9]+|build|changelog)\b", lower):
        targets.add("version_or_build")
    if re.search(r"\b(auth|admin|token|security|role|public|middleware)\b", lower):
        targets.add("security_or_auth")
    if re.search(r"\b(type|interface|schema|field|audit event)\b", lower):
        targets.add("data_schema_or_field")
    if re.search(r"\b(stack trace|traceback|root frame|root line|error class)\b", lower):
        targets.add("trace_diagnostic")

    return sorted(targets or {"unknown"})


def infer_failure_mode(case: dict[str, Any]) -> str:
    mechanism = str(case.get("meta", {}).get("mechanism") or "")
    if mechanism in MECHANISM_FAILURE_MODE:
        return MECHANISM_FAILURE_MODE[mechanism]

    label = str(case.get("governance", {}).get("classification") or "unknown").upper()
    pattern = str(case.get("taxonomy", {}).get("pattern") or "")
    if label == "ABSTAIN" and pattern == "version_build_mismatch":
        return "wrong_version"
    if label == "ABSTAIN" and pattern == "missing_execution_result":
        return "missing_execution_result"
    if label == "ABSTAIN" and pattern == "wrong_entity":
        return "wrong_symbol_or_entity"
    if label == "DISPUTED" and "conflict" in pattern:
        return f"{pattern}_conflict"
    if label == "TRUSTWORTHY":
        return "support"
    return "unknown"


def syntax_mismatches(paths: list[str], texts: list[str]) -> list[str]:
    text = "\n".join(texts)
    syntaxes = looks_like(text)
    flags = []
    for path in paths:
        lang = suffix_language(path)
        if not lang:
            continue
        if lang in {"sql", "yaml", "go", "java_kotlin", "shell_ci"} and (
            "typescript_javascript" in syntaxes or "python" in syntaxes
        ):
            flags.append(f"{path}: extension={lang}, syntax={','.join(sorted(syntaxes))}")
        if lang == "python" and "typescript_javascript" in syntaxes:
            flags.append(f"{path}: extension=python, syntax={','.join(sorted(syntaxes))}")
        if lang == "typescript_javascript" and "python" in syntaxes:
            flags.append(
                f"{path}: extension=typescript_javascript, syntax={','.join(sorted(syntaxes))}"
            )
    return flags


def axis_count(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in records:
        value = row[key]
        if isinstance(value, list):
            counts.update(str(item) for item in value)
        else:
            counts[str(value)] += 1
    return dict(counts.most_common())


def axis_by_label(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        values = row[key] if isinstance(row[key], list) else [row[key]]
        for value in values:
            out[str(value)][row["label"]] += 1
    return {name: dict(counter.most_common()) for name, counter in sorted(out.items())}


def target_ood_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode = Counter(row["failure_mode"] for row in records)
    out = {}
    for target, aliases in OOD_TARGET_ALIASES.items():
        count = sum(by_mode[alias] for alias in aliases)
        out[target] = {
            "rows": count,
            "aliases": sorted(aliases),
            "status": "present" if count else "gap",
        }
    return out


def build_records(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for case in cases:
        texts = context_texts(case)
        paths = sorted({path for text in texts for path in extract_paths(text)})
        mismatch_flags = []
        for text in texts:
            mismatch_flags.extend(syntax_mismatches(extract_paths(text), [text]))
        authority_signals = sorted(
            {str(ctx.get("authority_signal") or "unknown") for ctx in contexts(case)}
        )
        query = str(case.get("input", {}).get("query") or "")
        label = str(case.get("governance", {}).get("classification") or "unknown").upper()
        records.append(
            {
                "id": str(case.get("id") or ""),
                "label": label,
                "pattern": str(case.get("taxonomy", {}).get("pattern") or "unknown"),
                "domain": str(case.get("routing", {}).get("expert_fired") or "unknown"),
                "difficulty": str(case.get("meta", {}).get("difficulty") or "unknown"),
                "mechanism": str(case.get("meta", {}).get("mechanism") or "unknown"),
                "modality": str(case.get("meta", {}).get("modality") or "unknown"),
                "languages": infer_languages(texts, paths),
                "artifact_types": infer_artifacts(texts, paths, authority_signals),
                "question_targets": infer_question_targets(query, texts),
                "failure_mode": infer_failure_mode(case),
                "authority_signals": authority_signals,
                "path_mentions": paths,
                "syntax_mismatches": sorted(set(mismatch_flags)),
            }
        )
    return records


def summarize(cases: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    axes = [
        "label",
        "pattern",
        "domain",
        "difficulty",
        "mechanism",
        "modality",
        "languages",
        "artifact_types",
        "question_targets",
        "failure_mode",
        "authority_signals",
    ]
    mismatch_rows = [row for row in records if row["syntax_mismatches"]]
    modality_counts = axis_count(records, "modality")
    code_rows = modality_counts.get("code", 0)
    return {
        "rows": len(records),
        "input_rows": len(cases),
        "modality_counts": modality_counts,
        "code_modality_rate": code_rows / len(records) if records else 0.0,
        "axes": {axis: axis_count(records, axis) for axis in axes},
        "by_label": {
            axis: axis_by_label(records, axis)
            for axis in (
                "mechanism",
                "languages",
                "artifact_types",
                "question_targets",
                "failure_mode",
            )
        },
        "ood_target_coverage": target_ood_counts(records),
        "syntax_mismatch": {
            "rows": len(mismatch_rows),
            "rate": len(mismatch_rows) / len(records) if records else 0.0,
            "examples": [
                {
                    "id": row["id"],
                    "label": row["label"],
                    "mechanism": row["mechanism"],
                    "flags": row["syntax_mismatches"][:3],
                }
                for row in mismatch_rows[:50]
            ],
        },
        "unknown_rates": {
            axis: {
                "rows": sum(1 for row in records if "unknown" in row[axis]),
                "rate": sum(1 for row in records if "unknown" in row[axis]) / len(records)
                if records
                else 0.0,
            }
            for axis in ("languages", "artifact_types", "question_targets")
        },
    }


def pct(value: float) -> str:
    return f"{value:.2%}"


def top_table(title: str, counts: dict[str, int], top_n: int) -> list[str]:
    lines = ["", f"## {title}", "", "| Value | Rows |", "|---|---:|"]
    for name, count in list(counts.items())[:top_n]:
        lines.append(f"| `{name}` | {count} |")
    return lines


def write_report(path: Path, summary: dict[str, Any], top_n: int) -> None:
    mismatch = summary["syntax_mismatch"]
    lines = [
        "# Code Modality Axis Audit",
        "",
        f"- Rows: **{summary['rows']}**",
        f"- `meta.modality == code`: **{summary['modality_counts'].get('code', 0)}** "
        f"({pct(summary['code_modality_rate'])})",
        f"- Extension/syntax mismatch flags: **{mismatch['rows']}** ({pct(mismatch['rate'])})",
        "",
        "## Interpretation",
        "",
        "These are audit axes, not schema fields. `routing.expert_fired` stays the semantic "
        "subject route; code language, artifact type, question target, and failure mode are "
        "modality-specific generation/evaluation axes.",
    ]

    for title, axis in (
        ("Labels", "label"),
        ("Semantic Domains", "domain"),
        ("Taxonomy Patterns", "pattern"),
        ("Mechanisms", "mechanism"),
        ("Languages", "languages"),
        ("Artifact Types", "artifact_types"),
        ("Question Targets", "question_targets"),
        ("Failure Modes", "failure_mode"),
    ):
        lines.extend(top_table(title, summary["axes"][axis], top_n))

    lines.extend(
        [
            "",
            "## OOD Target Coverage",
            "",
            "| Target | Rows | Status | Aliases |",
            "|---|---:|---|---|",
        ]
    )
    for target, row in summary["ood_target_coverage"].items():
        aliases = ", ".join(f"`{alias}`" for alias in row["aliases"])
        lines.append(f"| `{target}` | {row['rows']} | `{row['status']}` | {aliases} |")

    lines.extend(
        [
            "",
            "## Syntax Mismatch Examples",
            "",
            "| ID | Label | Mechanism | Flags |",
            "|---|---|---|---|",
        ]
    )
    examples = mismatch["examples"][:top_n]
    if examples:
        for row in examples:
            flags = "<br>".join(f"`{flag}`" for flag in row["flags"])
            lines.append(
                f"| `{row['id']}` | `{row['label']}` | `{row['mechanism']}` | {flags} |"
            )
    else:
        lines.append("| none |  |  |  |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    inputs = args.input or [DEFAULT_INPUT]
    cases = [case for path in inputs for case in load_jsonl(path)]
    records = build_records(cases)
    summary = summarize(cases, records)
    summary["input_files"] = [str(path) for path in inputs]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    write_report(args.output_dir / "report.md", summary, args.top_n)

    print(f"Rows              : {summary['rows']}")
    print(f"Code modality     : {summary['modality_counts'].get('code', 0)}")
    print(
        "Syntax mismatches : "
        f"{summary['syntax_mismatch']['rows']} "
        f"({summary['syntax_mismatch']['rate']:.2%})"
    )
    gaps = [
        name
        for name, row in summary["ood_target_coverage"].items()
        if row["status"] == "gap"
    ]
    print(f"OOD target gaps   : {', '.join(gaps) if gaps else 'none'}")
    print(f"Wrote             : {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
