"""Conservative release checks; not a guarantee that all sensitive data is detected."""
from pathlib import Path
import csv
import json
import re

ROOT = Path(__file__).resolve().parent
PATTERNS = [
    r"ghp_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}",
    r"sk-[A-Za-z0-9]{24,}", r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+\.",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"postgres(?:ql)?://[^\s]+", r"(?<![A-Za-z])[A-Z]:[\\/]",
    r"C:[\\/]Users[\\/]", r"/home/[^\s]+", r"/Users/[^\s]+",
]
FORBIDDEN_COLUMNS = {'variant_id', 'chrom', 'chromosome', 'position', 'pos',
                     'ref', 'alt', 'label', 'component_id', 'patient_id',
                     'gene_blocked_probability', 'random_probability'}


def main():
    issues = []
    files = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and not any(x in {'.venv', '__pycache__', 'build', 'tmp'}
                               for x in p.relative_to(ROOT).parts))
    for p in files:
        rel = p.relative_to(ROOT)
        if p.name == Path(__file__).name:
            continue  # scanner contains its own detection patterns
        if p.name.startswith('.env') or p.suffix in {'.joblib', '.pkl', '.npz', '.vcf', '.gz', '.key'}:
            issues.append(f'{rel}: forbidden artifact type')
        if p.suffix == '.pdf':
            if rel.parts[0] != 'figures':
                issues.append(f'{rel}: only vector figure PDFs allowed')
            continue  # visually inspect PDF figure content separately
        text = p.read_text(encoding='utf-8-sig')
        if any(re.search(pattern, text) for pattern in PATTERNS):
            issues.append(f'{rel}: possible credential or machine-local path')
        if p.suffix == '.csv':
            with p.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                if FORBIDDEN_COLUMNS.intersection(reader.fieldnames or []):
                    issues.append(f'{rel}: row-level column')
                if sum(1 for _ in reader) > 500:
                    issues.append(f'{rel}: unexpected row count')
        if p.suffix == '.json':
            json.loads(text)
    if issues:
        raise SystemExit('\n'.join(issues))
    print(f'PASS: {len(files)} release files checked for forbidden artifacts, '
          'common secrets/local paths, CSV row-level columns and valid JSON.')
    print('This heuristic scan does not replace manual review or upstream license checks.')


if __name__ == '__main__':
    main()
