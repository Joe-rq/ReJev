"""重建校验的正控与数据安全/失效反例；夹具为自写无关短句。"""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import verify_rebuild as v


def row(i):
    return {"id": str(i), "source": "fixture", "group_id": str(i),
            "state": f"fixture number {i}"}


class RebuildCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.roots = {k: root / k for k in ("upstream", "rejev", "rejev2")}
        for d in self.roots.values():
            d.mkdir()
        train, dev = [row(i) for i in range(6)], [row(i) for i in range(7, 9)]
        self.rows = dict(zip(v.FILES, [train, dev, train[:3], train[3:], dev, train[:2], train[2:3]]))
        self.manifest = {"schema_version": 1, "upstream_commit": "a" * 40,
            "split_parameters": {"holdout1": {"seed": 20260924, "group_fraction": 0.05},
                                 "holdout2": {"seed": 20260925, "group_fraction": 0.05}}, "files": {}}
        for name in v.FILES:
            self.write(name)

    def path(self, name):
        folder, filename = name.split("/")
        return self.roots[folder] / filename

    def write(self, name):
        rows = self.rows[name]
        payload = "".join(json.dumps(r) + "\n" for r in rows).encode()
        self.path(name).write_bytes(payload)
        e = {"sha256": hashlib.sha256(payload).hexdigest(), "rows": len(rows),
             "groups": len({(r['source'], r['group_id']) for r in rows}), "sources": {"fixture": len(rows)}}
        if name == "rejev/rejev-holdout.jsonl":
            e["statehashes_sha256"] = hashlib.sha256("\n".join(sorted({v.statehash(r['state']) for r in rows})).encode()).hexdigest()
        self.manifest["files"][name] = e

    def test_complete_positive(self):
        self.assertEqual(len(v.verify(self.manifest, self.roots)), 7)

    def test_same_count_changed_content(self):
        p = self.path(v.FILES[0])
        p.write_bytes(p.read_bytes().replace(b"number", b"secret"))
        with self.assertRaisesRegex(v.Invalid, "SHA256"):
            v.verify(self.manifest, self.roots)

    def test_missing_input_is_failure(self):
        self.path(v.FILES[-1]).unlink()
        with self.assertRaisesRegex(v.Invalid, "缺普通文件"):
            v.verify(self.manifest, self.roots)

    def test_symbolic_link_is_failure(self):
        p = self.path(v.FILES[-1]); p.unlink()
        p.symlink_to(self.path(v.FILES[0]))
        with self.assertRaises(v.Invalid):
            v.verify(self.manifest, self.roots)

    def test_added_content_field_in_manifest_rejected(self):
        self.manifest['files'][v.FILES[0]]['state'] = 'private content'
        with self.assertRaisesRegex(v.Invalid, "清单字段"):
            v.validate_manifest(self.manifest)

    def test_missing_file_anchor_rejected(self):
        del self.manifest['files'][v.FILES[-1]]
        with self.assertRaisesRegex(v.Invalid, "文件集合"):
            v.validate_manifest(self.manifest)

    def test_count_type_is_strict(self):
        self.manifest['files'][v.FILES[0]]['rows'] = True
        with self.assertRaisesRegex(v.Invalid, "计数异常"):
            v.validate_manifest(self.manifest)

    def test_source_counts_checked(self):
        e=self.manifest['files'][v.FILES[0]]
        e['sources']={'different':e['rows']}
        with self.assertRaisesRegex(v.Invalid, "分源计数不匹配"):
            v.verify(self.manifest, self.roots)

    def test_group_counts_checked(self):
        self.manifest['files'][v.FILES[0]]['groups'] -= 1
        with self.assertRaisesRegex(v.Invalid, "分组计数不匹配"):
            v.verify(self.manifest, self.roots)

    def test_duplicate_ids_even_with_matching_hash(self):
        self.rows[v.FILES[0]][1]['id'] = '0'; self.write(v.FILES[0])
        with self.assertRaisesRegex(v.Invalid, "重复 ID"):
            v.verify(self.manifest, self.roots)

    def test_overlap_even_with_matching_file_hashes(self):
        # 换成重复状态，并更新各文件锚点；相交判据仍须独立抓住跨侧污染。
        for name in (v.FILES[0], v.FILES[3]):
            for r in self.rows[name]:
                if r['id'] == '3': r['state'] = 'fixture number 0'
            self.write(name)
        with self.assertRaisesRegex(v.Invalid, "states 跨侧重叠"):
            v.verify(self.manifest, self.roots)

    def test_group_overlap_with_distinct_states(self):
        for name in (v.FILES[0], v.FILES[3]):
            for r in self.rows[name]:
                if r['id'] == '3':
                    r['group_id'] = '0'
            self.write(name)
        with self.assertRaisesRegex(v.Invalid, 'groups 跨侧重叠'):
            v.verify(self.manifest, self.roots)

    def test_id_overlap_with_distinct_groups_and_states(self):
        self.rows[v.FILES[3]][0]['id'] = '0'
        self.write(v.FILES[3])
        with self.assertRaisesRegex(v.Invalid, 'ids 跨侧重叠'):
            v.verify(self.manifest, self.roots)

    def test_membership_conservation_even_with_matching_hashes(self):
        self.rows[v.FILES[-1]]=[row(99)];self.write(v.FILES[-1])
        with self.assertRaisesRegex(v.Invalid, "集合不守恒"):
            v.verify(self.manifest, self.roots)

    def test_state_summary_checked(self):
        self.manifest['files'][v.FILES[3]]['statehashes_sha256']='f'*64
        with self.assertRaisesRegex(v.Invalid, "state 集摘要"):
            v.verify(self.manifest, self.roots)

    def test_oversized_line_rejected_before_json_parse(self):
        self.path(v.FILES[0]).write_bytes(b'x'*(v.MAX_LINE_BYTES+1))
        with self.assertRaisesRegex(v.Invalid, "行过大"):
            v.verify(self.manifest, self.roots)

    def test_actual_bytes_cap_despite_stale_stat(self):
        name=v.FILES[0]
        path=Mock(wraps=self.path(name))
        path.stat.return_value=SimpleNamespace(st_size=0)
        with patch.object(v,'MAX_BYTES',16):
            with self.assertRaisesRegex(v.Invalid,'实际读取过大'):
                v.read_records(path,name,self.manifest['files'][name])

    def test_deep_json_is_sanitized(self):
        name=v.FILES[0]
        self.path(name).write_text('['*100000+'0'+']'*100000)
        with self.assertRaisesRegex(v.Invalid,'JSON 无效'):
            v.verify(self.manifest,self.roots)

    def test_duplicate_manifest_key_cli_rejected_without_hidden_content(self):
        root = Path(self.tmp.name) / 'cli'
        code = root / 'src/data'
        code.mkdir(parents=True)
        for name in ('verify_rebuild.py', 'render.py'):
            shutil.copyfile(Path(v.__file__).parent / name, code / name)
        manifest = root / 'docs/reproducibility/history-hashes.json'
        manifest.parent.mkdir(parents=True)
        payload = json.dumps(self.manifest)
        secret = 'PRIVATE_FIXTURE_HIDDEN_FIELD'
        manifest.write_text('{"files":{"state":"'+secret+'"},'+payload[1:])
        cmd = [sys.executable, str(code / 'verify_rebuild.py'),
               '--upstream-records', str(self.roots['upstream']),
               '--rejev-records', str(self.roots['rejev']),
               '--rejev2-records', str(self.roots['rejev2'])]
        p = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(p.returncode, 1)
        self.assertIn('重复 JSON 键', p.stderr)
        self.assertNotIn(secret, p.stdout+p.stderr)
        self.assertNotIn('Traceback', p.stderr)
        self.assertNotIn(str(self.tmp.name), p.stderr)

    def test_manifest_size_cap_before_parse(self):
        p = Path(self.tmp.name) / 'large-manifest.json'
        p.write_bytes(b'x' * (v.MAX_MANIFEST_BYTES + 1))
        with self.assertRaisesRegex(v.Invalid, '清单过大'):
            v.load_manifest(p)

    def test_cli_error_never_prints_record(self):
        secret='SENSITIVE_FIXTURE_DO_NOT_PRINT'
        cmd=[sys.executable,str(Path(v.__file__)), '--upstream-records',str(self.roots['upstream']),
             '--rejev-records',str(self.roots['rejev']), '--rejev2-records',str(self.roots['rejev2'])]
        for payload in ('{'+secret, '['*100000+'0'+']'*100000):
            with self.subTest(deep=payload.startswith('[')):
                self.path(v.FILES[0]).write_text(payload)
                p=subprocess.run(cmd,capture_output=True,text=True)
                self.assertEqual(p.returncode,1)
                self.assertNotIn(secret,p.stdout+p.stderr)
                self.assertNotIn(str(self.tmp.name),p.stdout+p.stderr)
                self.assertNotIn('Traceback',p.stderr)
                self.assertIn('JSON 无效',p.stderr)


if __name__ == '__main__':
    unittest.main()
