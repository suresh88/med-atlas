#!/usr/bin/env python3
# drugkg.py
# Build + query a drug knowledge graph from your Excel outputs.

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

# -------------------------
# Config (edit if needed)
# -------------------------
NEO4J_URI_DEFAULT = "bolt://localhost:7687"
NEO4J_USER_DEFAULT = "neo4j"
NEO4J_PASS_DEFAULT = "password"
HTTP_TIMEOUT_SEC = 30

# -------------------------
# Helpers
# -------------------------
def _iso(d: Optional[str]) -> Optional[str]:
    """Convert YYYYMMDD -> YYYY-MM-DD if possible; else return as-is."""
    if not d or not isinstance(d, str) or not d.isdigit() or len(d) != 8:
        return d
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

def _derive_action_type(sub: Dict[str, Any]) -> str:
    stype = (sub.get("submission_type") or "").upper()
    status = (sub.get("submission_status") or "").upper()
    desc = sub.get("submission_class_code_description") or sub.get("submission_class_code") or ""
    if stype == "ORIG" and status == "AP":
        return "Approval"
    if stype == "SUPPL" and status == "AP":
        return f"Supplement - {desc}" if desc else "Supplement"
    return f"Submission - {stype or 'UNKNOWN'}"

def _best_submission_date(sub: Dict[str, Any]) -> Optional[str]:
    """Pick an action_date: submission_status_date first, else latest doc.date."""
    ssd = _iso(sub.get("submission_status_date"))
    if ssd:
        return ssd
    docs = sub.get("application_docs") or []
    dates = [d for d in (doc.get("date") for doc in docs) if d]
    return _iso(sorted(dates)[-1]) if dates else None

def _safe_json_loads(x: Any) -> Any:
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return None
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return None
    return x

def _first_not_empty(*vals):
    for v in vals:
        if v:
            return v
    return None

def _hash_fallback(*parts: str, n=12) -> str:
    s = "|".join([p or "" for p in parts])
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

# -------------------------
# Neo4j Manager
# -------------------------
class Neo4jManager:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_write(self, cypher: str, **params):
        with self.driver.session() as session:
            return session.execute_write(lambda tx: tx.run(cypher, **params).consume())

    def run_query(self, cypher: str, **params):
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            return [r.data() for r in result]

    def ensure_constraints(self):
        stmts = [
            "CREATE CONSTRAINT drug_uid IF NOT EXISTS FOR (d:Drug) REQUIRE d.drug_uid IS UNIQUE",
            "CREATE CONSTRAINT app_key  IF NOT EXISTS FOR (a:Application) REQUIRE a.app_no IS UNIQUE",
            "CREATE CONSTRAINT sub_key  IF NOT EXISTS FOR (s:Submission) REQUIRE (s.app_no, s.type, s.number) IS UNIQUE",
            "CREATE CONSTRAINT doc_key  IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT prod_key IF NOT EXISTS FOR (p:Product) REQUIRE p.key IS UNIQUE",
            "CREATE CONSTRAINT icd_key  IF NOT EXISTS FOR (i:ICD10Code) REQUIRE i.code IS UNIQUE",
            "CREATE CONSTRAINT dis_key  IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
        ]
        for s in stmts:
            self.run_write(s)

# -------------------------
# Ingestion
# -------------------------
@dataclass
class IngestCounts:
    drugs: int = 0
    apps: int = 0
    subs: int = 0
    docs: int = 0
    prods: int = 0
    diseases: int = 0
    codes: int = 0

class DrugKGIngestor:
    def __init__(self, neo: Neo4jManager):
        self.neo = neo

    # ---- MERGE helpers ----
    def merge_drug(self, drug: Dict[str, Any]) -> None:
        self.neo.run_write("""
        MERGE (d:Drug {drug_uid:$uid})
          ON CREATE SET d.brand_names=$brand, d.generic_names=$generic,
                        d.spl_set_id=$spl, d.rxcui=$rxcui, d.unii=$unii
          ON MATCH  SET d.brand_names=coalesce(d.brand_names,$brand),
                      d.generic_names=coalesce(d.generic_names,$generic),
                      d.spl_set_id=coalesce(d.spl_set_id,$spl),
                      d.rxcui=coalesce(d.rxcui,$rxcui),
                      d.unii=coalesce(d.unii,$unii)
        """, uid=drug.get("drug_uid"), brand=drug.get("brand_names"),
           generic=drug.get("generic_names"), spl=drug.get("spl_set_id"),
           rxcui=drug.get("rxcui"), unii=drug.get("unii"))

    def merge_application(self, app_no: str, drug_uid: str) -> None:
        self.neo.run_write("""
        MERGE (a:Application {app_no:$app})
        WITH a
        MERGE (d:Drug {drug_uid:$uid})
        MERGE (d)-[:HAS_APPLICATION]->(a)
        """, app=app_no, uid=drug_uid)

    def merge_submission(self, app_no: str, sub: Dict[str, Any]) -> None:
        self.neo.run_write("""
        MERGE (s:Submission {app_no:$app, type:$type, number:$num})
          ON CREATE SET s.status=$status, s.status_date=$status_date,
                        s.class_code=$class_code, s.class_desc=$class_desc,
                        s.review_priority=$review_priority,
                        s.action_type=$action_type, s.action_date=$action_date
          ON MATCH  SET s.status=coalesce(s.status,$status),
                      s.status_date=coalesce(s.status_date,$status_date),
                      s.action_type=coalesce(s.action_type,$action_type),
                      s.action_date=coalesce(s.action_date,$action_date),
                      s.class_code=coalesce(s.class_code,$class_code),
                      s.class_desc=coalesce(s.class_desc,$class_desc),
                      s.review_priority=coalesce(s.review_priority,$review_priority)
        WITH s
        MATCH (a:Application {app_no:$app})
        MERGE (a)-[:HAS_SUBMISSION]->(s)
        """,
        app=app_no,
        type=sub.get("submission_type"),
        num=sub.get("submission_number"),
        status=sub.get("submission_status"),
        status_date=_iso(sub.get("submission_status_date")),
        class_code=sub.get("submission_class_code"),
        class_desc=sub.get("submission_class_code_description"),
        review_priority=sub.get("review_priority"),
        action_type=sub.get("action_type"),
        action_date=sub.get("action_date"),
        )

    def merge_document(self, app_no: str, sub: Dict[str, Any], doc: Dict[str, Any]) -> None:
        doc_id = str(doc.get("id") or _hash_fallback(doc.get("url") or "", n=16))
        self.neo.run_write("""
        MERGE (d:Document {id:$doc_id})
          ON CREATE SET d.type=$doc_type, d.date=$doc_date, d.url=$doc_url
          ON MATCH  SET d.type=coalesce(d.type,$doc_type),
                      d.date=coalesce(d.date,$doc_date),
                      d.url=coalesce(d.url,$doc_url)
        WITH d
        MATCH (a:Application {app_no:$app})
        MATCH (s:Submission {app_no:$app, type:$type, number:$num})
        MERGE (s)-[:HAS_DOCUMENT]->(d)
        """,
        app=app_no, type=sub.get("submission_type"), num=sub.get("submission_number"),
        doc_id=doc_id, doc_type=doc.get("type"),
        doc_date=_iso(doc.get("date")), doc_url=doc.get("url"))

    def merge_product(self, app_no: str, product: Dict[str, Any], drug_uid: str) -> None:
        # Build a stable product key: prefer NDC, else app_no#product_number
        ndcs = product.get("package_ndc") or product.get("product_ndc") or []
        if isinstance(ndcs, list) and ndcs:
            key = f"ndc:{ndcs[0]}"
        else:
            key = f"{app_no}#{product.get('product_number') or '000'}"

        self.neo.run_write("""
        MERGE (p:Product {key:$key})
          ON CREATE SET p.product_number=$num, p.dosage_form=$form,
                        p.route=$route, p.marketing_status=$status,
                        p.ndcs=$ndcs
          ON MATCH  SET p.dosage_form=coalesce(p.dosage_form,$form),
                      p.route=coalesce(p.route,$route),
                      p.marketing_status=coalesce(p.marketing_status,$status),
                      p.ndcs=coalesce(p.ndcs,$ndcs)
        WITH p
        MATCH (d:Drug {drug_uid:$uid})
        MERGE (d)-[:HAS_PRODUCT]->(p)
        """,
        key=key,
        num=product.get("product_number"),
        form=product.get("dosage_form"),
        route=product.get("route"),
        status=product.get("marketing_status"),
        ndcs=ndcs if isinstance(ndcs, list) else [ndcs] if ndcs else [],
        uid=drug_uid)

    def merge_disease_and_icd(self, drug_uid: str, disease_name: str, icd_list: List[str], rel_type: str) -> None:
        self.neo.run_write(f"""
        MERGE (dis:Disease {{name:$name}})
        WITH dis
        MATCH (d:Drug {{drug_uid:$uid}})
        MERGE (d)-[:{rel_type}]->(dis)
        """, name=disease_name, uid=drug_uid)
        for code in icd_list or []:
            self.neo.run_write("""
            MERGE (i:ICD10Code {code:$code})
            WITH i
            MATCH (dis:Disease {name:$name})
            MERGE (dis)-[:CODED_AS]->(i)
            """, code=code, name=disease_name)

    # ---- High-level ingest ----
    def ingest_fda_excel(self, path: str) -> IngestCounts:
        df = pd.read_excel(path)
        if "fda_approval_data" not in df.columns:
            raise ValueError(f"Expected column 'fda_approval_data' in {path}")
        # Optional helpful columns for IDs if you stored them:
        # brand_name, generic_names, spl_set_id, rxcui, unii
        counts = IngestCounts()
        for _, row in df.iterrows():
            raw = _safe_json_loads(row["fda_approval_data"])
            if not raw:
                continue
            # Each entry is a submission-like dict with application context
            # We also try to recover brand/generic and IDs from the same entries.
            # Prefer fields we know you stored; else derive from the first submission.
            first = raw[0]
            app_no = first.get("application_number") or ""
            brand = first.get("brand_names") or row.get("brand_name") or ""
            generic = first.get("generic_names") or row.get("generic_names") or ""
            # Optional identifiers if present in your pipeline
            spl = row.get("spl_set_id") or first.get("spl_set_id")
            rxcui = row.get("rxcui") or first.get("rxcui")
            unii = row.get("unii") or first.get("unii")

            # Build a stable drug_uid
            drug_uid = _first_not_empty(spl, None) or _hash_fallback(brand, generic, app_no)

            # Merge Drug
            self.merge_drug({
                "drug_uid": drug_uid,
                "brand_names": brand,
                "generic_names": generic,
                "spl_set_id": spl,
                "rxcui": rxcui,
                "unii": unii
            })
            counts.drugs += 1

            # Application node + submissions + docs
            if app_no:
                self.merge_application(app_no, drug_uid)
                counts.apps += 1

            # Some pipelines also keep a 'products' array alongside approvals; try to attach it if present on row
            products_blob = _safe_json_loads(row.get("products") if "products" in df.columns else None)

            for sub in raw:
                # Ensure derived action fields are present
                if not sub.get("action_type"):
                    sub["action_type"] = _derive_action_type(sub)
                if not sub.get("action_date"):
                    sub["action_date"] = _best_submission_date(sub)

                # Merge submission
                self.merge_submission(app_no, sub)
                counts.subs += 1

                # Merge docs
                for doc in sub.get("docs") or sub.get("application_docs") or []:
                    self.merge_document(app_no, sub, doc)
                    counts.docs += 1

                # Attach products if available on the same row
                if isinstance(products_blob, list):
                    for prod in products_blob:
                        self.merge_product(app_no, prod, drug_uid)
                        counts.prods += 1

        return counts

    def ingest_icd_excel(self, path: str, drug_uid_col: Optional[str] = None) -> IngestCounts:
        """
        Reads an Excel with four columns of list-of-dicts:
          - may_treat_diseases_icd10
          - may_prevent_diseases_icd10
          - may_diagnose_diseases_icd10
          - ci_with_diseases_icd10
        Optionally provide a column name (`drug_uid_col`) to join drugs; otherwise,
        we use a hash of brand/generic if present in the file.
        """
        df = pd.read_excel(path)
        counts = IngestCounts()
        rel_map = {
            "may_treat_diseases_icd10": "TREATS",
            "may_prevent_diseases_icd10": "MAY_PREVENT",
            "may_diagnose_diseases_icd10": "MAY_DIAGNOSE",
            "ci_with_diseases_icd10": "CI_WITH",
        }
        # Try to infer a drug key per row
        for _, row in df.iterrows():
            # Preferred: explicit drug_uid column if you saved it
            uid = row.get(drug_uid_col) if drug_uid_col else None
            if not uid:
                brand = row.get("brand_name") or row.get("Drug Name") or row.get("brand_names") or ""
                generic = row.get("generic_names") or ""
                uid = _hash_fallback(brand, generic, n=12)
            # For each relationship column
            for col, rel in rel_map.items():
                if col not in df.columns:
                    continue
                payload = _safe_json_loads(row[col])
                if not payload:
                    continue
                # payload is list of dicts: [{ "Fever": ["R50.9","R50.82"] }, ...]
                if isinstance(payload, list):
                    for d in payload:
                        if not isinstance(d, dict):
                            continue
                        for disease, codes in d.items():
                            codes = codes or []
                            self.merge_disease_and_icd(uid, disease, codes, rel)
                            counts.diseases += 1
                            counts.codes += len(codes)
        return counts

# -------------------------
# Query API (“package”-like)
# -------------------------
class DrugKG:
    def __init__(self, uri=NEO4J_URI_DEFAULT, user=NEO4J_USER_DEFAULT, password=NEO4J_PASS_DEFAULT):
        self.neo = Neo4jManager(uri, user, password)

    def close(self):
        self.neo.close()

    # Common lookups
    def get_drug_by_brand(self, brand: str) -> List[Dict[str, Any]]:
        return self.neo.run_query("""
        MATCH (d:Drug) WHERE toUpper(d.brand_names) CONTAINS toUpper($brand)
        OPTIONAL MATCH (d)-[:HAS_APPLICATION]->(a:Application)
        RETURN d AS drug, collect(DISTINCT a.app_no) AS application_numbers
        """, brand=brand)

    def latest_action_for_app(self, app_no: str) -> Optional[Dict[str, Any]]:
        rows = self.neo.run_query("""
        MATCH (:Application {app_no:$app})-[:HAS_SUBMISSION]->(s:Submission)
        RETURN s.action_type AS action_type, s.action_date AS action_date
        ORDER BY s.action_date DESC NULLS LAST LIMIT 1
        """, app=app_no)
        return rows[0] if rows else None

    def timeline_for_drug(self, brand: str) -> List[Dict[str, Any]]:
        return self.neo.run_query("""
        MATCH (d:Drug) WHERE toUpper(d.brand_names) CONTAINS toUpper($brand)
        MATCH (d)-[:HAS_APPLICATION]->(a:Application)-[:HAS_SUBMISSION]->(s:Submission)
        RETURN a.app_no AS app_no, s.type AS submission_type, s.number AS submission_number,
               s.action_type AS action_type, s.action_date AS action_date
        ORDER BY app_no, action_date
        """, brand=brand)

    def drugs_for_icd(self, icd_code: str) -> List[Dict[str, Any]]:
        return self.neo.run_query("""
        MATCH (d:Drug)-[:TREATS|:MAY_PREVENT|:MAY_DIAGNOSE|:CI_WITH]->(:Disease)-[:CODED_AS]->(i:ICD10Code {code:$code})
        RETURN DISTINCT d.brand_names AS brand_names, d.drug_uid AS drug_uid
        """, code=icd_code)

    def documents_for_app(self, app_no: str) -> List[Dict[str, Any]]:
        return self.neo.run_query("""
        MATCH (:Application {app_no:$app})-[:HAS_SUBMISSION]->(s:Submission)-[:HAS_DOCUMENT]->(doc:Document)
        RETURN s.type AS submission_type, s.number AS submission_number,
               doc.type AS doc_type, doc.date AS doc_date, doc.url AS doc_url
        ORDER BY doc_date
        """, app=app_no)

# -------------------------
# CLI
# -------------------------
def cmd_ingest(args):
    neo = Neo4jManager(args.uri, args.user, args.password)
    ing = DrugKGIngestor(neo)
    try:
        neo.ensure_constraints()
        total = IngestCounts()
        if args.fda:
            c = ing.ingest_fda_excel(args.fda)
            total.drugs += c.drugs; total.apps += c.apps; total.subs += c.subs
            total.docs += c.docs; total.prods += c.prods
            print(f"[FDA] drugs={c.drugs} apps={c.apps} subs={c.subs} docs={c.docs} prods={c.prods}")
        if args.icd:
            c = ing.ingest_icd_excel(args.icd, drug_uid_col=args.drug_uid_col)
            total.diseases += c.diseases; total.codes += c.codes
            print(f"[ICD] diseases={c.diseases} codes={c.codes}")
        print(f"[TOTAL] {total}")
    finally:
        neo.close()

def cmd_query(args):
    kg = DrugKG(args.uri, args.user, args.password)
    try:
        if args.op == "latest-action":
            out = kg.latest_action_for_app(args.app)
            print(json.dumps(out or {}, indent=2))
        elif args.op == "timeline":
            out = kg.timeline_for_drug(args.brand)
            print(json.dumps(out, indent=2))
        elif args.op == "drugs-for-icd":
            out = kg.drugs_for_icd(args.code)
            print(json.dumps(out, indent=2))
        elif args.op == "drug":
            out = kg.get_drug_by_brand(args.brand)
            print(json.dumps(out, indent=2))
        elif args.op == "docs":
            out = kg.documents_for_app(args.app)
            print(json.dumps(out, indent=2))
        else:
            print("Unknown op. Use one of: latest-action, timeline, drugs-for-icd, drug, docs")
    finally:
        kg.close()

def main():
    parser = argparse.ArgumentParser(description="Build and query a Drug Knowledge Graph from Excel outputs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest Excel files into Neo4j")
    p_ing.add_argument("--uri", default=NEO4J_URI_DEFAULT)
    p_ing.add_argument("--user", default=NEO4J_USER_DEFAULT)
    p_ing.add_argument("--password", default=NEO4J_PASS_DEFAULT)
    p_ing.add_argument("--fda", help="Path to rxnav_with_fda.xlsx (must include fda_approval_data column)")
    p_ing.add_argument("--icd", help="Path to rxnav_with_icd10.xlsx (with *_icd10 columns)")
    p_ing.add_argument("--drug-uid-col", help="If your ICD sheet has a column with the drug UID; else a hash is used")
    p_ing.set_defaults(func=cmd_ingest)

    p_q = sub.add_parser("query", help="Run basic queries")
    p_q.add_argument("--uri", default=NEO4J_URI_DEFAULT)
    p_q.add_argument("--user", default=NEO4J_USER_DEFAULT)
    p_q.add_argument("--password", default=NEO4J_PASS_DEFAULT)
    p_q.add_argument("op", choices=["latest-action","timeline","drugs-for-icd","drug","docs"])
    p_q.add_argument("--app", help="Application number, e.g. BLA761248")
    p_q.add_argument("--brand", help="Brand name contains (case-insensitive)")
    p_q.add_argument("--code", help="ICD-10 code, e.g. R50.9")
    p_q.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
