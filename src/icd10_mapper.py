"""Offline ICD-10 mapping utilities for drug indications."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class DiseaseMapping:
    """Container for the curated ICD-10 code list of a disease."""

    disease: str
    codes: Tuple[str, ...]


class ICD10Mapper:
    """Map disease names to curated ICD-10 codes using local knowledge."""

    def __init__(self, max_codes: int = 7) -> None:
        self.max_codes = max_codes
        self._exact: Dict[str, Tuple[str, ...]] = {
            self._normalise(name): tuple(codes)
            for name, codes in MANUAL_MAPPINGS.items()
        }
        self._keyword_rules: List[Tuple[re.Pattern[str], Tuple[str, ...]]] = self._build_keyword_rules()

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _build_keyword_rules(self) -> List[Tuple[re.Pattern[str], Tuple[str, ...]]]:
        return [
            (re.compile(r"diabetes", re.IGNORECASE), ("E11.9", "E11.65", "E11.69", "E11.22", "E11.40")),
            (re.compile(r"hypertension", re.IGNORECASE), ("I10", "I11.9", "I12.9", "I15.9")),
            (re.compile(r"infection", re.IGNORECASE), ("A49.9", "B99.9", "A41.9", "B96.89")),
            (re.compile(r"pain", re.IGNORECASE), ("R52", "G89.29", "G89.4", "G89.18")),
            (re.compile(r"cancer|neoplasm|carcinoma|sarcoma", re.IGNORECASE), ("C80.1", "C79.9", "D49.9", "C76.0")),
            (re.compile(r"depress|anxiety|bipolar|schizo", re.IGNORECASE), ("F32.9", "F41.9", "F31.9", "F20.9")),
            (re.compile(r"asthma|bronch|pneum|respir", re.IGNORECASE), ("J45.909", "J40", "J18.9", "J06.9")),
            (re.compile(r"arthritis|arthropathy|osteo", re.IGNORECASE), ("M19.90", "M06.9", "M17.9", "M16.9")),
            (re.compile(r"sepsis|septic", re.IGNORECASE), ("A41.9", "R65.21", "A41.51", "R65.20")),
            (re.compile(r"abscess", re.IGNORECASE), ("L02.91", "K65.1", "L03.90", "A41.9")),
            (re.compile(r"acidosis", re.IGNORECASE), ("E87.2", "E87.4", "E87.20")),
            (re.compile(r"alkalosis", re.IGNORECASE), ("E87.3", "E87.4")),
            (re.compile(r"acne", re.IGNORECASE), ("L70.0", "L70.9", "L70.3", "L70.8")),
            (re.compile(r"adenoma", re.IGNORECASE), ("D36.9", "D35.0", "D35.2", "D37.9")),
            (re.compile(r"adenomatous", re.IGNORECASE), ("D12.6", "D12.2", "K63.5", "Z83.71")),
            (re.compile(r"hyperplasia", re.IGNORECASE), ("N40.0", "E27.1", "E21.0", "N85.00")),
            (re.compile(r"agammaglobulinemia", re.IGNORECASE), ("D80.0", "D80.1", "D80.5")),
            (re.compile(r"afibrinogenemia", re.IGNORECASE), ("D68.2", "D68.0", "D68.4")),
            (re.compile(r"agoraphobia", re.IGNORECASE), ("F40.00", "F40.01", "F40.02")),
            (re.compile(r"airway obstruction", re.IGNORECASE), ("J98.8", "J39.8", "J96.00")),
            (re.compile(r"akathisia", re.IGNORECASE), ("G25.71", "T43.215A", "R25.8")),
            (re.compile(r"alcohol", re.IGNORECASE), ("F10.20", "F10.288", "G31.2", "G62.1")),
            (re.compile(r"alopecia", re.IGNORECASE), ("L65.9", "L63.9", "L64.9", "L63.0")),
            (re.compile(r"altitude", re.IGNORECASE), ("T70.20XA", "T70.21XA", "T70.29XA")),
            (re.compile(r"amebiasis", re.IGNORECASE), ("A06.9", "A06.5", "A06.0")),
            (re.compile(r"amyloidosis", re.IGNORECASE), ("E85.9", "E85.4", "E85.0")),
            (re.compile(r"amyotrophic", re.IGNORECASE), ("G12.21", "G12.20", "G12.22")),
            (re.compile(r"anaphylaxis", re.IGNORECASE), ("T78.2XXA", "T78.2XXD", "T78.40XA", "Z91.010")),
            (re.compile(r"ascites", re.IGNORECASE), ("R18.8", "K70.31", "K74.60")),
            (re.compile(r"tachycardia", re.IGNORECASE), ("R00.0", "I47.9", "I47.1", "I47.2")),
            (re.compile(r"rhinitis", re.IGNORECASE), ("J31.0", "J30.9", "J30.89", "J31.1")),
            (re.compile(r"colitis", re.IGNORECASE), ("K52.9", "K51.90", "A09")),
            (re.compile(r"dermatitis", re.IGNORECASE), ("L30.9", "L20.9", "L23.9", "L29.9")),
            (re.compile(r"dermatoses", re.IGNORECASE), ("L98.8", "L29.8", "L71.9")),
            (re.compile(r"edema", re.IGNORECASE), ("R60.9", "R60.0", "R60.1", "I50.9")),
            (re.compile(r"uveitis", re.IGNORECASE), ("H20.9", "H20.0", "H20.1", "H44.119")),
            (re.compile(r"gonorrhea", re.IGNORECASE), ("A54.00", "A54.9", "A54.30", "A54.5")),
            (re.compile(r"meningitis", re.IGNORECASE), ("G03.9", "A87.9", "G00.9", "A87.2")),
            (re.compile(r"lymphoma", re.IGNORECASE), ("C85.90", "C83.30", "C83.19", "C85.19")),
            (re.compile(r"crohn", re.IGNORECASE), ("K50.90", "K50.10", "K50.80", "K50.00")),
            (re.compile(r"nephropath", re.IGNORECASE), ("E11.21", "E11.22", "N18.9", "N18.4")),
            (re.compile(r"psychotic", re.IGNORECASE), ("F29", "F23", "F20.9", "F25.9")),
            (re.compile(r"bone disease", re.IGNORECASE), ("M89.9", "M94.9", "M90.9", "M86.9")),
            (re.compile(r"pemphigus", re.IGNORECASE), ("L10.9", "L10.0", "L10.1", "L10.2")),
            (re.compile(r"mycosis", re.IGNORECASE), ("C84.0", "B35.9", "B36.9")),
            (re.compile(r"conjunctivitis", re.IGNORECASE), ("H10.9", "H10.33", "H10.023", "H10.30")),
            (re.compile(r"hypotension", re.IGNORECASE), ("I95.9", "I95.1", "I95.2", "I95.89")),
            (re.compile(r"neuropathy", re.IGNORECASE), ("G62.9", "G63", "G60.9", "G64")),
            (re.compile(r"menopause", re.IGNORECASE), ("N95.1", "E28.39", "Z78.0")),
            (re.compile(r"infertilit", re.IGNORECASE), ("N97.9", "N97.0", "N46.9", "Z31.41")),
            (re.compile(r"embolism", re.IGNORECASE), ("I26.99", "I26.09", "I26.02", "I26.90")),
            (re.compile(r"ulcer", re.IGNORECASE), ("K25.9", "K26.9", "K27.9", "L98.499")),
            (re.compile(r"polyposis", re.IGNORECASE), ("D12.6", "K63.5", "Z83.71")),
            (re.compile(r"sclerosis", re.IGNORECASE), ("G35", "M34.9", "M47.819")),
            (re.compile(r"psori", re.IGNORECASE), ("L40.0", "L40.9", "L40.50", "L40.52")),
            (re.compile(r"eczema", re.IGNORECASE), ("L30.9", "L20.9", "L30.8")),
            (re.compile(r"bursitis", re.IGNORECASE), ("M71.9", "M70.60", "M70.51", "M71.40")),
            (re.compile(r"spasm", re.IGNORECASE), ("J98.01", "M62.838", "R25.2")),
            (re.compile(r"ischemi", re.IGNORECASE), ("I63.9", "I63.89", "G45.9")),
            (re.compile(r"neuropathi", re.IGNORECASE), ("G62.9", "G63", "G60.9", "G64")),
            (re.compile(r"sickle", re.IGNORECASE), ("D57.1", "D57.00", "D57.01", "D57.20")),
            (re.compile(r"dementia", re.IGNORECASE), ("F03.90", "F03.91", "G30.9", "G31.83")),
            (re.compile(r"malaria", re.IGNORECASE), ("B54", "B53.8", "B50.9")),
            (re.compile(r"cholangi", re.IGNORECASE), ("K83.09", "C22.1", "K80.50")),
            (re.compile(r"cholang", re.IGNORECASE), ("K83.09", "C22.1", "K80.50")),
            (re.compile(r"hepatitis", re.IGNORECASE), ("B19.20", "B19.10", "K75.9", "B17.10")),
            (re.compile(r"encephalitis", re.IGNORECASE), ("G04.90", "A86", "A85.8")),
            (re.compile(r"encephalopathy", re.IGNORECASE), ("G93.40", "G94", "I63.9")),
            (re.compile(r"cystic fibrosis", re.IGNORECASE), ("E84.9", "E84.0", "E84.19")),
            (re.compile(r"ulcerative", re.IGNORECASE), ("K51.90", "K51.80", "K51.00", "K51.211")),
            (re.compile(r"polyposis", re.IGNORECASE), ("D12.6", "K63.5", "Z83.71")),
            (re.compile(r"palsy", re.IGNORECASE), ("G80.9", "G83.1", "G80.8")),
            (re.compile(r"neuromuscular", re.IGNORECASE), ("G70.00", "G70.9", "G71.00")),
            (re.compile(r"shock", re.IGNORECASE), ("R57.9", "R57.1", "R57.8", "T79.4XXA")),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def lookup(self, disease: str) -> List[str]:
        """Return a curated ICD-10 code list for ``disease``."""
        if not disease or not isinstance(disease, str):
            return []
        key = self._normalise(disease)
        codes: List[str] = []
        if key in self._exact:
            codes.extend(self._exact[key])
        else:
            for stored_key, stored_codes in self._exact.items():
                if stored_key in key and stored_key != key and len(stored_key) > 3:
                    for code in stored_codes:
                        if code not in codes:
                            codes.append(code)
        for pattern, rule_codes in self._keyword_rules:
            if pattern.search(disease):
                for code in rule_codes:
                    if code not in codes:
                        codes.append(code)
        return codes[: self.max_codes]

    def map_many(self, diseases: Iterable[str]) -> List[DiseaseMapping]:
        results: List[DiseaseMapping] = []
        for disease in diseases:
            results.append(DiseaseMapping(disease=disease, codes=tuple(self.lookup(disease))))
        return results


def load_may_treat_diseases(excel_path: Path, column: str = "may_treat_diseases") -> List[str]:
    """Extract the flattened list of diseases from the workbook column."""
    import ast
    import pandas as pd

    df = pd.read_excel(excel_path)
    diseases: List[str] = []
    for value in df[column].dropna():
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                continue
            if isinstance(parsed, (list, tuple)):
                diseases.extend(str(item).strip() for item in parsed if str(item).strip())
    return diseases


def build_mapping_for_workbook(excel_path: Path) -> Dict[str, List[str]]:
    mapper = ICD10Mapper()
    disease_list = load_may_treat_diseases(excel_path)
    mapping: Dict[str, List[str]] = {}
    for disease in sorted(set(disease_list)):
        mapping[disease] = mapper.lookup(disease)
    return mapping


def map_disease_list(diseases: Iterable[str]) -> List[Dict[str, List[str]]]:
    mapper = ICD10Mapper()
    results: List[Dict[str, List[str]]] = []
    for disease in diseases:
        results.append({"disease": disease, "icd10_codes": mapper.lookup(disease)})
    return results


MANUAL_MAPPINGS: Dict[str, List[str]] = {
    "Hypertension": ["I10", "I11.9", "I12.9", "I13.10", "I15.9"],
    "Diabetes Mellitus, Type 2": ["E11.9", "E11.65", "E11.69", "E11.22", "E11.51"],
    "Diabetes Mellitus, Type 1": ["E10.9", "E10.65", "E10.69", "E10.21", "E10.40"],
    "Abnormalities, Drug-Induced": ["T50.905A", "T88.7XXA", "Y57.9", "Y59.9"],
    "Acute Kidney Injury": ["N17.9", "N17.0", "N17.1", "N17.8", "N99.0"],
    "Heart Failure": ["I50.9", "I50.32", "I50.22", "I50.82", "I11.0"],
    "Coronary Artery Disease": ["I25.10", "I25.110", "I25.119", "I25.9"],
    "Myocardial Infarction": ["I21.3", "I21.4", "I22.9", "I25.2", "I21.19"],
    "Atrial Fibrillation": ["I48.91", "I48.0", "I48.1", "I48.2", "I48.3"],
    "Atrial Flutter": ["I48.92", "I48.3", "I48.4", "I48.0"],
    "Ventricular Fibrillation": ["I49.01", "I47.2", "I46.2", "I49.02"],
    "Ventricular Premature Complexes": ["I49.3", "I49.40", "I49.49", "I47.1"],
    "Angina Pectoris": ["I20.9", "I20.8", "I20.0", "I25.119"],
    "Angina, Unstable": ["I20.0", "I21.4", "I25.110", "I20.8"],
    "Cardiomyopathy, Hypertrophic": ["I42.2", "I42.1", "I42.0", "I42.8"],
    "Hypertension, Pulmonary": ["I27.20", "I27.21", "I27.0", "I27.24"],
    "Cerebral Infarction": ["I63.9", "I63.50", "I63.40", "I63.00"],
    "Stroke": ["I63.9", "I63.50", "I63.40", "I63.00"],
    "Sepsis": ["A41.9", "A41.51", "A41.89", "R65.21"],
    "Shock, Septic": ["R65.21", "A41.9", "A41.51", "R65.20"],
    "Shock": ["R57.9", "R57.1", "R57.8", "T79.4XXA"],
    "Hyponatremia": ["E87.1", "E87.5", "E87.8"],
    "Hypernatremia": ["E87.0", "E86.0"],
    "Hypokalemia": ["E87.6", "E87.2", "E87.8"],
    "Hyperkalemia": ["E87.5", "E87.2", "E87.8"],
    "Hypercalcemia": ["E83.52", "E83.59", "E21.0", "E21.3"],
    "Hypocalcemia": ["E83.51", "E20.9", "E55.9"],
    "Magnesium Deficiency": ["E61.2", "E63.8", "E83.42", "E87.8"],
    "Hyperglycemia": ["R73.9", "E11.65", "E10.65", "E16.0"],
    "Hypoglycemia": ["E16.2", "E16.1", "E10.641", "E11.649"],
    "Hyperlipidemia": ["E78.5", "E78.2", "E78.49", "E78.0"],
    "Hyperlipoproteinemias": ["E78.5", "E78.49", "E78.2", "E78.00"],
    "Hypertriglyceridemia": ["E78.1", "E78.2", "E78.3", "E78.5"],
    "Hypercholesterolemia": ["E78.0", "E78.2", "E78.49", "E78.5"],
    "Goiter": ["E04.9", "E04.1", "E04.2", "E01.0"],
    "Obesity": ["E66.9", "E66.01", "E66.09", "E66.8"],
    "Metabolic Syndrome": ["E88.81", "E88.810", "E88.89"],
    "Dehydration": ["E86.0", "E86.1", "E87.1", "E87.6"],
    "Constipation": ["K59.00", "K59.03", "K59.09", "K59.04"],
    "Dry Eye Syndromes": ["H04.123", "H04.129", "H16.229", "H16.149"],
    "Gastroesophageal Reflux": ["K21.9", "K21.0", "K44.9"],
    "Hemorrhoids": ["K64.9", "K64.8", "K64.4", "K64.2"],
    "Dyspepsia": ["K30", "R10.13", "K21.9", "R10.11"],
    "Pain": ["R52", "G89.4", "G89.29", "G89.18", "M79.2"],
    "Pain, Chronic": ["G89.29", "R52", "G89.4", "G89.21", "G89.22"],
    "Pain, Intractable": ["G89.4", "R52", "G89.29", "G89.3", "G89.0"],
    "Pain, Acute": ["G89.1", "R52", "G89.11", "G89.12"],
    "Pain, Postoperative": ["G89.18", "G89.12", "G89.11", "G89.3", "R52"],
    "Fever": ["R50.9", "R50.2", "R50.81", "A41.9", "B34.9"],
    "Fatigue": ["R53.83", "R53.82", "R53.81", "R68.89"],
    "Cough": ["R05.9", "R05.1", "R05.8", "J20.9", "J40"],
    "Diarrhea": ["R19.7", "A09", "K52.9", "K52.1", "K52.2"],
    "Vomiting": ["R11.10", "R11.2", "R11.0", "K52.9"],
    "Nausea": ["R11.0", "R11.2", "R11.10", "T45.1X5A"],
    "Osteoarthritis": ["M19.90", "M17.9", "M16.9", "M15.9", "M19.91"],
    "Arthritis, Rheumatoid": ["M06.9", "M05.79", "M05.711", "M05.742", "M05.749"],
    "Arthritis, Psoriatic": ["L40.50", "L40.51", "L40.52", "M07.60", "M07.61"],
    "Spondylitis, Ankylosing": ["M45.9", "M45.6", "M45.0", "M45.4"],
    "Gout": ["M10.9", "M10.00", "M10.01", "M10.3", "M1A.9XX0"],
    "Fibromyalgia": ["M79.7", "R52", "G90.50"],
    "Back Pain": ["M54.5", "M54.50", "M54.51", "M54.59"],
    "Headache": ["R51.9", "G44.1", "G44.209", "G43.909"],
    "Heartburn": ["R12", "K21.9", "K21.0", "K30"],
    "Cluster Headache": ["G44.009", "G44.019", "G44.029", "G44.041"],
    "Migraine Disorders": ["G43.909", "G43.109", "G43.119", "G43.709"],
    "Multiple Sclerosis": ["G35", "G37.3", "G36.0", "G36.8"],
    "Alzheimer Disease": ["G30.9", "G30.1", "G30.8", "F02.80", "F02.81"],
    "Myasthenia Gravis": ["G70.00", "G70.01", "G70.80", "G70.81"],
    "Parkinson Disease": ["G20", "G20.A1", "G20.B1", "G20.C1", "G21.9"],
    "Muscle Spasticity": ["R25.2", "G80.1", "G81.10", "G82.50"],
    "Muscle Rigidity": ["R25.2", "G24.3", "G20.A1", "G83.89"],
    "Seizures": ["R56.9", "G40.909", "G40.409", "R56.1"],
    "Status Epilepticus": ["G41.909", "G41.409", "G41.801", "G41.101"],
    "Lennox Gastaut Syndrome": ["G40.814", "G40.813", "G40.812", "G40.811"],
    "Epilepsies, Partial": ["G40.209", "G40.219", "G40.109", "G40.119"],
    "Attention Deficit Disorder with Hyperactivity": ["F90.9", "F90.0", "F90.1", "F90.2"],
    "Autistic Disorder": ["F84.0", "F84.5", "F84.9"],
    "Anxiety Disorders": ["F41.9", "F41.1", "F41.0", "F40.01", "F40.00"],
    "Stress Disorders, Post-Traumatic": ["F43.10", "F43.11", "F43.12", "F43.8"],
    "Panic Disorder": ["F41.0", "F41.1", "F41.8", "F41.9"],
    "Obsessive-Compulsive Disorder": ["F42.9", "F42.2", "F42.3", "F42.8"],
    "Depressive Disorder": ["F32.9", "F33.9", "F32.1", "F33.1", "F32.2"],
    "Bipolar Disorder": ["F31.9", "F31.0", "F31.60", "F31.70", "F31.81"],
    "Schizophrenia": ["F20.9", "F20.0", "F20.3", "F20.5"],
    "Tourette Syndrome": ["F95.2", "F95.1", "F95.9", "F95.8"],
    "Sleep Initiation and Maintenance Disorders": ["G47.00", "G47.09", "F51.01", "F51.02"],
    "Insomnia": ["G47.00", "G47.09", "F51.01", "F51.02"],
    "Sleep Apnea": ["G47.33", "G47.30", "G47.39"],
    "Apnea": ["R06.81", "G47.30", "G47.33", "R06.89"],
    "Nephrotic Syndrome": ["N04.9", "N04.1", "N04.2", "N04.8"],
    "Urinary Tract Infections": ["N39.0", "N30.90", "N30.00", "N10"],
    "Urethritis": ["N34.1", "N34.2", "N34.3", "N37"],
    "Cystitis": ["N30.90", "N30.00", "N30.01", "N30.10"],
    "Kidney Disease": ["N18.9", "N18.3", "N18.4", "N18.5", "N18.6"],
    "Kidney Failure": ["N18.9", "N17.9", "N18.4", "N18.6"],
    "Kidney Stones": ["N20.0", "N20.1", "N20.2", "N20.9"],
    "Anemia": ["D64.9", "D50.9", "D53.9", "D63.8", "D55.9"],
    "Anemia, Iron-Deficiency": ["D50.9", "D50.8", "D50.0", "D50.1"],
    "Anemia, Sickle Cell": ["D57.1", "D57.00", "D57.01", "D57.20"],
    "Acquired Immunodeficiency Syndrome": ["B20", "Z21", "B24", "O98.7"],
    "AIDS-Related Opportunistic Infections": ["B20", "B59", "B45.0", "B45.7", "B58.9", "B37.0"],
    "Tuberculosis, Pulmonary": ["A15.0", "A15.4", "A16.0", "A16.2"],
    "Pneumonia, Bacterial": ["J15.9", "J18.9", "J15.20", "J15.8"],
    "Pneumonia, Pneumocystis": ["B59", "J16.8", "J18.9"],
    "Bronchitis": ["J40", "J20.9", "J41.0", "J42"],
    "Bronchitis, Chronic": ["J41.0", "J41.1", "J42", "J44.9"],
    "Bronchial Spasm": ["J98.01", "J45.901", "J45.41", "J45.42"],
    "Asthma": ["J45.909", "J45.40", "J45.50", "J45.901"],
    "Chronic Obstructive Pulmonary Disease": ["J44.9", "J44.1", "J44.0", "J44.89"],
    "Common Cold": ["J00", "J06.9", "J06.0"],
    "Influenza": ["J10.1", "J11.1", "J10.00", "J11.00"],
    "COVID-19": ["U07.1", "J12.82", "J96.01", "Z20.822"],
    "Skin Diseases": ["L98.9", "L08.9", "L30.9", "L29.9"],
    "Hidradenitis Suppurativa": ["L73.2", "L08.0", "L03.317", "L02.219"],
    "Inflammation": ["R76.8", "M79.89", "R60.9"],
    "Tachycardia": ["R00.0", "I47.9", "I47.1", "I47.2"],
    "Bradycardia": ["R00.1", "I49.8", "I44.2", "I44.1"],
    "Heart Block": ["I44.2", "I44.1", "I44.0", "I45.9"],
    "Rhinitis": ["J31.0", "J30.9", "J30.89", "J31.1"],
    "Colitis": ["K52.9", "K51.90", "K52.1", "A09"],
    "Facial Dermatoses": ["L71.9", "L98.8", "L29.8", "L85.3"],
    "Acne Vulgaris": ["L70.0", "L70.9", "L70.3", "L70.8"],
    "Scalp Dermatoses": ["L21.9", "L20.9", "L29.8", "L98.8"],
    "Foot Dermatoses": ["L98.8", "L29.8", "B35.3", "L30.9"],
    "Hand Dermatoses": ["L98.8", "L23.9", "L30.9", "L29.8"],
    "Leg Dermatoses": ["L98.8", "L23.9", "L30.9", "L29.8"],
    "Ventricular Dysfunction": ["I51.9", "I50.20", "I50.22", "I50.82"],
    "Epilepsy": ["G40.909", "G40.409", "G40.319", "G40.119"],
    "Rheumatic Diseases": ["M79.7", "M35.9", "M05.9", "M06.9", "M35.00"],
    "Collagen Diseases": ["M35.9", "M35.00", "M35.01", "M35.02"],
    "Synovitis": ["M65.9", "M65.80", "M65.812", "M65.819"],
    "Edema": ["R60.9", "R60.0", "R60.1", "I50.9"],
    "Uveitis": ["H20.9", "H20.0", "H20.1", "H44.119"],
    "Gonorrhea": ["A54.00", "A54.9", "A54.30", "A54.5"],
    "Meningitis": ["G03.9", "A87.9", "G00.9", "A87.2"],
    "Lymphoma": ["C85.90", "C83.30", "C83.19", "C85.19"],
    "Drug-Related Side Effects and Adverse Reactions": ["T50.905A", "T88.7XXA", "Y57.9", "Y59.9"],
    "Poisoning": ["T50.901A", "T50.902A", "T50.905A", "T36.0X1A"],
    "Drug Overdose": ["T50.901A", "T50.902A", "T50.904A", "T50.905A"],
    "Crohn Disease": ["K50.90", "K50.10", "K50.80", "K50.00"],
    "Diabetic Nephropathies": ["E11.21", "E11.22", "N18.9", "N18.4", "R80.9"],
    "Heart Arrest": ["I46.9", "I46.2", "I46.8", "I46.1"],
    "Psychotic Disorders": ["F29", "F23", "F20.9", "F25.9"],
    "Dermatitis": ["L30.9", "L20.9", "L23.9", "L29.9"],
    "Erythema Multiforme": ["L51.9", "L51.0", "L51.1", "L51.8"],
    "Hyperhidrosis": ["R61", "L74.512", "L74.522", "L74.51"],
    "Bone Diseases": ["M89.9", "M94.9", "M90.9", "M86.9"],
    "Dermatitis Herpetiformis": ["L13.0", "L13.1", "L13.9"],
    "Pemphigus": ["L10.9", "L10.0", "L10.1", "L10.2"],
    "Mycosis Fungoides": ["C84.0", "C84.00", "C84.04", "C84.08"],
    "Enterocolitis": ["K52.9", "K52.2", "K52.89"],
    "Giardiasis": ["A07.1", "A07.2", "A07.3", "A07.8"],
    "Cholera": ["A00.9", "A00.0", "A00.1", "A00.8"],
    "Psittacosis": ["A70", "A70.0", "A70.9", "J16.0"],
    "Conjunctivitis": ["H10.9", "H10.33", "H10.023", "H10.30"],
    "Keratitis": ["H16.9", "H16.001", "H16.009", "H16.211"],
    "Iritis": ["H20.9", "H20.0", "H20.1", "H20.049"],
    "Hypotension": ["I95.9", "I95.1", "I95.2", "I95.89"],
    "Embolism": ["I26.99", "I26.09", "I26.02", "I26.90"],
    "Thrombophlebitis": ["I80.9", "I80.00", "I80.01", "I82.409"],
    "Psoriasis": ["L40.0", "L40.9", "L40.50", "L40.52"],
    "Dermatitis, Atopic": ["L20.9", "L20.89", "L20.81", "L20.84"],
    "Dermatitis, Contact": ["L23.9", "L24.9", "L25.9", "L23.7"],
    "Rosacea": ["L71.9", "L71.0", "L71.8", "L71.1"],
    "Cellulitis": ["L03.90", "L03.119", "L03.116", "L03.115"],
    "Soft Tissue Infections": ["L08.9", "M72.6", "A49.9", "T81.4XXA"],
    "Abscess": ["L02.91", "L02.419", "K65.0", "J36"],
    "Eye Infections, Bacterial": ["H10.9", "H10.023", "H16.009", "H44.009"],
    "Glaucoma": ["H40.9", "H40.11X0", "H40.10X0", "H40.20X0"],
    "Conjunctivitis, Allergic": ["H10.45", "H10.44", "H10.10", "H10.13"],
    "Blepharitis": ["H01.009", "H01.006", "H01.003"],
    "Otitis Media": ["H66.90", "H66.91", "H66.92", "H66.93"],
    "Otitis Externa": ["H60.90", "H60.91", "H60.92", "H60.93"],
    "Sinusitis": ["J01.90", "J01.80", "J32.9", "J32.0"],
    "Esophageal Diseases": ["K22.9", "K20.9", "K22.10", "K22.2"],
    "Mouth Diseases": ["K13.79", "K12.30", "K14.0", "K12.2"],
    "Urinary Bladder Diseases": ["N32.9", "N30.90", "N32.81", "N31.9"],
    "Endometriosis": ["N80.9", "N80.3", "N80.4", "N80.5"],
    "Menorrhagia": ["N92.0", "N92.1", "N92.4", "N92.5"],
    "Pregnancy Complications, Hematologic": ["O99.119", "O99.111", "O99.12", "O99.13"],
    "Dysmenorrhea": ["N94.6", "N94.4", "N94.5"],
    "Amenorrhea": ["N91.2", "N91.1", "N91.0", "N91.3"],
    "Premenstrual Syndrome": ["N94.3", "F32.81", "N94.3"],
    "Uterine Hemorrhage": ["N93.9", "N93.8", "N93.0", "O46.9"],
    "Vaginosis, Bacterial": ["N76.0", "B96.89", "A49.9"],
    "Candidiasis, Vulvovaginal": ["B37.3", "N76.0", "N76.1", "A49.9"],
    "Candidiasis": ["B37.9", "B37.0", "B37.2", "B37.3"],
    "Staphylococcal Infections": ["A41.0", "A41.01", "A41.02", "A49.01", "B95.61"],
    "Streptococcal Infections": ["A40.0", "A40.1", "A49.1", "B95.5"],
    "Escherichia coli Infections": ["A49.8", "B96.20", "B96.29", "N39.0"],
    "Pseudomonas Infections": ["A49.8", "B96.5", "B96.89", "J15.1"],
    "Klebsiella Infections": ["A49.8", "B96.1", "B96.20", "J15.0"],
    "Proteus Infections": ["A49.8", "B96.4", "B96.89", "N39.0"],
    "Haemophilus Infections": ["A49.2", "B96.3", "B96.89", "J14"],
    "Syphilis": ["A53.9", "A51.9", "A52.9", "A50.9"],
    "Chancroid": ["A57", "A57.0", "A57.9"],
    "Tetanus": ["A35", "A33", "A34", "A36.0"],
    "Anthrax": ["A22.9", "A22.1", "A22.0", "A22.7"],
    "Legionnaires' Disease": ["A48.1", "A48.2", "J15.7", "J18.9"],
    "Toxoplasmosis": ["B58.9", "B58.0", "B58.2", "B58.8"],
    "Trachoma": ["A71.9", "A71.0", "A71.1", "A71.2"],
    "Tinea Pedis": ["B35.3", "B35.30", "B35.31", "B35.39"],
    "Tinea Versicolor": ["B36.0", "B36.9", "B36.8", "B36.3"],
    "Breast Neoplasms": ["C50.919", "C50.911", "C50.912", "D05.10", "D05.11"],
    "Prostatic Neoplasms": ["C61", "D07.5", "C79.82", "C79.51"],
    "Ovarian Neoplasms": ["C56.9", "C56.1", "C56.2", "C56.3"],
    "Melanoma": ["C43.9", "C43.59", "C43.39", "C43.21"],
    "Neuroblastoma": ["C74.90", "C74.10", "C47.9", "C79.7"],
    "Hodgkin Disease": ["C81.90", "C81.00", "C81.10", "C81.70"],
    "Endometrial Hyperplasia": ["N85.00", "N85.01", "N85.02", "N85.1"],
    "Primary Ovarian Insufficiency": ["E28.39", "E28.8", "N95.1"],
    "Menopause": ["N95.1", "E28.39", "Z78.0"],
    "Hot Flashes": ["N95.1", "E28.39", "R23.2", "R23.8"],
    "Addison Disease": ["E27.1", "E27.2", "E27.40"],
    "Cushing Syndrome": ["E24.9", "E24.0", "E24.8"],
    "Hyperthyroidism": ["E05.90", "E05.00", "E05.10", "E05.20"],
    "Thyrotoxicosis": ["E05.90", "E05.01", "E05.11", "E05.21"],
    "Hypothyroidism": ["E03.9", "E03.8", "E03.4", "E89.0"],
    "Thyroiditis": ["E06.9", "E06.3", "E06.5", "E06.1"],
    "Hyperparathyroidism": ["E21.3", "E21.0", "E21.1"],
    "Hypoparathyroidism": ["E20.9", "E20.8", "E89.2"],
    "Acromegaly": ["E22.0", "D35.2", "E22.1"],
    "Adrenal Insufficiency": ["E27.40", "E27.1", "E27.2"],
    "Hypogonadism": ["E29.1", "E29.8", "E28.39", "E23.0"],
    "Hyperprolactinemia": ["E22.1", "E22.2", "D35.2"],
    "Lupus Erythematosus, Systemic": ["M32.9", "M32.10", "M32.12", "M32.0"],
    "Sarcoidosis": ["D86.9", "D86.0", "D86.1", "D86.2"],
    "Berylliosis": ["J63.2", "J99", "J70.8", "D70.9"],
    "Serum Sickness": ["T80.6XXA", "T80.6XXD", "T80.6XXS", "T78.2XXA"],
    "Sjogren Syndrome": ["M35.00", "M35.01", "M35.02", "M35.03"],
    "Behcet Syndrome": ["M35.2", "M35.21", "M35.22", "M35.23"],
    "Graft vs Host Disease": ["D89.810", "D89.811", "D89.812", "D89.813"],
    "Dermatomyositis": ["M33.90", "M33.10", "M33.20", "M33.21"],
    "Polymyositis": ["M33.20", "M33.21", "M33.22"],
    "Fibromyalgia": ["M79.7", "R52", "G90.50"],
    "Multiple Myeloma": ["C90.00", "C90.01", "C90.02", "C90.20"],
    "Leukemia": ["C95.90", "C92.10", "C92.00", "C91.00", "C95.10"],
    "Purpura, Thrombocytopenic, Idiopathic": ["D69.3", "D69.41", "D69.42", "D69.49"],
    "Postoperative Complications": ["T81.9XXA", "T81.4XXA", "T88.8XXA", "T88.9XXA"],
    "Opioid-Related Disorders": ["F11.20", "F11.23", "F11.24", "F11.90"],
    "Heroin Dependence": ["F11.20", "F11.21", "F11.23", "F11.29"],
    "Alcohol Withdrawal Delirium": ["F10.231", "F10.239", "F10.230", "F10.20"],
    "Alcoholism": ["F10.20", "F10.21", "F10.24", "F10.29"],
    "Substance Withdrawal Syndrome": ["F19.239", "F11.23", "F13.23", "F10.23"],
    "Burns": ["T30.0", "T21.00XA", "T22.099A", "T23.099A"],
    "Wounds and Injuries": ["T14.90XA", "S01.81XA", "S41.109A", "S31.109A"],
    "Bites, Human": ["W50.3XXA", "S61.459A", "S51.859A", "S01.85XA"],
    "Cachexia": ["R64", "E43", "C80.0", "C15.9"],
    "Vitamin D Deficiency": ["E55.9", "E83.51", "M83.9"],
    "Hyperuricemia": ["E79.0", "M10.9", "M1A.9XX0"],
    "Endocarditis, Bacterial": ["I33.0", "I33.9", "I38", "I39"],
    "Endocarditis": ["I33.9", "I33.0", "I38", "I39"],
    "Rheumatic Fever": ["I00", "I01.9", "I02.0", "I01.1"],
    "Psoriatic Arthritis": ["L40.50", "L40.51", "M07.60", "M07.61"],
    "Osteomyelitis": ["M86.9", "M86.00", "M86.10", "M86.6"],
    "Otitis": ["H66.90", "H66.91", "H66.92", "H66.93"],
    "Urticaria": ["L50.9", "L50.1", "L50.8", "L50.6"],
    "Pruritus": ["L29.9", "L29.2", "L29.3", "L29.8"],
    "Pruritus Ani": ["L29.0", "K64.5", "L29.9"],
    "Rosacea": ["L71.9", "L71.0", "L71.8", "L71.1"],
    "Neutropenia": ["D70.9", "D70.1", "D70.3", "D70.8"],
    "Nephritis": ["N05.9", "N05.7", "N05.8", "N01.9"],
    "Menopause": ["N95.1", "E28.39", "Z78.0"]
}


__all__ = [
    "ICD10Mapper",
    "DiseaseMapping",
    "load_may_treat_diseases",
    "build_mapping_for_workbook",
    "map_disease_list",
]
