import re
from medkit.text.preprocessing import CharReplacer, LIGATURE_RULES, SIGN_RULES
from medkit.core.text import Segment, Span

# -----------------------------
# Helper functions
# -----------------------------


def detect_sex(text: str) -> str:
    """Detect patient sex from text and normalize to 'F' or 'M'."""
    text = text.lower()
    if re.search(r'\b(f|female|femme|feminin)\b', text):
        return 'F'
    elif re.search(r'\b(m|male|homme|masculin)\b', text):
        return 'M'
    else:
        return 'F'  # default if unknown

def detect_lab_problems(text: str, sex: str = "F", margin_ratio: float = 0.015) -> str:
    """
    Detect lab abnormalities and annotate inline with ICD-10-compatible interpretations.
    Adds tolerance margin (e.g. Na 134 won't be flagged as abnormal).
    """
    
    lab_rules = [
        # --- Hemoglobin ---
        {
            "name": "Anémie",
            "regex": r"\b(?:Hb|Hémoglobine|hémoglobine|hemoglobine)\b[^\d]{0,10}?([\d.,]+)",
            "threshold_low": 12 if sex == "F" else 13,
            "direction": "low",  # only low values are abnormal
            "unit_fix": lambda v: v / 10 if v > 20 else v,  # g/L → g/dL
        },
        # --- Potassium ----
        {
            "name": "Hypokaliémie/Hyperkaliémie",
            "regex": r"\b(?:K|Potassium|potassium)\s*[⁺:=]?\s*[^\d]{0,5}([\d.,]+)",
            "threshold_low": 3.5,
            "threshold_high": 5.0,
            "direction": "both",
        },
        # --- Sodium ---
        {
            "name": "Hyponatrémie/Hypernatrémie",
            "regex": r"\b(?:Na|Sodium)\s*[⁺:]?\s*[^\d]{0,5}([\d.,]+)",
            "threshold_low": 135,
            "threshold_high": 145,
            "direction": "both",
        },
        # --- Creatinine ---
        {
            "name": "Insuffisance rénale",
            "regex": r"\b(?:Créat(?:inine)?|Creatinine|créatininémie)\b[^\d]{0,10}?([\d.,]+)",
            "threshold_high": 120,
            "direction": "high",
        },
        # --- Vitamine D ---
        {
            "name": "Carence en vitamine D / Hypervitaminose D",
            "regex": r"\b(?:vit(?:amine)?\s*d)\b[^\d]{0,10}?([\d.,]+)",
            "threshold_low": 30,
            "threshold_high": 100,
            "direction": "both",
        },
        # --- Urée ---
        {
            "name": "Rétention azotée",
            "regex": r"\b(?:Urée|Urea)\b[^\d]{0,10}?([\d.,]+)",
            "threshold_high": 7.5,
            "direction": "high",
        },
        # --- DFG ---
        {
            "name": "DFG bas",
            "regex": r"\b(?:DFG|clairance|filtration)\b[^\d]{0,10}?([\d.,]+)",
            "threshold_low": 15,
            "direction": "low",
        },
        # --- CRP ---
        {
            "name": "CRP élevée / Syndrome inflammatoire",
            "regex": r"\b(?:CRP|C[-\s]?reactive\s*protein|protéine\s*C\s*réactive)\b[^\d]{0,10}?([\d.,]+)",
            "threshold_low": 10,
            "direction": "high", # higher means abnormal
        },
        # --- Gamma-GT ---
        {
            "name": "augmentation de la gamma-glutamyl transférase / augmentation de la GGT",
            #"regex": r"\b(?:Gamma[\s-]*GT|γ[\s-]*GT|Gamma\s*glutamyl[\s-]*transpeptidase)\b[^\d]{0,10}?([\d.,]+)",
            "regex": r"\b(?:[Gg]amma[\s\u2010\u2011\u2012\u2013\u2014-]*GT|γ[\s-]*GT|Gamma\s*glutamyl[\s-]*transpeptidase)\b[^0-9]{0,40}?([\d.,]+)",

            "threshold_high": 55 if sex == "M" else 45,
            "direction": "high",
        },
    ]
    
    for lab in lab_rules:
        pattern = re.compile(lab["regex"], flags=re.IGNORECASE)

        def replacer(match):
            val_str = match.group(1).replace(",", ".")
            try:
                val = float(val_str)
            except ValueError:
                return match.group(0)

            if "unit_fix" in lab:
                val = lab["unit_fix"](val)

            annotation = match.group(0)
            low = lab.get("threshold_low")
            high = lab.get("threshold_high")

            # Apply tolerance margin
            low_margin = low * (1 - margin_ratio) if low else None
            high_margin = high * (1 + margin_ratio) if high else None
            direction = lab.get("direction", "both")

            # Apply only relevant checks
            if direction in ["low", "both"] and low and val < low_margin:
                annotation += f" [{lab['name']} ({val} < {low})]"
            elif direction in ["high", "both"] and high and val > high_margin:
                annotation += f" [{lab['name']} ({val} > {high})]"
            elif direction == "high" and not high and low and val > low * (1 + margin_ratio):
                annotation += f" [{lab['name']} ({val} > {low})]"

            return annotation

        text = pattern.sub(replacer, text)

    return text


def detect_potassium_recharge(text: str) -> str:
    """
    Detect Hypokaliémie if text mentions 'recharge potassium' 
    regardless of ionogram results.
    """
    lines = text.split("\n")
    new_lines = []
    for line in lines:
        l = line
        if re.search(r'recharge potassium', line, flags=re.IGNORECASE):
            # Append annotation if not already present
            if "[Detected problem: Hypokaliémie" not in line:
                l += " [Hypokaliémie (assumed due to potassium recharge)]"
        new_lines.append(l)
    return "\n".join(new_lines)

import re

def detect_creatinine_semantic(text: str) -> str:
    """
    Detect 'Créatinine élevée / augmentée / haute' etc.
    and annotate as Insuffisance rénale even if numeric value is low.
    """
    lines = text.split("\n")
    new_lines = []

    for line in lines:
        l = line
        # detect créatinine and an adjective indicating elevation
        if re.search(r'\bcr[ée]atinine\b', line, flags=re.IGNORECASE) and \
           re.search(r'\b(élevée?s?|augmentée?s?|haute?s?)\b', line, flags=re.IGNORECASE):
            if "Insuffisance rénale" not in line:
                l += " [Insuffisance rénale (valeur élevée – N18)]"
        elif re.search(r'\bcr[ée]atinine\b', line, flags=re.IGNORECASE) and \
             re.search(r'\b(basse?s?|diminuée?s?|abaissée?s?)\b', line, flags=re.IGNORECASE):
            if "Insuffisance rénale" not in line:
                l += " [Valeur de créatinine basse – à interpréter selon contexte]"
        new_lines.append(l)

    return "\n".join(new_lines)


def detect_glasgow_gcs(text: str) -> str:
    """
    Detect Glasgow Coma Scale / GCS score and annotate if <15 as R40 (Impaired consciousness).
    Matches patterns like "Glasgow 12", "GCS 8", "Glasgow/GCS xxx", etc.
    """
    lines = text.split("\n")
    new_lines = []
    
    for line in lines:
        l = line
        # Match Glasgow/GCS with a numeric value
        match = re.search(r'\b(?:Glasgow|GCS|Glasgow\s*Coma\s*Scale)\b[^\d]{0,10}?([\d]+)', line, flags=re.IGNORECASE)
        if match:
            try:
                gcs_value = int(match.group(1))
                # If GCS < 15, it indicates impaired consciousness (R40)
                if gcs_value < 15:
                    annotation = f" [Perte de conscience / Obnubilation (R40 – GCS {gcs_value})]"
                    if "R40" not in line and "[Perte de conscience" not in line:
                        l += annotation
            except (ValueError, IndexError):
                pass
        new_lines.append(l)
    
    return "\n".join(new_lines)


def enrich_text_with_clinically_significant_findings(text: str, sex: str = "F") -> str:
    """
    Pre-process text by running all detection functions to annotate clinically significant 
    findings (lab abnormalities, Glasgow/GCS, potassium recharge, creatinine semantic elevation).
    
    This enriches the text so the LLM can extract findings that have diagnostic significance,
    rather than blindly excluding all constants/lab values.
    """
    text = detect_lab_problems(text, sex=sex)
    text = detect_potassium_recharge(text)
    text = detect_creatinine_semantic(text)
    text = detect_glasgow_gcs(text)
    text = detect_clinical_exam_signs(text)
    return text


def detect_clinical_exam_signs(text: str) -> str:
    """
    Detect clinical disorders/signs mentioned in exam narrative (e.g., "L'examen clinique retrouvait...")
    and annotate them as codifiable diagnoses.
    
    These findings are often buried in exam descriptions and may be missed by the LLM
    without an explicit annotation signal. Each pattern maps to a CIM-10-relevant label.
    """
    # Each rule: (regex to detect the finding, annotation label)
    clinical_sign_rules = [
        # --- Hydration ---
        (r"\bdés?hydratation\b(?:\s+clinique)?", "Déshydratation (E86)"),
        # --- Nutrition ---
        (r"\bdénutrition\b", "Dénutrition / Malnutrition (E46)"),
        (r"\bcachexie\b", "Cachexie / Dénutrition sévère (E41)"),
        # --- Abdominal signs potentially indicating occlusion ---
        (r"(?:arrêt des matières et des gaz|occlusion intestinale|iléus)", "Occlusion intestinale (K56)"),
        # --- Globe vésical / Urinary retention ---
        (r"\bglobe\s+vésical\b", "Rétention aiguë d'urine / Globe vésical (R33)"),
        (r"\brétention\s+(?:aiguë\s+)?(?:d['']urine|urinaire)\b", "Rétention aiguë d'urine (R33)"),
        # --- Fecaloma / Stercoral stasis ---
        (r"\bfécalome\b", "Fécalome / Constipation sévère (K59.0)"),
        # --- Confusion / Altered consciousness ---
        (r"\bconfusion\s+(?:mentale|aiguë)?\b", "Confusion mentale (F05)"),
        (r"\bobnubilation\b", "Obnubilation / Perte de conscience (R40)"),
        # --- Falls ---
        (r"\bchute(?:s)?\b(?:\s+(?:récente|à\s+répétition|traumatique))?", "Chute (W19)"),
        # --- Pressure ulcers / Skin ---
        (r"\bescare(?:s)?\b", "Escarre (L89)"),
        # --- Edema ---
        (r"\bœdème(?:s)?\s+(?:des\s+membres\s+inférieurs|des\s+MI|bilatéral(?:aux)?)\b", "Œdème des membres inférieurs (R60)"),
        # --- Dyspnea ---
        (r"\bdyspnée\b", "Dyspnée (R06.0)"),
        # --- Ascites ---
        (r"\bascite\b", "Ascite (R18)"),
    ]

    lines = text.split("\n")
    new_lines = []
    for line in lines:
        l = line
        for pattern, label in clinical_sign_rules:
            if re.search(pattern, line, flags=re.IGNORECASE):
                if f"[{label}" not in line:
                    l += f" [{label}]"
                    break  # one annotation per line to avoid duplicates from overlapping patterns
        new_lines.append(l)

    return "\n".join(new_lines)