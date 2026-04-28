# Fix: Ensure re is imported for regex usage
from matplotlib import text
import requests
import json
import pandas as pd
from typing import Any
import re


def clean_text(text: str, insert_diagnostics: bool = False, processor=None) -> str:
    """
    Clean clinical note text by removing obvious administrative fragments using
    regex/rule-based line filtering.  Every line is evaluated independently —
    there is no clinical-keyword gate — so clinical content is never silently
    dropped because it happened to appear before a recognised section header.

    Lines are removed only when they positively match an admin pattern.  When
    in doubt a line is kept (conservative / safe).

    If insert_diagnostics is True, extract active diagnoses via the processor
    and append them under a 'Diagnostics' header.
    """
    # --- Input normalisation ---
    if not isinstance(text, str):
        if isinstance(text, float) and pd.isna(text):
            text = ""
        else:
            text = str(text) if text is not None else ""

    # ------------------------------------------------------------------
    # Regex/rule-based removal of obvious admin fragments.
    # Each pattern targets a specific class of admin line.
    # Patterns are anchored or narrowly scoped to avoid false positives.
    # ------------------------------------------------------------------
    _ADMIN_LINE_PATTERNS = [
        # Standalone page numbers / separators  (e.g. "Page 3", "3 / 5", "---")
        r'^\s*(page\s*\d+|\d+\s*/\s*\d+|[-=_*]{3,})\s*$',
        # Lines that are purely a bare number (not part of a sentence)
        r'^\s*\d+\s*$',
        # Staff title lines  (Dr, Docteur, Pr, Professeur, Praticien …)
        r'^\s*(dr\.?\s|docteur\s|pr\.?\s|professeur\s|praticien\s+(hospitalier|contractuel|attach[eé])|attach[eé]\s+associ[eé]|chef\s+(de\s+service|du\s+p[oô]le))\b',
        # Facility / ward headers  (Service de …, Unité de …, EHPAD, …)
        r'^\s*(service\s+de\b|unit[eé]\s+(de|du|d[e\'])|centre\s+(hospitalier|m[eé]dical)|h[oô]pital\b|clinique\b|ehpad\b|r[eé]sidence\b|[eé]quipe\s+mobile\b|soins\s+(de\s+(longue\s+dur[eé]e|suite)|longue\s+dur[eé]e)\b)',
        # Street address lines  (12 rue …, 75000 Paris …)
        r'^\s*\d{1,4}\s+(rue|avenue|bd|boulevard|impasse|all[eé]e|place|chemin|route|lotissement|r[eé]sidence|appartement|appt|b[aâ]timent|bt)\b',
        r'^\s*\d{5}\s+[A-ZÀ-Ü]',
        # Phone / fax / contact lines
        r'^\s*(t[eé]l\.?|t[eé]l[eé]phone|fax|ligne\s+directe|poste\s+\d|secr[eé]tariat|accueil|standardiste?)\s*[:\-–]',
        r'^\s*(t[eé]l\.?|t[eé]l[eé]phone|fax|ligne\s+directe)\s*[:\-–]?\s*[\d\s\.\(\)\-\/]+$',
        # Web / API references
        r'^\s*(www\.|https?://|api[-\s]?\d*\b)',
        # Administrative identifiers (IPP, N° dossier, N° séjour …)
        r'\b(ipp\s*[:\-–=]|n[°o]\s*(d[\'\']archives?|dossier|s[eé]jour)|identifiant\s+patient)\b',
        # Closing salutations / signatures
        r'\b(cordialement|confraternellement|bien\s+cordialement|en\s+restant\s+[àa]\s+votre\s+disposition|cher\s+confr[eè]re)\b',
        # Purely administrative phrases
        r'\b(dict[eé]\s+le|document\s+[àa]\s+envoyer|prise\s+de\s+rendez[-\s]vous\s+au|consultation\s+avanc[eé]e)\b',
        # Date-only lines (no surrounding clinical text)
        r'^\s*(le\s+)?\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\s*$',
    ]
    _compiled_admin = [re.compile(p, re.IGNORECASE) for p in _ADMIN_LINE_PATTERNS]

    lines = text.splitlines()
    cleaned_lines: list[str] = []
    seen: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Deduplicate identical lines
        key = stripped.lower()
        if key in seen:
            continue

        # Drop lines that are ALL-CAPS and longer than 20 chars (banner headers)
        if len(stripped) > 20 and stripped == stripped.upper() and re.search(r'[A-ZÀ-Ü]{3}', stripped):
            continue

        # Drop lines where >70 % of letters are upper-case (shouted headers)
        letters = re.sub(r'[^a-zA-ZÀ-ÿ]', '', stripped)
        if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7:
            continue

        # Drop lines matching an admin pattern
        if any(pat.search(stripped) for pat in _compiled_admin):
            continue

        seen.add(key)
        cleaned_lines.append(stripped)

    cleaned_text = ' '.join(cleaned_lines)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # --- Optional diagnostics insertion ---
    if insert_diagnostics and processor and hasattr(processor, 'extract_active_diagnoses_llm'):
        diagnoses = processor.extract_active_diagnoses_llm(cleaned_text)
        if diagnoses:
            cleaned_text += '\nDiagnostics :\n' + '\n'.join(f'- {d}' for d in diagnoses) + '\n'

    return cleaned_text


def extract_z_codes_hybrid(text):
    """
    Rule-based Z code extraction with descriptions. Returns a list of dicts: [{"code": Zxx, "description": ...}, ...]
    """
    z_code_descriptions = {
        "Z87": "antécédents personnels de maladies non cancéreuses",
        "Z22": "porteurs d’infection / surveillance",
        "Z50": "rééducation / réadaptation",
        "Z60": "isolement social ou environnement difficile",
        "Z73": "stress, surcharge, adaptation difficile",
        "Z74": "dépendance fonctionnelle (aide à la toilette, mobilité)",
        "Z75": "difficultés d’accès aux soins",
        "Z91": "non-observance thérapeutique / allergies",
        "Z93": "dispositifs artificiels (stomie, sonde)",
        "Z99": "dépendance à un dispositif"
    }
    z_codes_rule = set(extract_icd_codes_from_rules(text))
    results = []
    for code in z_codes_rule:
        description = z_code_descriptions.get(code, "")
        results.append({"code": code, "problem": description})
    return results


def extract_icd_codes_from_rules(input_data) -> list:
    """
    Detect ICD-10 Z-codes (Z22, Z50, Z60, Z74, Z75, Z91, Z93, Z99, Z87) using rule-based patterns, considering negation:
      - text negation ("pas", "sans", "autonome", etc.)
      - medkit entity attribute 'is_negated' = True
    Supports input as either:
      - raw text (str)
      - list of medkit Entities or Segments
    """
    if input_data is None:
        return []

    # --- Handle Medkit entities or text ---
    if isinstance(input_data, str):
        text = input_data.lower()
        entities = []
    else:
        # Likely list of Entities or Segments
        from medkit.core.text import Segment, Entity
        if isinstance(input_data, (list, tuple)):
            entities = [e for e in input_data if hasattr(e, "text")]
            text = " ".join([e.text.lower() for e in entities])
        elif hasattr(input_data, "text"):
            text = input_data.text.lower()
            entities = [input_data]
        else:
            text = str(input_data).lower()
            entities = []

    detected_codes = set()

    # --- Negation keywords for context-based detection ---
    negation_keywords = [
        r"\bpas\b", r"\baucun[e]?\b", r"\bne\b", r"\bsans\b", r"\bni\b",
        r"\bautonome\b", r"\bindépendant[e]?\b",
        r"\bne nécessite pas\b", r"\bpas besoin\b",
        r"\bpas d[' ]?aide\b", r"\bpas de soins\b",
    ]

    def text_has_negation(context):
        """Return True if a generic negation word appears in context."""
        return any(re.search(neg, context) for neg in negation_keywords)

    z74_negation_keywords = [
        r"\bpas\s+d[' ]?aide\b",
        r"\bpas\s+besoin\s+d[' ]?aide\b",
        r"\bne\s+n[ée]cessite\s+pas\s+d[' ]?aide\b",
        r"\bautonome\b",
        r"\bind[ée]pendant[e]?\b",
        r"\bmarche\s+sans\s+aide\b",
    ]

    def text_has_negation_for_code(context, code):
        """Use stricter negation for Z74 to avoid false negatives from unrelated 'pas'."""
        if code == "Z74":
            return any(re.search(neg, context) for neg in z74_negation_keywords)
        return text_has_negation(context)

    def is_negated_entity(ent_text):
        """Check if a text or entity is negated either by tag or context."""
        # Check Medkit entity attribute
        if hasattr(ent_text, "attrs"):
            for attr in getattr(ent_text, "attrs", []):
                if getattr(attr, "label", "") == "is_negated" and bool(getattr(attr, "value", False)):
                    return True
        # Fallback to textual detection
        window = ent_text.text.lower() if hasattr(ent_text, "text") else ent_text.lower()
        return text_has_negation(window)

    # Matches Zxx or Zxx.x (e.g., Z22, Z22.1, Z99.0) -- only add if not negated in local context
    for m in re.finditer(r"\bZ\d{2}(?:\.\d+)?\b", text, flags=re.IGNORECASE):
        code = m.group()[:3].upper()
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 30)
        window = text[start:end]
        if not text_has_negation(window):
            detected_codes.add(code)
        
    # --- ICD code rules ---
    code_patterns = {
  
        "Z22": [
            # Generic carrier mentions
            r"\bcolonis[ée]\b",
            r"\bcolonisation\b",
            r"\bsujet porteur\b",
            r"\bporteur asymptomatique\b",
            r"\bporteur de germes\b",
            r"\bporteur de (?:virus|bact[ée]rie|germe)\b",
            r"\bporteur de (?:staphylocoque|streptocoque|clostridium|salmonella|h[ée]patite|mrsa|blse)\b",

            # Common virus carrier mentions
            r"\bporteur de l'antigène\b",
            r"\bhbsag\b",
            r"\bhtlv[- ]?1\b",

            # Specific bacterial carrier states
            r"\bporteur de dipht[ée]rie\b",
            r"\bporteur de typho[iï]de\b",

            # Resistant organism carrier / colonization wording
            r"\bbmr\b.{0,80}\bblse\b"
            r"\be\.?\s*coli\b.{0,40}\bblse\b"
            r"\batcd\b.{0,20}\bsarm\b"
            r"\brisque sanitaire bmr\b"
            r"\bbhr\b",
            r"\bsarm\b",
            r"\bm(?:r)?sa\b",
            r"\brisque sanitaire bhr\b",
            r"\bblse rectal\b",
            r"\be\.?\s*coli\s+blse\b",
            r"\bklebsiella\s+blse\b",
            r"\batcd\s+sarm\b",
            r"\batcd\s+blse\b",
        ],
        "Z50": [
            # 1️⃣ Explicit SSR / rehab context
            r"\b(?:suivi\s+en\s+ssr|en\s+(?:suivi|suive)\s+ssr)\b",
            r"\bhospitalis[ée]?\s+en\s+ssr\b",
            r"\bentr[éeé]\s+en\s+ssr\b",
            r"\bretour\s+en\s+ssr\b",
            r"\bssr\s+(?:de|pour)\s+r[ée]adaptation\b",
            r"\bssr\s+(?:locomoteur|orthop[ée]dique|neurologique)\b",
            r"\br[ée]adaptation\s+en\s+ssr\b",
            r"\bsuivi\s+en\s+ssr\b",
            r"\ben\s+(?:suivi|suive)\s+ssr\b",
            r"\bretour\s+en\s+ssr\b",
            r"\bhospitalis[ée]?\s+en\s+ssr\b",
            r"\bpris[ée]?\s+en\s+charge\s+en\s+ssr\b",

            # 2️⃣ Structured functional / physical rehabilitation
            r"\bsoins\s+de\s+suite\s+et\s+de\s+réadaptation\b",
            r"\br[ée]adaptation\b",
            r"\br[ée]adapt[ée]?\s+fonctionnelle\b",
            r"\br[ée]adaptation\s+(?:orthop[ée]dique|cardiaque|neurologique|respiratoire|proth[ée]tique)\b",
            r"\br[ée]adaptation\s+(?:motrice|locomotrice|fonctionnelle)\b",
            r"\br[ée]adaptation\s+(?:cognitive|neuropsychologique)\b",
            r"\br[ée]adaptation\s+(?:post[-\s]?AVC|après\s+AVC)\b",
            r"\br[ée]adaptation\s+du\s+(?:marche|membre|rachis)\b",
            r"\br[ée][ée]ducation\s+(?:fonctionnelle|respiratoire|orthop[ée]dique|cardiaque|locomotrice)\b",
            r"\br[ée][ée]ducation\s+(?:physique|motrice|neuro|respiratoire|orthophonique)\b",
            r"\bdemande\s+(?:de\s+)?r[ée][ée]ducation(?:\s+et\s+appareillage)?\b",
            r"\br[ée]-?autonomisation\b",
            r"\b(?:r[ée]-?autonomisation|autonomisation)\s+(?:aux?|à)\s+d[ée]placements?\b",
            # Catch French past participles and verb forms of rééducation (e.g., "Rééducate verticalisée", "rééducate mobilisation")
            r"\br[ée][ée]ducat[ée]?\b",
            
            # --- 2️⃣ Implicit reeducation / rehabilitation contexts ---
            r"\bkin[ée]\b.*\b(ssr|réadaptation|rééducation|réentraînement)\b",
            r"\bsuivi\s+(?:de\s+)?kin[ée]sith[ée]rapie.*\b(ssr|réadaptation|rééducation|réentraînement)\b",
            r"\bs[ée]ance[s]?\s+de\s+kin[ée].*\b(ssr|réadaptation|rééducation|réentraînement)\b",
            r"\bprise\s+en\s+charge\s+kin[ée].*\b(ssr|réadaptation|rééducation|réentraînement)\b",
            r"\br[ée]adaptation\s+en\s+ssr\b",
            r"\br[ée]adaptation\s+apr[èe]s\s+(?:fracture|AVC|chirurgie|traumatisme)\b",
            r"\br[ée]adaptation\s+post[-\s]op[ée]ratoire\b",
            r"\br[ée]adaptation\s+post\s+fracture\b",
            r"\br[ée][ée]ducation\s+apr[èe]s\s+(?:fracture|chirurgie|AVC)\b",
            r"\br[ée]adaptation\s+orth[ée]tique\b",

            # --- 3️⃣ Rehabilitation services or staff ---
            r"\br[ée][ée]ducateur\s+fonctionnel\b",
            r"\br[ée][ée]ducateur\s+sportif\b",
            r"\br[ée][ée]ducateur\s+en\s+r[ée]adaptation\b",
            r"\br[ée][ée]ducateur\s+physique\b",
            r"\br[ée]adaptateur\b",
            r"\bservice\s+de\s+r[ée]adaptation\b",
            r"\bunit[ée]\s+de\s+r[ée]adaptation\b",
            r"\bcentre\s+de\s+r[ée]adaptation\b",

            # --- 4️⃣ Functional recovery / rehabilitation objectives ---
            r"\br[ée]cup[ée]ration\s+fonctionnelle\b",
            r"\br[ée]entraînement\s+(?:cardiaque|à\s+l['’]effort|respiratoire)\b",
            r"\br[ée]adaptation\s+à\s+l['’]effort\b",
            r"\br[ée]adaptation\s+à\s+la\s+marche\b",
            r"\br[ée]entrainement\s+de\s+la\s+marche\b",
            r"\bprise\s+en\s+charge\s+en\s+r[ée]adaptation\b",
            r"\bsuivi\s+en\s+r[ée]adaptation\b",
            r"\bs[ée]jour\s+de\s+r[ée]adaptation\b",

            # --- 5️⃣ Speech, neuro, occupational rehab ---
            r"\br[ée][ée]ducation\s+orthophonique\b",
            r"\btype\s+de\s+s[ée]ance\s+orthophoniste\b",
            r"\bs[ée]ance\b.{0,30}\borthophon(?:iste|ique)\b",
            r"\br[ée][ée]ducation\s+neuropsychologique\b",
            r"\br[ée][ée]ducation\s+cognitive\b",
            r"\br[ée]adaptation\s+neuropsychologique\b",
            r"\br[ée]adaptation\s+ergoth[ée]rapique\b",
            r"\bergoth[ée]rapie\b",
            r"\br[ée]adaptation\s+psychomotrice\b",
        ],
        "Z60": [
            # Social isolation / lack of support
            r"\bisolement\s+(?:social|familial|affectif)\b",
            r"\bsituation\s+d['’]isolement\b",
            r"\babsence\s+d['’]entourage\b",
            r"\baucun\s+soutien\b",
            r"\bsolitude\b",
            r"\brompu[e]?\s+de\s+ses\s+liens\s+sociaux\b",
            r"\bprobl[èe]mes?\s+sociaux?\b",

            # Family conflict / abuse / neglect
            r"\bmaltraitance\b",
            r"\babus\b",
            r"\bviolence[s]?\s+(?:physiques?|psychologiques?)\b",
            r"\benferme[ée]?\s+(?:à|a)\s+clef\b",
            r"\bsous\s+l['’]emprise\b",
            r"\bexploitation\s+financi[èe]re\b",
            r"\bconflit\s+familial\b",
            r"\brelations?\s+familiales?\s+difficiles?\b",
            r"\bfamille\s+conflictuelle\b",
            r"\bnégligence\b",
            r"\bprobl[èe]me\s+familial\b",

            # Social / financial precarity
            r"\bpr[ée]carit[ée]\b",
            r"\bdifficult[ée]s?\s+sociales?\b",
            r"\bdifficult[ée]s?\s+(?:financi[èe]res?|matérielles?)\b",
            r"\bressource[s]?\s+(?:limitées|insuffisantes)\b",
            r"\bfaibles?\s+revenus?\b",
            r"\bsituation\s+(?:financi[èe]re|sociale)\s+précaire\b",
            r"\bexclusion\s+sociale\b",
            r"\bsdf\b",
            r"\bsans\s+domicile\b",
            r"\bsans\s+abri\b",

            # Literacy / language barriers
            r"\banalphab[èe]te\b",
            r"\bill[ée]tr[ée]isme\b",
            r"\bne\s+sait\s+(ni\s+)?lire\b",
            r"\bne\s+sait\s+(ni\s+)?écrire\b",
            r"\bbarri[èe]re\s+linguistique\b",
            r"\bparle\s+peu\s+français\b",

            # Cultural / social integration
            r"\benvironnement\s+(?:social|familial|culturel)\s+difficile\b",
            r"\bsituation\s+culturelle\s+complexe\b",
            r"\bdiscrimination\b",
            r"\brejet\s+social\b",
            r"\bisolement\s+culturel\b",
            r"\bprobl[èe]mes?\s+d['’]int[ée]gration\b",
        ],

        "Z74": [
            # Dépendance générale
            r"\behpad\b",
            r"\bd[ée]pendance\s+fonctionnelle\b",
            r"\bd[ée]pendant[e]?\b",
            r"\bd[ée]pendance\s+(?:partielle|totale)\b",
            r"\bperte\s+d'autonomie\b",
            r"\bperte\s+fonctionnelle\b",
            #r"\bvit\s+seul[ea]?\s+à\s+domicile\b",
            r"\bvit\s+seul[ea]?\s+en\s+foyer\b",
            r"\bbesoin\s+d'assistance\b",
            r"\bassist[ée]?\s+pour\s+les\s+actes\s+de\s+la\s+vie\b",
            r"\baide[s]?\s+(?:humaine[s]?|quotidienne[s]?|toilette|à\s+domicile)\b",
            r"\baide\s+à\s+la\s+toilette\b",
            r"\baide\s+m[ée]nag[èe]re\b",
            r"\bide\s+(?:matin|soir|quotidienne?|r[ée]guli[èe]re)\b",
            r"\bide\s*\d+\s*/\s*(?:j|jour)\b",
            r"\bauxiliaire\s+(?:de\s+vie\s+)?\d*h?\s*/\s*(?:j|jour)\b",
            r"\bauxiliaire\b.{0,60}\b(?:m[ée]nage|toilette|repas|pr[ée]paration\s+des\s+repas)\b",
            r"\bpassage\s+ide\b",
            r"\baides?\s+formelles?\b",
            r"\baides?\s+logistiques?\b",
            r"\baides?\s+à\s+domicile\b",
            r"\bmaximum\s+d['']aides?\b",
            r"\bn[''']est\s+(?:plus\s+)?(?:du\s+tout\s+)?autonome\b",
            r"\bplus\s+(?:du\s+tout\s+)?autonome\b",
            r"\bperd\s+(?:son\s+)?autonomie\b",
            r"\bsoins\s+(?:infirmiers|quotidiens)\s+(?:r[ée]guliers|3x/j|trois\s+fois\s+par\s+jour)\b",
            r"\badl\s*[:=]?\s*\(?\s*[0-5]\s*/\s*6\s*\)?\b",
            r"\biadl\s*[:=]?\s*\(?\s*[0-7]\s*/\s*8\s*\)?\b",

            # Mobilité réduite / dépendance physique
            r"\bmobilit[ée]\s+restreinte\b",
            r"\bne\s+marche\s+pas\b",
            r"\bnon\s+ambulatoire\b",
            r"\bmarche\s+avec\s+(?:canne|d[ée]ambulateur|b[ée]quilles|aide[\s-]marche)\b",
            r"\bavec\s+(?:d[ée]ambulateur|canne|b[ée]quilles|aide[\s-]marche)\b",
            r"\butilise\s+(?:un|une)\s+(?:fauteuil\s+roulant|d[ée]ambulateur)\b",
            r"\bd[ée]ambulateur\b",
            r"\bau\s+fauteuil\b",
            r"\ben\s+fauteuil\s+roulant\b",
            r"\blev[ée]\s+(?:avec\s+)?(?:aide|assistance|d[ée]ambulateur)\b",
            r"\brecouch[ée]\s+(?:en\s+(?:le\s+)?)?verticalisant\b",
            r"\bverticalisation\s+(?:avec\s+)?(?:aide|assistance)\b",
            r"\bne\s+se\s+l[èe]ve\s+pas\b",
            r"\bne\s+se\s+d[ée]place\s+pas\b",
            r"\btoilette\s+au\s+lit\b",           
            
            # Handicap / paraplégie / hémiplégie
            r"\bhandicap[ée]?\b",
            r"\bhandicap\s+physique\b",
            r"\bh[ée]mipl[ée]gie\b",
            r"\bh[ée]mipl[ée]gique\b",
            r"\bt[ée]trapl[ée]gie\b",            
            r"\bt[ée]trapl[ée]gique\b",
            r"\bparapl[ée]gique\b",
            r"\bparapl[ée]gie\b",
            r"\bmonopl[ée]gie\b",
            r"\bd[ée]ficit\s+moteur\s+(?:majeur|s[ée]v[èe]re)\b",
            r"\binfirmit[ée]\s+motrice\b",
            r"\btrouble\s+de\s+la\s+marche\b",
            r"\bpatient\s+(?:grabataire|non\s+ambulatoire)\b",

            # Vie à domicile difficile
            r"\bmaintien\s+à\s+domicile\s+difficile\b",
            r"\bmaintien\s+à\s+domicile\b.{0,60}?\bdifficile\b",
            r"\bdomicile\s+difficile\b",
        ],

        "Z75": [
            r"\bdifficulté d'accès aux soins\b",
            r"\babsence de médecin traitant\b",
            r"\bpas de médecin traitant\b",
            r"\bretard de prise en charge\b",
            r"\bdésert médical\b",
        ],
        "Z91": [
            r"\bantécédents personnels\b.*(allergie|abus|non[- ]observance|mauvaise hygiène|risque|suicide|parasuicide|intoxication)",
            r"\brefus de traitement\b",
            r"\bnon compliance\b",
            r"\boubli de traitement\b",
            r"\bfacteur de risque\b",
        ],
        "Z93": [
            r"\bstomie\b",
            r"\bbricker\b",
            r"\bcolostomie\b",
            r"\biléostomie\b",
            r"\bgastrostomie\b",
            r"\btrachéostomie\b",
            r"\bcystostomie\b",
            r"\bnéphrostomie\b",
            r"\burétérostomie\b",
            r"\burétrostomie\b",
            r"\bdérivation urinaire\b",
            r"\bappareillage de néphrostomie\b",
        ],
        "Z99": [
            r"\boxyg[ée]no[-\s]?d[ée]pendant[e]?\b",
            r"\bd[ée]pendant[e]?\s+de\s+l['’]oxyg[èe]ne\b",
            r"\bsous\s+oxyg[èe]ne\s+(?:à\s+domicile|au\s+long\s+cours)\b",

            r"\bd[ée]pendan[tce]\b.*\b(respirateur|ventilation|oxyg[èe]ne|dialyse|nutrition)\b",
            r"\bventilation\s+(?:assist[ée]e|m[ée]canique|non\s+invasive)\s+à\s+domicile\b",
            r"\bbranch[ée]?\s+au\s+respirateur\s+à\s+domicile\b",

            r"\ben\s+(?:h[ée]modialyse|dialyse)\s+chronique\b",
            r"\bd[ée]pendant[e]?\s+de\s+la\s+dialyse\b",
            r"\bdialys[ée]?\s+chronique\b",

            r"\bnutrition\s+(?:ent[ée]rale|parent[ée]rale)\s+(?:chronique|à\s+domicile|au\s+long\s+cours)\b",
            r"\bNPT\s+(?:chronique|à\s+domicile|au\s+long\s+cours)\b",
        ],
    }

    # --- Apply rules ---
    for code, patterns in code_patterns.items():
        found = False
        # Special handling for Z50: only add for strong main-care context, not for weak/ambiguous mentions
        if code == "Z50":
            # Only add Z50 if strong main-care context is present
            strong_patterns = [
                # SSR/rehab main context
                r"\b(?:suivi\s+en\s+ssr|en\s+(?:suivi|suive)\s+ssr)\b",
                r"\bhospitalis[ée]?\s+en\s+ssr\b",
                r"\bentr[éeé]\s+en\s+ssr\b",
                r"\bretour\s+en\s+ssr\b",
                r"\bssr\s+(?:de|pour)\s+r[ée]adaptation\b",
                r"\bssr\s+(?:locomoteur|orthop[ée]dique|neurologique)\b",
                r"\br[ée]adaptation\s+en\s+ssr\b",
                r"\bsuivi\s+en\s+ssr\b",
                r"\ben\s+(?:suivi|suive)\s+ssr\b",
                r"\bretour\s+en\s+ssr\b",
                r"\bhospitalis[ée]?\s+en\s+ssr\b",
                r"\bpris[ée]?\s+en\s+charge\s+en\s+ssr\b",
                # Structured functional/physical rehabilitation
                r"\bsoins\s+de\s+suite\s+et\s+de\s+réadaptation\b",
                r"\br[ée]adaptation\b",
                r"\br[ée]adapt[ée]?\s+fonctionnelle\b",
                r"\br[ée]adaptation\s+(?:orthop[ée]dique|cardiaque|neurologique|respiratoire|proth[ée]tique)\b",
                r"\br[ée]adaptation\s+(?:motrice|locomotrice|fonctionnelle)\b",
                r"\br[ée]adaptation\s+(?:cognitive|neuropsychologique)\b",
                r"\br[ée]adaptation\s+(?:post[-\s]?AVC|après\s+AVC)\b",
                r"\br[ée]adaptation\s+du\s+(?:marche|membre|rachis)\b",
                r"\br[ée][ée]ducation\s+(?:fonctionnelle|respiratoire|orthop[ée]dique|cardiaque|locomotrice)\b",
                r"\br[ée][ée]ducation\s+(?:physique|motrice|neuro|respiratoire|orthophonique)\b",
                # r"\bdemande\s+(?:de\s+)?r[ée][ée]ducation(?:\s+et\s+appareillage)?\b",  # Removed: prescription only, not main-care context
                r"\br[ée]-?autonomisation\b",
                r"\b(?:r[ée]-?autonomisation|autonomisation)\s+(?:aux?|à)\s+d[ée]placements?\b",
                r"\br[ée][ée]ducat[ée]?\b",
                # Rehabilitation services or staff
                r"\br[ée][ée]ducateur\s+fonctionnel\b",
                r"\br[ée][ée]ducateur\s+sportif\b",
                r"\br[ée][ée]ducateur\s+en\s+r[ée]adaptation\b",
                r"\br[ée][ée]ducateur\s+physique\b",
                r"\br[ée]adaptateur\b",
                r"\bservice\s+de\s+r[ée]adaptation\b",
                r"\bunit[ée]\s+de\s+r[ée]adaptation\b",
                r"\bcentre\s+de\s+r[ée]adaptation\b",
                # Functional recovery / rehabilitation objectives
                r"\br[ée]cup[ée]ration\s+fonctionnelle\b",
                r"\br[ée]entraînement\s+(?:cardiaque|à\s+l['’]effort|respiratoire)\b",
                r"\br[ée]adaptation\s+à\s+l['’]effort\b",
                r"\br[ée]adaptation\s+à\s+la\s+marche\b",
                r"\br[ée]entrainement\s+de\s+la\s+marche\b",
                r"\bprise\s+en\s+charge\s+en\s+r[ée]adaptation\b",
                r"\bsuivi\s+en\s+r[ée]adaptation\b",
                r"\bs[ée]jour\s+de\s+r[ée]adaptation\b",
            ]
            for p in strong_patterns:
                for m in re.finditer(p, text):
                    start = max(0, m.start() - 60)
                    end = min(len(text), m.end() + 30)
                    window = text[start:end]
                    if not text_has_negation_for_code(window, code):
                        detected_codes.add(code)
                        found = True
                        break
                if found:
                    break
            # Entities: only add Z50 if entity context is strong
            if not found and entities:
                for ent in entities:
                    if not is_negated_entity(ent):
                        ent_text = ent.text.lower() if hasattr(ent, "text") else str(ent).lower()
                        for p in strong_patterns:
                            if re.search(p, ent_text):
                                detected_codes.add(code)
                                found = True
                                break
                        if found:
                            break
            continue  # skip default pattern logic for Z50
        # Default logic for other codes
        for p in patterns:
            for m in re.finditer(p, text):
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 30)
                window = text[start:end]
                if not text_has_negation_for_code(window, code):
                    detected_codes.add(code)
                    found = True
                    break
            if found:
                break
        if entities:
            for ent in entities:
                if not is_negated_entity(ent):
                    ent_text = ent.text.lower() if hasattr(ent, "text") else str(ent).lower()
                    for p in patterns:
                        if re.search(p, ent_text):
                            detected_codes.add(code)
                            break
    # --- Z87: Antécédents personnels d'autres maladies ---
    # Only for medkit_structuring output (dict with section_entities)
    if isinstance(input_data, dict) and 'section_entities' in input_data:
        antecedents = input_data['section_entities'].get('antecedent', [])
        diagnostics = input_data['section_entities'].get('diagnostics', [])
        # Collect disorder texts from both sections
        antecedent_disorders = set(ent['text'].lower() for ent in antecedents if not ent.get('negated'))
        diagnostic_disorders = set(ent['text'].lower() for ent in diagnostics if not ent.get('negated'))
        # Exclude if disorder is cancer or drug allergy
        cancer_patterns = [r"cancer", r"carcinome", r"sarcome", r"lymphome", r"leucémie", r"tumeur maligne", r"néoplasme malin", r"melanome", r"myélome"]
        allergy_patterns = [r"allergie", r"allergic", r"hypersensibilité", r"réaction médicamenteuse", r"allergie médicamenteuse", r"allergie à"]
        for disorder in antecedent_disorders:
            if disorder in diagnostic_disorders:
                continue
            if any(re.search(p, disorder) for p in cancer_patterns):
                continue
            if any(re.search(p, disorder) for p in allergy_patterns):
                continue
            detected_codes.add("Z87")
            break  # Only code Z87 once if any valid antecedent
    return detected_codes

NUMERIC_VALUE_RULES = [
    {
        "rulename": "Glasgow Coma Score",
        "regex": r"\b(?:GCS|Glasgow|G)(?:\s*(?:score)?\s*[:=]?\s*)([0-9]{1,2})\b",
        "critical_value": 14,
        "type_of_inequality": "<=",
        "diagnosis": "R40",
        "label": "Perte de conscience / Obnubilation",
        "value_type": "int",
    },
    {
        "rulename": "Hypernatremia, with known value",
        "regex": r"\bNa(?:tr[ée]mie)?(?:[+àa :>=]+?|de\s+)*([0-9]{3})\b",
        "critical_value": 145,
        "type_of_inequality": ">",
        "diagnosis": "E87",
        "label": "Hypernatrémie",
        "value_type": "int",
    },
    {
        "rulename": "Hyponatremia, with known value",
        "regex": r"\bNa(?:tr[ée]mie)?(?:[+àa :<=]+?|de\s+)*([0-9]{3})\b",
        "critical_value": 135,
        "type_of_inequality": "<",
        "diagnosis": "E87",
        "label": "Hyponatrémie",
        "value_type": "int",
    },
    {
        "rulename": "Hypokalemia, with known value",
        "regex": r"\bK(?:ali[ée]mie)?(?:[+àa :<=]+?|de\s+)+([0-9]{1,2}(?:[.,][0-9])?)\b",
        "critical_value": 3.5,
        "type_of_inequality": "<",
        "diagnosis": "E87",
        "label": "Hypokaliémie",
        "value_type": "float",
    },
    {
        "rulename": "Hyperkalemia, with known value",
        "regex": r"\bK(?:ali[ée]mie)?(?:[+àa :>=]+?|de\s+)*([0-9]{1,2}(?:[.,][0-9])?)\b",
        "critical_value": 4.5,
        "type_of_inequality": ">",
        "diagnosis": "E87",
        "label": "Hyperkaliémie",
        "value_type": "float",
    },
    {
        "rulename": "Anemia, known Hb",
        "regex": r"\b(?:hb|h[ée]moglobin[ée]mie|h[ée]moglobine)\b\s*(?:[:=]|est\s+de\s+|de\s+|à\s+)?\s*([0-9]{1,2}(?:[.,][0-9])?)\b",
        "critical_value": 12,
        "type_of_inequality": "<",
        "diagnosis": "D64",
        "label": "Anémie",
        "value_type": "float",
    },
    {
        "rulename": "Insuffisance rénale (créatinine)",
        "regex": r"\b(?:cr[ée]at(?:inine|inin[ée]mie)?|creatinine)\b(?:\s+(?:s[ée]rique|plasmatique|sanguine))?\s*(?:[:=]|est\s+de\s+|de\s+|à\s+|[<>]=?)?\s*([0-9]{3,4}(?:[.,][0-9]{1,2})?)\s*(?:µ?mol\s*/?\s*l|umol\s*/?\s*l)?\b",
        "critical_value": 120,
        "type_of_inequality": ">",
        "diagnosis": "N18",
        "label": "Insuffisance rénale",
        "value_type": "float",
        "resolver": "renal_failure",
        "marker": "creatinine",
    },
    {
        "rulename": "Rétention azotée (urée)",
        "regex": r"\b(?:ur[ée]e|urea)\b(?:\s+(?:s[ée]rique|plasmatique|sanguine))?\s*(?:[:=]|est\s+de\s+|de\s+|à\s+|[<>]=?)?\s*([0-9]{1,3}(?:[.,][0-9]{1,2})?)\s*(?:mmol\s*/?\s*l|g\s*/?\s*l|mg\s*/?\s*l)?\b",
        "critical_value": 7.5,
        "type_of_inequality": ">",
        "diagnosis": "N18",
        "label": "Rétention azotée",
        "value_type": "float",
        "resolver": "renal_failure",
        "marker": "urea",
    },
    {
        "rulename": "DFG bas",
        "regex": r"\b(?:dfg|egfr|clairance(?:\s+de\s+la\s+cr[ée]atinine)?|filtration\s+glom[ée]rulaire(?:\s+estim[ée]e?)?)\b\s*(?:[:=]|est\s+de\s+|de\s+|à\s+|[<>]=?)?\s*([0-9]{1,3}(?:[.,][0-9]{1,2})?)\s*(?:ml\s*/?\s*min(?:\s*/\s*1[.,]73\s*m2)?)?\b",
        "critical_value": 15,
        "type_of_inequality": "<",
        "diagnosis": "N18",
        "label": "DFG bas",
        "value_type": "float",
        "resolver": "renal_failure",
        "marker": "gfr",
    },
    {
        "rulename": "Rétention urinaire (bladder scan / RPM)",
        "regex": r"\b(?:bladder(?:\s+scan)?|rpm|r[ée]sidu\s+post[- ]?mictionnel)\b[^0-9]{0,30}([0-9]{3,4})\s*(?:cc|ml)\b",
        "critical_value": 300,
        "type_of_inequality": ">",
        "diagnosis": "R33",
        "label": "Rétention urinaire",
        "value_type": "float",
    },
    {
        "rulename": "Vit D deficiency, with known value",
        "regex": r"\bVit(?:amin)?e?\s*D(?:[+àa :<=]+?|de\s+)*([0-9]{1,3}(?:[.,][0-9])?)\b",
        "critical_value": 30,
        "type_of_inequality": "<",
        "diagnosis": "E55",
        "label": "Carence en vitamine D",
        "value_type": "float",
    },
    {
        "rulename": "Vit B9 deficiency, ng",
        "regex": r"\b(?:Vit(?:amin)?e?\s*B9|folate[s]?)(?:[+àa :<=]+?|de\s+)*([0-9]{1,3}(?:[.,][0-9])?)\b\s*(?:ng)?",
        "critical_value": 3,
        "type_of_inequality": "<",
        "diagnosis": "E53",
        "label": "Carence en vitamine B9",
        "value_type": "float",
    },
    {
        "rulename": "Vit B9 deficiency, nmol",
        "regex": r"\b(?:Vit(?:amin)?e?\s*B9|folate[s]?)(?:[+àa :<=]+?|de\s+)*([0-9]{1,3}(?:[.,][0-9])?)\b\s*nmol",
        "critical_value": 6.8,
        "type_of_inequality": "<",
        "diagnosis": "E53",
        "label": "Carence en vitamine B9",
        "value_type": "float",
    },
    {
        "rulename": "Hypoxemie par saturation basse",
        "regex": r"\b(?:sat|sao2|spo2|saturation)\b\s*(?:[:=]|a|à|de)?\s*([0-9]{1,3}(?:[.,][0-9]+)?)\s*%?\b",
        "critical_value": 90,
        "type_of_inequality": "<=",
        "diagnosis": "J96",
        "label": "Insuffisance respiratoire",
        "value_type": "float",
    },
    {
        "rulename": "FiO2 élevée (>=60%)",
        "regex": r"(?:\w+\s+)*(fio2|fi o2|fraction inspirée d'oxygène|fraction d'oxygène inspirée)\s*(?:[:=]|a|à|de)?\s*([0-9]{1,3}(?:[.,][0-9]+)?)\s*%?",
        "critical_value": 60,
        "type_of_inequality": ">=",
        "diagnosis": "J96",
        "label": "Insuffisance respiratoire (FiO2)",
        "value_type": "float",
    },
    {
        "rulename": "Hypoxemie par PaO2 basse",
        "regex": r"\b(?:pao2|po2)\b\s*(?:[:=]|a|à|de)?\s*([0-9]{1,3}(?:[.,][0-9]+)?)\b",
        "critical_value": 60,
        "type_of_inequality": "<=",
        "diagnosis": "J96",
        "label": "Insuffisance respiratoire",
        "value_type": "float",
    },
    {
        "rulename": "Tachycardie (FC élevée)",
        "regex": r"\b(?:fc|fr[ée]quence\s+cardiaque|pouls|fc\s+[àa])\b\s*(?:[:=àa]|de\s+|[àa]\s+)?\s*([0-9]{2,3})\s*(?:/\s*min|bpm|bat(?:tements)?(?:/min)?)?\b",
        "critical_value": 100,
        "type_of_inequality": ">",
        "diagnosis": "R00",
        "label": "Tachycardie",
        "value_type": "int",
    },
    {
        "rulename": "Bradycardie (FC basse)",
        "regex": r"\b(?:fc|fr[ée]quence\s+cardiaque|pouls)\b\s*(?:[:=àa]|de\s+|[àa]\s+)?\s*([0-9]{2,3})\s*(?:/\s*min|bpm|bat(?:tements)?(?:/min)?)?\b",
        "critical_value": 50,
        "type_of_inequality": "<",
        "diagnosis": "R00",
        "label": "Bradycardie",
        "value_type": "int",
    },
    {
        "rulename": "Allongement espace QT",
        "regex": r"\bQT(?:c)?(?:[+àa :>=]+?|de\s+)*([0-9]{3})\b",
        "critical_value": 440,
        "type_of_inequality": ">=",
        "diagnosis": "R94",
        "label": "Allongement du QT",
        "value_type": "int",
    },
    {
        "rulename": "Hypotension artérielle (TA/PA)",
        "regex": r"(?:\bTA\b|T\.A\.?|\btension\s+art[ée]rielle\b|\bpression\s+art[ée]rielle\b)\s*(?:[:=àa]|de\s+)?\s*([0-9]{2,3})\s*/\s*[0-9]{1,3}(?:\s*mm\s*Hg)?",
        "critical_value": 90,
        "type_of_inequality": "<",
        "diagnosis": "I95",
        "label": "Hypotension artérielle",
        "value_type": "int",
    },
    {
        "rulename": "Hypotension artérielle (TA/PA format abrégé)",
        "regex": r"(?:\bTA\b|T\.A\.?|\btension\s+art[ée]rielle\b|\bpression\s+art[ée]rielle\b)\s*(?:[:=àa]|de\s+)?\s*([0-9])\s*/\s*[0-9](?:\s*mm\s*Hg)?",
        "critical_value": 9,
        "type_of_inequality": "<",
        "diagnosis": "I95",
        "label": "Hypotension artérielle",
        "value_type": "int",
    },
    {
        "rulename": "Hypotension artérielle (PAS)",
        "regex": r"\bPAS\s*(?:[:=àa]|de\s+)?\s*([0-9]{2,3})\s*(?:mm\s*Hg)?\b",
        "critical_value": 90,
        "type_of_inequality": "<",
        "diagnosis": "I95",
        "label": "Hypotension artérielle",
        "value_type": "int",
    },
    {
        "rulename": "Score SOFA",
        "regex": r"\bSOFA(?:[+àa :>=]+?|de\s+)*([0-9]{1,2})\b",
        "critical_value": 2,
        "type_of_inequality": ">=",
        "diagnosis": "A41",
        "label": "Sepsis",
        "value_type": "int",
    },
    {
        "rulename": "CRP quantifiée",
        "regex": r"\b(?:CRP|prot[ée]ine C r[ée]active)(?:[+àa :>=]+?|de\s+)*([0-9]{1,5})\b",
        "critical_value": 6,
        "type_of_inequality": ">=",
        "diagnosis": "R79",
        "label": "Syndrome inflammatoire biologique",
        "value_type": "int",
    },
    {
        "rulename": "Hypercalcémie quantifiée",
        "regex": r"\b(?:Ca(?:lc[ée]mie|lcium|2\+)?|Ca\+\+)(?:[\s:àa+>=]+|de\s+)*([0-9]{1,2}(?:[.,][0-9]{1,2})?)\b",
        "critical_value": 2.6,
        "type_of_inequality": ">=",
        "diagnosis": "E83",
        "label": "Hypercalcémie",
        "value_type": "float",
    },
    {
        "rulename": "Hypercalcémie quantifiée, ionisée",
        "regex": r"(?:calc[ée]mie|calcium|ca(?:\+\+)?)?\s*ionis[ée]e?(?:[\s:=-]+)([0-9]{1,2}(?:[.,][0-9]{1,2})?)",
        "critical_value": 1.35,
        "type_of_inequality": ">=",
        "diagnosis": "E83",
        "label": "Hypercalcémie",
        "value_type": "float",
    },
    {
        "rulename": "Hypocalcémie quantifiée",
        "regex": r"\b(?:Ca(?:lc[ée]mie|lcium|2\+)?|Ca\+\+)(?:[\s:àa+<=]+|de\s+)*([0-9]{1,2}(?:[.,][0-9]{1,2})?)\b",
        "critical_value": 2.2,
        "type_of_inequality": "<",
        "diagnosis": "E83",
        "label": "Hypocalcémie",
        "value_type": "float",
    },
    {
        "rulename": "Hypocalcémie quantifiée, ionisée",
        "regex": r"(?:calc[ée]mie|calcium|ca(?:\+\+)?)?\s*ionis[ée]e?(?:[\s:=-]+)([0-9]{1,2}(?:[.,][0-9]{1,2})?)",
        "critical_value": 1.17,
        "type_of_inequality": "<",
        "diagnosis": "E83",
        "label": "Hypocalcémie",
        "value_type": "float",
    },
]
import re

def extract_numeric_rule_based_codes(text: str) -> list:
    """Extract CIM-10 codes from quantified clinical values using threshold rules."""
    if not text:
        return []

    normalized = re.sub(r"\s+", " ", text.lower())

    negation_patterns = [
        r"\babsence\b",
        r"\babsent(?:e)?\b",
        r"\bpas\s+d[e']",
        r"\baucun(?:e)?\b",
        r"\bsans\s+(?:anomalie|signe\s+de?)\b",
        r"\bnon\s+(?:d[ée]celable|document[ée]e?|confirm[ée]e?|retrouv[ée]e?|visible|objectiv[ée]e?)\b",
        r"\bn[ée]gatif(?:ve)?\b",
    ]

    def is_negated_in_context(match_start: int, match_end: int) -> bool:
        window = normalized[max(0, match_start - 60): min(len(normalized), match_end + 40)]
        return any(re.search(pattern, window) for pattern in negation_patterns)

    def parse_value(value: str, value_type: str):
        parsed = value.replace(",", ".")
        return int(float(parsed)) if value_type == "int" else float(parsed)

    def compare(value, operator: str, critical_value) -> bool:
        if operator == "<":
            return value < critical_value
        if operator == "<=":
            return value <= critical_value
        if operator == ">":
            return value > critical_value
        if operator == ">=":
            return value >= critical_value
        return False

    def resolve_renal_numeric_code(markers: list[dict[str, Any]]) -> dict | None:
        if not markers:
            return None

        marker_names = {item["marker"] for item in markers}

        chronic_patterns = [
            r"\binsuffisance\s+r[ée]nale\s+chronique\b",
            r"\bmaladie\s+r[ée]nale\s+chronique\b",
            r"\bmrc\b",
            r"\bn[ée]phropathie\s+chronique\b",
            r"\bdialyse\s+chronique\b",
            r"\bh[ée]modialyse\s+chronique\b",
        ]
        acute_patterns = [
            r"\binsuffisance\s+r[ée]nale\s+aigu[ëe]?\b",
            r"\bacute\s+kidney\s+injury\b",
            r"\baggravation\s+(?:aigu[ëe]?|r[ée]cente)\s+de\s+la\s+fonction\s+r[ée]nale\b",
            r"\bd[ée]gradation\s+r[ée]nale\s+r[ée]cente\b",
        ]
        renal_context_patterns = [
            r"\binsuffisance\s+r[ée]nale\b",
            r"\bf(onction)?\s*r[ée]nale\s+alt[ée]r[ée]e?\b",
            r"\batteinte\s+r[ée]nale\b",
            r"\bn[ée]phropathie\b",
            r"\bdialyse\b",
            r"\bh[ée]modialyse\b",
        ]

        has_chronic_context = any(re.search(pattern, normalized) for pattern in chronic_patterns)
        has_acute_context = any(re.search(pattern, normalized) for pattern in acute_patterns)
        has_renal_context = has_chronic_context or has_acute_context or any(
            re.search(pattern, normalized) for pattern in renal_context_patterns
        )
        has_strong_marker = bool(marker_names.intersection({"creatinine", "gfr"}))

        if has_chronic_context:
            return {"problem": "Insuffisance rénale chronique", "code": "N18"}

        if has_acute_context:
            return {"problem": "Insuffisance rénale aiguë", "code": "N17"}

        if marker_names == {"urea"} and not has_renal_context:
            return None

        if has_strong_marker or len(marker_names) >= 2 or has_renal_context:
            return {"problem": "Insuffisance rénale non précisée", "code": "N19"}

        return None

    results = []
    seen = set()
    renal_markers = []

    for rule in NUMERIC_VALUE_RULES:
        pattern = re.compile(rule["regex"], re.IGNORECASE)

        for match in pattern.finditer(text):
            if is_negated_in_context(match.start(), match.end()):
                continue

            raw_value = None

            # First try: captured numeric value in regex group(1)
            try:
                raw_value = match.group(1)
            except IndexError:
                raw_value = None

            # Fallback: use number_regex on matched text
            if raw_value is None and rule.get("number_regex"):
                value_match = re.search(rule["number_regex"], match.group(0), re.IGNORECASE)
                if value_match:
                    raw_value = value_match.group(1) if value_match.lastindex else value_match.group(0)

            if raw_value is None:
                continue

            try:
                numeric_value = parse_value(raw_value, rule["value_type"])
            except (ValueError, TypeError):
                continue

            if not compare(numeric_value, rule["type_of_inequality"], rule["critical_value"]):
                continue

            if rule.get("resolver") == "renal_failure":
                renal_markers.append({
                    "marker": rule.get("marker", rule["rulename"]),
                    "value": numeric_value,
                    "problem": rule.get("label", rule["rulename"]),
                })
                break

            key = (rule["diagnosis"], rule.get("label", rule["rulename"]))
            if key in seen:
                continue
       
            results.append({
                "problem": rule.get("label", rule["rulename"]),
                "code": rule["diagnosis"]
            })

            seen.add(key)
            break

    renal_item = resolve_renal_numeric_code(renal_markers)
    if renal_item:
        key = (renal_item["code"], renal_item["problem"])
        if key not in seen:
            results.append(renal_item)
            seen.add(key)

    return results


def severe_context(text: str) -> bool:
    """
    Return True when severe functional/cognitive/palliative context is present.

    This is used as a conservative fallback to infer F00 in Alzheimer context
    when explicit dementia/confusion wording is absent from the note.
    """
    t = (text or "").lower()

    functional = [
        r"\bgrabataire\b",
        r"\balit[ée]e?\b",
        r"\bd[ée]pendance\b",
        r"\bperte\s+d['’]?autonomie\b",
    ]
    cognitive = [
        r"\bpas\s+de\s+contact\b",
        r"\bne\s+r[ée]pond\s+pas\b",
        r"\bmutique\b",
    ]
    palliative = [
        r"\bsoins\s+palliatifs\b",
        r"\bsoins\s+de\s+confort\b",
        r"\bnon\s+r[ée]animatoire\b",
    ]

    score = 0
    if any(re.search(p, t) for p in functional):
        score += 1
    if any(re.search(p, t) for p in cognitive):
        score += 1
    if any(re.search(p, t) for p in palliative):
        score += 1

    return score >= 2

def rule_based_diagnosis_codes(disorder_text: str, context_terms: list | None = None, context_codes: list | None = None) -> list:
    """
    Return deterministic CIM-10 codes for a single diagnosis string, optionally using
    surrounding diagnosis terms/codes as context.

    This is the term-level companion to scan_full_text_rule_based_codes(). It is used
    inside the processing fallback chain when an extracted diagnosis string needs a fast,
    deterministic code before trying mapping/classifier fallback.
    """
    normalized = (disorder_text or "").strip().lower()
    if not normalized:
        return []

    # Cognitive disorder debug and rule
    """
    cognitive_match = re.search(r"\btroubles?\s+(?:neuro)?cognitifs?\b", normalized)
    alzheimer_match = re.search(r"alzheimer", normalized)
    if cognitive_match and not alzheimer_match:
        return ["F03"]
    """
    # Normalize common Alzheimer spelling first if needed
    normalized = re.sub(r"\balz[a-z]*heimer\b", "alzheimer", normalized, flags=re.IGNORECASE)

    # 1. Explicit dementia / major neurocognitive disorder
    dementia_match = re.search(
        r"\bd[ée]mence\b|\bsyndrome\s+d[ée]mentiel\b|\btroubles?\s+neurocognitifs?\s+majeurs?\b",
        normalized,
        flags=re.IGNORECASE
    )

    # 2. Alzheimer disease
    alzheimer_match = re.search(r"\balzheimer\b", normalized, flags=re.IGNORECASE)

    context_terms = [(term or "").strip().lower() for term in (context_terms or [])]
    context_codes = [str(code).strip().upper()[:3] for code in (context_codes or []) if code]

    # 3. Generic cognitive symptoms
    cognitive_match = re.search(
        r"\btroubles?\s+(?:neuro)?cognitifs?\b|\bamn[ée]sie\b|\bd[ée]sorient(?:[ée]|ation)?\b",
        normalized,
        flags=re.IGNORECASE
    )

    severe_cognitive_pattern = (
        r"\btroubles?\s+cognitifs?\s+s[ée]v[èe]res?\b|"
        r"\btroubles?\s+neurocognitifs?\s+s[ée]v[èe]res?\b|"
        r"\bd[ée]clin\s+cognitif\s+s[ée]v[èe]re\b|"
        r"\balt[ée]ration\s+cognitive\s+s[ée]v[èe]re\b"
    )

    severe_cognitive_match = re.search(
        severe_cognitive_pattern,
        normalized,
        flags=re.IGNORECASE
    )

    context_blob = " ".join([normalized] + context_terms)

    has_severe_cognitive_context = bool(re.search(severe_cognitive_pattern, context_blob, re.IGNORECASE))

    if severe_cognitive_match or (cognitive_match and has_severe_cognitive_context):
        if re.search(r"\bd[ée]mence\b|\batrophie\b|\bd[ée]clin\b|\bchronique\b", context_blob, re.IGNORECASE):
            return ["F03"]
        if re.search(r"\bconfusion\b|\baigu[ëe]?\b|\bfluctuan\w*\b|\bfi[ée]vre\b|\binfection\b", context_blob, re.IGNORECASE):
            return ["F05"]
        return ["F03"]

    if dementia_match:
        return ["F03"]

    if alzheimer_match:
        return ["G30"]

    if cognitive_match:
        return ["R41"]

    def has_context_term(pattern: str) -> bool:
        return any(re.search(pattern, term) for term in context_terms)

    has_stroke_context = "I63" in context_codes or "I69" in context_codes or has_context_term(r"\bavc\b")
    if re.search(r"\b(?:s[ée]quelles?|s[ée]quellaire)\b", normalized):
        if re.search(r"\bavc\b", normalized) or has_stroke_context:
            return ["I69"]

    if re.search(r"\b(?:depuis|post[-\s]?|apr[èe]s)\s+avc\b", normalized):
        return ["I69"]

    rules = [
        # N39 - ECBU positif (urinary infection proven by lab)
        (r"\becbu\b.{0,20}\bpositif\b", ["N39"]),
        # N39 - urinary infection (explicit, covers more bacteria)
        (r"\b(?:infection\s+urinaire|i\.?\s*u\.?|cystite)\b(?:\s+[àa]\s+[a-z0-9\.-]{1,30})?", ["N39"]),
        # B96 - Raoultella planticola as agent (explicit rule)
        (r"\braoultella\s+planticola\b", ["B96"]),
        
        # S01 - open wound of head / face
        (r"\bplaie\s+(?:du|de\s+la)\s+scalp\b|\bplaie\s+de\s+scalp\b|\bscalp\b.{0,20}\bplaie\b|\bplaie\b.{0,20}\bscalp\b|\bplaie\s+d['’]arcade\b|\bplaie\s+arcade\s+sourcili[èe]re\b|\barcade\s+sourcili[èe]re\b.{0,20}\bplaie\b|\bplaie\b.{0,20}\barcade\s+sourcili[èe]re\b", ["S01"]),
        # S09 - head trauma, unspecified (non-severe wording)
        (r"\b(?:traumatisme|trauma)\s+cr[âa]nien\b(?!\s+grave)", ["S09"]),
        # S06 - intracranial injury
        (r"\b(?:tc\s+grave|(?:tc|traumatisme|trauma)\s+cr[âa]nien\s+grave)\b", ["S06"]),
        (r"\bh[ée]morragie\s+sous[- ]arachno[iï]dienne\b", ["S06"]),
        (r"\bh[ée]morragie\s+m[ée]ning[ée]e\b", ["S06"]),
        (r"\bh[ée]morragie\s+intra[- ]?ventriculaire\b", ["S06"]),
        (r"\bp[ée]t[ée]chie\s+h[ée]morragique\b", ["S06"]),
        (r"\bh[ée]matome\s+(?:sous[- ]dural|extra[- ]dural|[ée]pidural)\b", ["S06"]),
        (r"\bcontusion\s+c[ée]r[ée]brale\b", ["S06"]),
        (r"\bl[ée]sion\s+intracr[âa]nienne\b", ["S06"]),

        # S22 - thoracic vertebra / thoracic cage
        (r"\b(?:fract(?:\.|ure)?s?|fx)\s+(?:costales?|c[ôo]tes|des?\s+c[ôo]tes|arcs?\s+costaux?|des?\s+arcs?\s+costaux?)(?:\s+[àa]\s+(?:g|d|gauche|droite))?\b", ["S22"]),
        (r"\bfracture[-\s]?tassement\s+du\s+corps\s+vert[ée]bral\s+de\s+t\d+\b", ["S22"]),
        (r"\bfracture\s+horizontale\s+du\s+corps\s+vert[ée]bral\s+de\s+t\d+\b", ["S22"]),
        (r"\bfracture\s+du\s+corps\s+vert[ée]bral\s+de\s+t\d+\b", ["S22"]),
        (r"\bh[ée]mopneumothorax\b", ["S27"]),

        # S32 - lumbar spine / pelvis
        (r"\bfracture[-\s]?tassement\s+du\s+corps\s+vert[ée]bral\s+de\s+l\d+\b", ["S32"]),
        (r"\bfracture\s+du\s+corps\s+vert[ée]bral\s+de\s+l\d+\b", ["S32"]),
        # General lumbar vertebra fracture: 'fracture de L1', 'fracture de L2', ..., 'fracture de L5'
        (r"\bfracture\s+de\s+[L,l][1-5]\b", ["S32"]),
        (r"\bfracture\s+[L,l][1-5]\b", ["S32"]),
        (r"\bfracture\s+de\s+la\s+branche\s+ischio[- ]pubienne\b", ["S32"]),
        (r"\bfracture\s+comminutive\s+de\s+la\s+branche\s+ilio[- ]pubienne\b", ["S32"]),
        (r"\bfracture\s+de\s+la\s+branche\s+ilio[- ]pubienne\b", ["S32"]),
        (r"\bfracture\s+articulaire\s+compl[èe]te\s+du\s+cotyle\b", ["S32"]),
        (r"\bfracture\s+du\s+cotyle\b", ["S32"]),

        # S72 - femur (femoral neck)
        (r"\bfracture(?:\s+\w+){0,2}\s+(?:du\s+)?col(?:\s+(?:du\s+f[ée]mur|f[ée]moral))?(?:\s+(?:droit|gauche))?\b", ["S72"]),
        (r"\bfracture\s+cervico[- ]?c[ée]phalique(?:\s+(?:droit|gauche))?\b", ["S72"]),

        # S42 - scapula / proximal humerus
        (r"\bfracture\s+(?:de\s+l['’])?omoplate(?:\s+(?:gauche|droite))?(?:\s+non\s+d[ée]plac[ée]e?)?\b", ["S42"]),
        (r"\bfracture\s+(?:de\s+la\s+)?scapula(?:\s+(?:gauche|droite))?(?:\s+non\s+d[ée]plac[ée]e?)?\b", ["S42"]),
        (r"\bfracture\s+articulaire\s+de\s+la\s+gl[èe]ne\s+scapulaire\b", ["S42"]),
        (r"\bfracture\s+de\s+la\s+gl[èe]ne\s+scapulaire\b", ["S42"]),
        (r"\bfracture\s+(?:de\s+la\s+)?clavicule(?:\s+(?:droite|gauche))?\b", ["S42"]),
        (r"\bfracture\s+de\s+l['’]extr[ée]mit[ée]\s+sup[ée]rieure\s+de\s+l['’]hum[ée]rus\b", ["S42"]),
        (r"\bfracture\s+de\s+l['’]hum[ée]rus\s+proximal\b", ["S42"]),

        # I63 - acute ischemic stroke (radiology/report wording)
        (r"\b(?:avc\s+isch[ée]mique|aic|accident\s+isch[ée]mique\s+c[ée]r[ée]bral)\b|\b(?:infarctus|isch[ée]mie|hypodensit[ée])\b.{0,80}\bcapsulo[- ]?lenticulaire\b|\bocclusion\s+de\s+m1\b|\bm1\b.{0,30}\bocclu(?:sion|[eé])\b", ["I63"]),
        (r"\bavc\s+(?:ancien|ant[ée]c[ée]dent|en\s+\d{4})\b|\bancien\s+avc\b|\bant[ée]c[ée]dent(?:s)?\s+d['’]?avc\b|\batcd\s*[:\-]?\s*avc\b|\batcd\b.{0,20}\bavc\b", ["I69"]),
        #R40 - coma / consciousness disorders
        (r"\btroubles?\s+de\s+(?:la\s+)?conscience\b|\balt[ée]ration\s+de\s+la\s+conscience\b|\bcoma\b|\bcomat(?:eux|euse)\b|\b(?:gcs|glasgow|g)\s*(?:score)?\s*[:=]?\s*(?:[3-9]|1[0-4])\b", ["R40"]),
        #F10 - mental and behavioral disorders due to use of alcohol
        (r"\b(?:oh\b.{0,20}\b(?:chronique|non\s+sevr[ée]?|sevrage)|alcool(?:isme|o[- ]d[ée]pendance)?|[ée]thylisme|exog[ée]nose)(?:\s+(?:chronique|aigu[ëe]?|active|non\s+sevr[ée]?|non\s+sevr[eé]))*\b", ["F10"]),

        (r"r[ée]tention\s+(?:d['’]urine|urinaire)|globe\s+v[ée]sical", ["R33"]),
        (r"\b(?:infection\s+urinaire|i\.?\s*u\.?|cystite)\b(?:\s+[àa]\s+[a-z0-9\.-]{1,30})?", ["N39"]),
        (r"\bky(?:s|st)e?s?\s+(?:cortic(?:al|aux)(?:\s+r[ée]n(?:al|aux))?|r[ée]n(?:al|aux)\s+cortic(?:al|aux)|r[ée]n(?:al|aux)|du\s+rein)\b", ["N28"]),
        (r"syndrome\s+inflammatoire(?:\s+biologique)?", ["R79"]),
        (r"\bdifficult[ée]\s+[àa]\s+la\s+marche\b|\btroubles?\s+de\s+la\s+marche\b|\btroubles?\s+de\s+l['’]?[ée]quilibre\b|\binstabilit[ée]\s+posturale\b|\bmarche\b.{0,30}\bdifficilement\b|\bmarche\s+(?:tr[èe]s\s+)?difficile(?:ment)?\b|\b(?:[ée]tat\s+)?grabataire\b|\b(?:reprise\s+de\s+la\s+marche|reprend\s+la\s+marche)\s+[àa]\s+l['’]aide\s+d['’]un\s+d[ée]ambulateur\b|\bmarche\s+avec\s+(?:un\s+)?d[ée]ambulateur\b|\bimpossibilit[ée]\s+de\s+se\s+d[ée]placer(?:\s+dans\s+son\s+propre\s+domicile)?\b|\bincapacit[ée]\s+[àa]\s+se\s+d[ée]placer(?:\s+(?:au\s+domicile|[àa]\s+domicile|dans\s+son\s+domicile|chez\s+(?:elle|lui)))?\b|\bne\s+peut\s+plus\s+se\s+d[ée]placer(?:\s+(?:au\s+domicile|[àa]\s+domicile|dans\s+son\s+domicile|chez\s+(?:elle|lui)))?\b|\br[ée]-?autonomisation\b.{0,40}\b(?:marche|transferts?)\b", ["R26"]),
        (r"\bretrouv(?:e|é|ée)\s+au\s+sol\b|\bchutes?\s+s['’]?acc[ée]l[èe]rent\b", ["W19"]),
        (r"\bchutes?\s+[àa]\s+r[ée]p[ée]titions?\b|\bchuteur\s+chronique\b|\b(?:r[ée]flexe\s+de\s+)?grasping\b|\bsigne\s+de\s+pr[ée]hension\b", ["R29"]),
        (r"(?:\b(?:antibioth[ée]rapie|antibiotique(?:s)?)\b.{0,160}\b(?:vis[ée]e?\s+respiratoire|respiratoire)\b.{0,160}\bfoyer\s+auscultatoire\b|"
           r"\bfoyer\s+auscultatoire\b.{0,160}\b(?:antibioth[ée]rapie|antibiotique(?:s)?)\b.{0,160}\b(?:vis[ée]e?\s+respiratoire|respiratoire)\b|"
           r"\bfoyer\s+auscultatoire\b.{0,120}\b(?:vis[ée]e?\s+respiratoire|respiratoire)\b|"
           r"\b(?:vis[ée]e?\s+respiratoire|respiratoire)\b.{0,120}\bfoyer\s+auscultatoire\b)", ["J18"]),
        (
           r"(?:\binfection\s+pulmonaire\b.{0,80}\b(?:sous|trait[ée]e?\s+par|ttt\s+par)\b.{0,40}\b(?:rocephine|ceftriaxone|antibioth[ée]rapie|antibiotique(?:s)?)\b|"
           r"\b(?:rocephine|ceftriaxone|antibioth[ée]rapie|antibiotique(?:s)?)\b.{0,80}\binfection\s+pulmonaire\b)", ["J18"],
        ),
    ]

    matched_codes = []

    for pattern, codes in rules:
        if re.search(pattern, normalized):
            single_code = codes[0] if len(codes) == 1 else None

            # Guard against short negated phrases in term-level fallback.
            if single_code in {"S06", "S09", "S22", "S27", "S32", "S42", "S72"}:
                trauma_negation_patterns = [
                    r"\bpas\s+de\s+(?:traumatisme\s+cr[âa]nien|h[ée]morragie|h[ée]matome|contusion|l[ée]sion\s+intracr[âa]nienne|fracture)\b",
                    r"\bsans\s+(?:traumatisme\s+cr[âa]nien|h[ée]morragie|h[ée]matome|contusion|l[ée]sion\s+intracr[âa]nienne|fracture)\b",
                    r"\babsence\s+de\s+(?:traumatisme\s+cr[âa]nien|h[ée]morragie|h[ée]matome|contusion|l[ée]sion\s+intracr[âa]nienne|fracture)\b",
                    r"\b(?:traumatisme\s+cr[âa]nien|h[ée]morragie|h[ée]matome|contusion|l[ée]sion\s+intracr[âa]nienne|fracture)\s+(?:non\s+retrouv[ée]e?|absent(?:e)?|[ée]cart[ée]e?|exclu(?:e)?)\b",
                ]
                if any(re.search(p, normalized) for p in trauma_negation_patterns):
                    continue

            if single_code == "R33":
                r33_term_negation_patterns = [
                    r"\bpas\s+de\s+(?:globe\s+v[ée]sical|r[ée]tention\s+urinaire)\b",
                    r"\bsans\s+(?:globe\s+v[ée]sical|r[ée]tention\s+urinaire)\b",
                    r"\babsence\s+de\s+(?:globe\s+v[ée]sical|r[ée]tention\s+urinaire)\b",
                    r"\b(?:[ée]liminer|[ée]limin[ée]e?s?)\b.{0,40}\b(?:globe\s+v[ée]sical|r[ée]tention\s+urinaire)\b",
                ]
                if any(re.search(p, normalized) for p in r33_term_negation_patterns):
                    continue

            if single_code == "N39":
                n39_term_negation_patterns = [
                    r"\bpas\s+d(?:e|['’])\s*(?:infection\s+urinaire|i\.?\s*u\.?|cystite)\b",
                    r"\bsans\s+(?:infection\s+urinaire|i\.?\s*u\.?|cystite)\b",
                    r"\babsence\s+d(?:e|['’])\s*(?:infection\s+urinaire|i\.?\s*u\.?|cystite)\b",
                    r"\b(?:infection\s+urinaire|i\.?\s*u\.?|cystite)\s+(?:non\s+retrouv[ée]e?|absente?|[ée]cart[ée]e?|exclu(?:e)?)\b",
                ]
                if any(re.search(p, normalized) for p in n39_term_negation_patterns):
                    continue

            if codes == ["R26"]:
                r26_negation_patterns = [
                    r"\bpas\s+de\s+(?:difficult[ée]\s+[àa]\s+la\s+marche|troubles?\s+de\s+la\s+marche|troubles?\s+de\s+l['’]?[ée]quilibre)\b",
                    r"\bsans\s+(?:difficult[ée]\s+[àa]\s+la\s+marche|troubles?\s+de\s+la\s+marche|troubles?\s+de\s+l['’]?[ée]quilibre)\b",
                    r"\babsence\s+de\s+(?:difficult[ée]\s+[àa]\s+la\s+marche|troubles?\s+de\s+la\s+marche|troubles?\s+de\s+l['’]?[ée]quilibre)\b",
                ]
                if any(re.search(p, normalized) for p in r26_negation_patterns):
                    continue
            for code in codes:
                if code not in matched_codes:
                    matched_codes.append(code)

    if matched_codes:
        return matched_codes
            
    if re.search(r"\b(d[ée]mence|troubles?\s+(?:neuro)?cognitifs?\s+majeurs?)\b", normalized):
        if "G30" in context_codes or has_context_term(r"\balzheimer\b"):
            return ["F00"]
        
    return []
def scan_full_text_rule_based_codes(text: str, existing_codes: list = None) -> list:

    """
    Scan the full clinical text for high-value clinical diagnoses using rule-based patterns,
    mirroring how extract_icd_codes_from_rules() works for Z-codes.
    Each pattern is checked with a negation-context window so negated mentions are skipped.
    Returns [{'problem': '<label>', 'code': '<code3>'}].

    Add new rules here to cover diagnoses normally not detected as medical disorders
    by NER/QuickUMLS (e.g., lab abnormalities, syndromes, cognitive disorders).
    """
    import re
    if not text:
        return []

    normalized = re.sub(r"\s+", " ", text.lower())

    # If existing_codes is provided, use it as the initial results list
    results = list(existing_codes) if existing_codes else []
    # Only include dicts with a 'code' key to avoid errors
    def extract_code_val(val):
        if isinstance(val, dict) and "code" in val:
            return val["code"]
        return val

    negation_patterns = [
        r"\babsence\b",
        r"\babsent(?:e)?\b",
        r"\bpas\s+d[e']",
        r"\baucun(?:e)?\b",
        r"\bsans\s+(?:anomalie|signe\s+de?)\b",
        r"\bnon\s+(?:d[ée]celable|document[ée]e?|confirm[ée]e?|retrouv[ée]e?|visible|objectiv[ée]e?)\b",
        r"\bn[ée]gatif(?:ve)?\b",
        r"\bexclure?\b",
        r"\bexclu(?:e)?\b",
    ]

    def is_negated_in_context(match_start: int, match_end: int) -> bool:
        # Increased window for robust negation detection
        window = normalized[max(0, match_start - 60): min(len(normalized), match_end + 40)]
        return any(re.search(p, window) for p in negation_patterns)

    e86_negation_patterns = [
        r"\babsence\s+de\s+d[ée]shydrat(?:ation|[ée]e?)\b",
        r"\bpas\s+de\s+d[ée]shydrat(?:ation|[ée]e?)\b",
        r"\bsans\s+d[ée]shydrat(?:ation|[ée]e?)\b",
        r"\bd[ée]shydrat(?:ation|[ée]e?)\s+(?:non\s+retrouv[ée]e?|absente?)\b",
        r"\bpas\s+de\s+signe(?:s)?\s+de\s+d[ée]shydrat(?:ation|[ée]e?)\b",
    ]

    r33_negation_patterns = [
        r"\b(?:[ée]liminer|[ée]limin[ée]e?s?)\b.{0,40}\b(?:globe\s+v[ée]sical|r[ée]tention\s+urinaire)\b",
        r"\b(?:globe\s+v[ée]sical|r[ée]tention\s+urinaire)\b.{0,40}\b(?:[ée]liminer|[ée]limin[ée]e?s?)\b",
        r"\babsence\s+de\s+(?:globe\s+v[ée]sical|r[ée]tention\s+urinaire)\b",
        r"\bpas\s+de\s+(?:globe\s+v[ée]sical|r[ée]tention\s+urinaire)\b",
        r"\bsans\s+(?:globe\s+v[ée]sical|r[ée]tention\s+urinaire)\b",
    ]

    j18_strong_positive_patterns = [
        r"\bfoyer\s+auscultatoire\b",
        r"\bvis[ée]e?\s+respiratoire\b",
    ]

    j18_negation_patterns = [
        r"\babsence\s+de\s+pneumopathie\b",
        r"\bpas\s+de\s+pneumopathie\b",
        r"\bsans\s+pneumopathie\b",
        r"\bpneumopathie\s+(?:exclue|[ée]cart[ée]e|non\s+retenue)\b",
        r"\babsence\s+de\s+foyer\s+auscultatoire\b",
        r"\bpas\s+de\s+foyer\s+auscultatoire\b",
        r"\bsans\s+foyer\s+auscultatoire\b",
    ]

    n39_negation_patterns = [
        r"\bpas\s+d(?:e|['’])\s*(?:infection\s+urinaire|i\.?\s*u\.?|cystite)\b",
        r"\bsans\s+(?:infection\s+urinaire|i\.?\s*u\.?|cystite)\b",
        r"\babsence\s+d(?:e|['’])\s*(?:infection\s+urinaire|i\.?\s*u\.?|cystite)\b",
        r"\b(?:infection\s+urinaire|i\.?\s*u\.?|cystite)\s+(?:non\s+retrouv[ée]e?|absente?|[ée]cart[ée]e?|exclu(?:e)?)\b",
    ]

    r26_negation_patterns = [
        r"\bpas\s+de\s+(?:difficult[ée]\s+[àa]\s+la\s+marche|troubles?\s+de\s+la\s+marche|troubles?\s+de\s+l['’]?[ée]quilibre)\b",
        r"\bsans\s+(?:difficult[ée]\s+[àa]\s+la\s+marche|troubles?\s+de\s+la\s+marche|troubles?\s+de\s+l['’]?[ée]quilibre)\b",
        r"\babsence\s+de\s+(?:difficult[ée]\s+[àa]\s+la\s+marche|troubles?\s+de\s+la\s+marche|troubles?\s+de\s+l['’]?[ée]quilibre)\b",
    ]

    def is_negated_for_code(match_start: int, match_end: int, code: str) -> bool:
        window = normalized[max(0, match_start - 60): min(len(normalized), match_end + 40)]
        if code == "E86":
            return any(re.search(p, window) for p in e86_negation_patterns)
        if code == "R33":
            return any(re.search(p, window) for p in r33_negation_patterns) or is_negated_in_context(match_start, match_end)
        if code == "N39":
            # Avoid suppressing urinary infection on unrelated negation (e.g., "pas d'OMI").
            return any(re.search(p, window) for p in n39_negation_patterns)
        if code == "R26":
            # Avoid suppressing gait disorder mentions on unrelated negation (e.g., "pas d'IRA").
            return any(re.search(p, window) for p in r26_negation_patterns)
        if code == "J18":
            match_span = normalized[match_start:match_end]
            has_strong_positive = all(re.search(p, match_span) for p in j18_strong_positive_patterns)
            if has_strong_positive:
                # Keep strong implicit pneumonia clues unless J18-specific negation is explicit.
                return any(re.search(p, window) for p in j18_negation_patterns)
        return is_negated_in_context(match_start, match_end)

    def has_stroke_sequelae_context() -> bool:
        has_stroke_code = any(
            isinstance(item, dict)
            and "code" in item
            and extract_code_val(item["code"]) in {"I63", "I69"}
            for item in results
        )
        if has_stroke_code and re.search(r"\b(?:s[ée]quelles?|s[ée]quellaire)\b", normalized):
            return True

        sequela_patterns = [
            r"\bs[ée]quelles?\b.{0,30}\bavc\b",
            r"\bavc\b.{0,30}\bs[ée]quelles?\b",
            r"\bs[ée]quellaire\b.{0,30}\bavc\b",
            r"\b(?:depuis|post[-\s]?|apr[èe]s)\s+avc\b",
            r"\bavc\s+(?:ancien|ant[ée]c[ée]dent|en\s+\d{4})\b",
            r"\bancien\s+avc\b",
            r"\bant[ée]c[ée]dent(?:s)?\s+d['’]?avc\b",
            r"\batcd\s*[:\-]?\s*avc\b",
            r"\batcd\b.{0,20}\bavc\b",
        ]
        return any(re.search(pattern, normalized) for pattern in sequela_patterns)

    def has_acute_stroke_context() -> bool:
        acute_patterns = [
            r"\baic\b",
            r"\bavc\s+r[ée]cent\b",
            r"\bavc\s+isch[ée]mique\b",
            r"\baccident\s+isch[ée]mique\s+c[ée]r[ée]bral\b",
            r"\bocclusion\s+de\s+m1\b",
            r"\bm1\b.{0,30}\bocclu(?:sion|[eé])\b",
        ]
        return any(re.search(pattern, normalized) for pattern in acute_patterns)

    severe_cognitive_pattern = r"\btroubles?\s+(?:neuro)?cognitifs?\s+s[ée]v[èe]res?\b|\bd[ée]clin\s+cognitif\s+s[ée]v[èe]re\b|\balt[ée]ration\s+cognitive\s+s[ée]v[èe]re\b"
    chronic_cognitive_context_patterns = [
        r"\bd[ée]mence\b",
        r"\batrophie\b",
        r"\bd[ée]clin\b",
        r"\bchronique\b",
    ]
    acute_cognitive_context_patterns = [
        r"\bconfusion\b",
        r"\baigu[ëe]?\b",
        r"\bfluctuan\w*\b",
        r"\bfi[ée]vre\b",
        r"\binfection\b",
    ]

    seen_codes: set = set([
        extract_code_val(item["code"])
        for item in results
        if (
            isinstance(item, dict)
            and "code" in item
            and isinstance(extract_code_val(item["code"]), (str, int, float))
        )
    ])

    # Add numeric rule-based codes to the initial results and seen_codes, with negation check
    numeric_items = extract_numeric_rule_based_codes(text)
    for item in numeric_items:
        # Try to find the code mention in the text and check for negation
        code_val = extract_code_val(item["code"])
        # Try to find the code's label or code in the text for negation context
        label = item.get("problem", "")
        # Build a regex for the label or code (if label is present)
        label_pattern = re.escape(label.lower()) if label else None
        code_pattern = re.escape(str(code_val))
        found = False
        if label_pattern:
            for m in re.finditer(label_pattern, normalized):
                found = True
                if is_negated_in_context(m.start(), m.end()):
                    break
            else:
                # If not negated, add
                if code_val not in seen_codes:
                    results.append(item)
                    seen_codes.add(code_val)
        else:
            # Try to find code in text (rare)
            for m in re.finditer(code_pattern, normalized):
                found = True
                if is_negated_in_context(m.start(), m.end()):
                    break
            else:
                if code_val not in seen_codes:
                    results.append(item)
                    seen_codes.add(code_val)
        # If not found in text, conservatively add (legacy behavior)
        if not found and code_val not in seen_codes:
            results.append(item)
            seen_codes.add(code_val)

    def has_infectious_context_local(seen_codes) -> bool:
        infectious_patterns = [
            r"\bpneumopathie\b",
            r"\bpneumonie\b",
            r"\bpneumopathie d['’]inhalation\b",
            r"\binfection urinaire\b",
            r"\bcystite\b",
            r"\bi\.?\s*u\.?\b",
            r"\bpy[eé]l[oo]n[ée]phrite\b",
            r"\bsepsis\b",
            r"\bchoc septique\b",
            r"\blba\b",
            r"\becbu\b",
            r"\bh[ée]moculture\b",
            r"\bculture\b",
        ]
        return any(re.search(p, normalized) for p in infectious_patterns) or bool(
            seen_codes.intersection({"J18", "J69", "N39", "N10", "A41"})
        )
    import re

    def _is_pattern_negated(match_obj, text: str, context_window: int = 60) -> bool:
        """
        Check if a regex match is negated by looking at surrounding context.
        Returns True if negation keywords (sans, pas, sans...) appear before the match.
        """
        start_pos = max(0, match_obj.start() - context_window)
        context = text[start_pos:match_obj.end()].lower()
        negation_patterns = [
            r"\bsans\b",
            r"\bpas\s+(?:de|d[' ])?",
            r"\baucun[e]?\b",
            r"\bne\s+",
            r"\babsenc[e]?\s+(?:de|d[' ])?",
        ]
        return any(re.search(neg, context) for neg in negation_patterns)

    def has_hemorrhagic_context(text: str) -> bool:
        t = text.lower()
        patterns = [
            r"\bh[ée]morrag",
            r"\bsaignement",
           # r"\btransfus(?:ion|[ée]e?)\b",
            r"\bcgr\b",
            r"\bpfc\b",
            r"\bpolytraum\w*\b.{0,30}\b(h[ée]morrag|saignement|actif)\b", #|transfus)\b",
            r"\bfracture\b.{0,20}\b(saigne|h[ée]morrag|actif)\b",
            r"\bchoc h[ée]morragique\b",
            r"\bh[ée]matome\b",
            r"\bembolisation\b",
        ]
        # Check for ANY match that is NOT negated
        for pattern in patterns:
            for match in re.finditer(pattern, t):
                if not _is_pattern_negated(match, t):
                    return True
        return False

    def shock_score(text: str) -> int:
        t = text.lower()
        score = 0
        rules = [
            (r"\bnoradr[ée]naline\b", 3),
            (r"\bvasopresseur", 3),
            (r"\bchoc\b", 4),
            (r"\bhypotendu", 2),
            (r"\binstable", 2),
            (r"\bpa\s*[:=]?\s*\d{2,3}\s*/\s*\d{2,3}", 2),
            (r"\blactate[s]?\s*[>:]?\s*\d+", 2),
            #(r"\btransfus", 2),
            (r"\bhémorragie|saignement actif", 2),
        ]

        for pattern, pts in rules:
            # Only score if pattern found and NOT negated
            for match in re.finditer(pattern, t):
                if not _is_pattern_negated(match, t):
                    score += pts
                    break  # Once pattern matches (non-negated), apply points once

        return score
    
    def has_shock_context(text: str) -> bool:
        return shock_score(text) >= 4

    def has_rhabdo_context(text: str) -> bool:
        t = text.lower()
        patterns = [
            r"\brhabdomyolyse\b",
            r"\bmyoglobinurie\b",
            r"\bck\b",
            r"\bcpk\b",
        ]
        # Return True if ANY pattern is found and NOT negated
        for pattern in patterns:
            for match in re.finditer(pattern, t):
                if not _is_pattern_negated(match, t):
                    return True
        return False

    def has_urinary_context(text: str) -> bool:
        t = text.lower()
        patterns = [
            r"\bsondage\b",
            r"\bsonde\b",
            r"\burinaire\b",
            r"\becbu\b",
            r"\bcolique n[ée]phr[ée]tique\b",
            r"\bdysurie\b",
            r"\bmiction\b",
            r"\bvessie\b",
            r"\burolog",
            r"\bh[ée]maturie\b",
            r"\burines?\s+h[ée]matiques\b",
            r"\burines?\s+rouges\b",
        ]
        # Return True if ANY pattern is found and NOT negated
        for pattern in patterns:
            for match in re.finditer(pattern, t):
                if not _is_pattern_negated(match, t):
                    return True
        return False

    # --- Custom rule for IRA (Insuffisance rénale aiguë) ---
    ira_exclude = ("mieux", "voir", "en", "à", "chez", "avec", "pour", "dans", "sur", "par", "de", "du", "des")
    for m in re.finditer(r"\bIRA\b", normalized):
        after = normalized[m.end():].lstrip()
        next_word = after.split(" ", 1)[0] if after else ""
        if next_word in ira_exclude:
            continue  # skip verb cases
        if not is_negated_in_context(m.start(), m.end()) and "N17" not in seen_codes:
            results.append({"problem": "Insuffisance rénale aiguë (IRA)", "code": "N17"})
            seen_codes.add("N17")
    # -----------------------------------------------------------------------
    # Rules: (regex_pattern, base_cim10_code, human_readable_label)
    # Add new entries here to cover diagnoses not picked up by NER/QuickUMLS.
    # -----------------------------------------------------------------------
    rules = [
        # Canal lombaire rétréci (M48)
        (r"canal lombaire r[ée]tr[ée]ci", "M48", "Canal lombaire rétréci"),
        # Epilepsy: crises d'épilepsie généralisées multiples et récidivantes (G40.0)
        (r"crises? d[’']?épilepsie généralisées multiples et récidivantes", "G40.0", "Crises d'épilepsie généralisées multiples et récidivantes"),
        # Urines hémorragiques (R31)
        (r"urines?\s+h[ée]morragiques?", "R31", "Hématurie (urines hémorragiques)"),
        (r"\becbu\b.{0,40}\bpositif\b.{0,120}\b(?:bactrim|antibiotique|antibioth[ée]rapie|traitement|infection\s+urinaire|cystite)\b|"
         r"\b(?:bactrim|antibiotique|antibioth[ée]rapie|traitement|infection\s+urinaire|cystite)\b.{0,120}\becbu\b.{0,40}\bpositif\b",
         "N39", "ECBU positif avec prise en charge infectieuse"),
        # N39 - urinary infection (explicit, covers more bacteria)
        (r"\b(?:infection\s+urinaire|i\.?\s*u\.?|cystite)\b(?:\s+[àa]\s+[a-z0-9\.-]{1,30})?", "N39", "Infection urinaire"),
        # B96 - Raoultella planticola as agent (explicit rule)
        (r"\braoultella\s+planticola\b", "B96", "Raoultella planticola comme agent infectieux"),
        
        # DRA (Détresse respiratoire aiguë) generic, map to J96
        (r"\bdra\b(?:\s+aigu[ëe])?", "J96", "Détresse respiratoire aiguë (DRA)"),
        # J80 SDRA (Acute Respiratory Distress Syndrome)
        (r"\bsdra\b|\bards\b|syndrome de détresse respiratoire aigu[ëe]", "J80", "Syndrome de détresse respiratoire aiguë (SDRA)"),
        # S01 - open wound of head / face
        (r"\bplaie\s+(?:du|de\s+la)\s+scalp\b|\bplaie\s+de\s+scalp\b|\bscalp\b.{0,20}\bplaie\b|\bplaie\b.{0,20}\bscalp\b|\bplaie\s+d['’]arcade\b|\bplaie\s+arcade\s+sourcili[èe]re\b|\barcade\s+sourcili[èe]re\b.{0,20}\bplaie\b|\bplaie\b.{0,20}\barcade\s+sourcili[èe]re\b", "S01", "Plaie de la tête / arcade"),
        # Rib fracture rule (S22) - broader, user-provided
        (r"\b(?:fract(?:\.|ure)?s?|fx)\s+(?:costales?|c[ôo]tes|des?\s+c[ôo]tes|arcs?\s+costaux?|des?\s+arcs?\s+costaux?)(?:\s+[àa]\s+(?:g|d|gauche|droite))?\b", "S22", "Fracture costale"),
        # R40
        (r"\bcoma\b|\bcomat(?:eux|euse)\b|\bobnubilation\b|\bsomnolence\b|\btroubles?\s+de\s+(?:la\s+)?conscience\b|\balt[ée]ration\s+de\s+la\s+conscience\b|\b(?:glasgow|gcs|g)\s*(?:score)?\s*[:=]?\s*(?:[3-9]|1[0-4])\b", "R40", "Perte de conscience / Obnubilation"),
        (r"\btroubles?\s+de\s+(?:la\s+)?conscience\b|\balt[ée]ration\s+de\s+la\s+conscience\b|\bcoma\b|\bcomat(?:eux|euse)\b|\b(?:gcs|glasgow|g)\s*(?:score)?\s*[:=]?\s*(?:[3-9]|1[0-4])\b|"
            r"rass\s*[-:]?\s*-?5\b|"
            r"echelle\s+de\s+richmond\s+agitation\s+s[ée]dation\s*scale.*-?5\b|"
            r"non\s+r[ée]veillable\b|"
            r"aucune\s+r[ée]ponse(\s+ni\s+\w+)*\b|"
            r"aucune\s+r[ée]ponse\s+[àa]\s+l['’]appel\b|"
            r"aucune\s+r[ée]ponse\s+[àa]\s+la\s+stimulation\b","R40", "Perte de conscience / Coma / Obnubilation"),
        # T83
        (r"\bcomplication\s+de\s+(?:la\s+)?sonde\b|\binfection\s+li[ée]e\s+à\s+la\s+sonde\b|\bobstruction\s+de\s+sonde\b|\btraumatisme\s+sur\s+sonde\b", "T83", "Complication de sonde"),
        # Y84
        (r"\biatrog[èe]ne\b|\bcomplication\s+de\s+la\s+proc[ée]dure\b|\bsecondaire\s+au\s+geste\b|\bpost[- ]proc[ée]dural\b", "Y84", "Iatrogène / Complication de procédure"),
        # Urinary
        # RAU (Rétention Aiguë d’Urines) maps to R33
        (r"\brau\b|\baur\b",                                                        "R33", "Rétention aiguë d’urines (RAU/AUR)"),
        (r"r[ée]tention\s+(?:d['’]urine|urinaire)|globe\s+v[ée]sical|sondage\s+[\u00e0a]\s+demeure", "R33", "Rétention d'urine / Sondage à demeure"),
        (r"\bh[ée]maturie[s]?\b|\bh[ée]maturie[s]?\s+macro(?:scopique)?[s]?\b|\bh[ée]maturie[s]?\s+micro(?:scopique)?[s]?\b|\burines?\s+h[ée]matiques\b|\burines?\s+rouges\b",  "R31", "Hématurie"),
        (r"\bbu[- ]?sang\s*:\s*(traces?|faible|mod[ée]r[ée]e?|fort|\+{1,3})\b",      "R31_BU", "Hématurie"),
        # Inflammatory
        (r"syndrome\s+inflammatoire(?:\s+biologique)?",                             "R79", "Syndrome inflammatoire biologique"),
        # Syndrome occlusif (intestinal obstruction syndrome) maps to K56
        (r"syndrome\s+occlusif", "K56", "Syndrome occlusif (occlusion intestinale)"),
        # Neurocognitive / dementia
        (r"\bmaladie\s+d['\u2019]alzheimer\b",                                       "G30", "Maladie d'Alzheimer"),
        (r"\balzheimer\b",                                                           "G30", "Maladie d'Alzheimer"),
        (r"troubles?\s+(?:neuro)?cognitifs?\s+majeurs?",                             "F00", "Troubles neurocognitifs majeurs"),
        (severe_cognitive_pattern,                                                   "F03", "Troubles cognitifs sévères"),
        (r"\bd[ée]mence\b|\bsyndrome\s+d[ée]mentiel\b",                              "F03", "Démence"),
        # Digestive
        (r"\bh[ée]morragie\s+digestive\b",                                           "K92", "Hémorragie digestive"),
        # Renal
        (r"\binsuffisance\s+r[ée]nale\s+aigu[ëe]?\b",                                "N17", "Insuffisance rénale aiguë"),
        (r"\banur(i[eq]|ique)\b",                                                    "N17", "Insuffisance rénale aiguë (anurie)"),
        (r"\binsuffisance\s+r[ée]nale\s+chronique\b",                                "N18", "Insuffisance rénale chronique"),
        (r"\bdilatation\s+(?:des?\s+)?(?:cavit[ée]s?\s+py[ée]localicielles?\b|py[ée]localicielle\b|\bbassinets?\b|uret[ée]ro[- ]py[ée]lo[- ]calicielle\b|uret[ée]rale\b)", "N28", "Dilatation des voies urinaires"),
        (r"\bky(?:s|st)e?s?\s+(?:cortic(?:al|aux)(?:\s+r[ée]n(?:al|aux))?|r[ée]n(?:al|aux)\s+cortic(?:al|aux)|r[ée]n(?:al|aux)|du\s+rein)\b", "N28", "Kystes rénaux"),
        (r"\bpoly(?:kystose|kyste)\s+r[ée]nale?\b",                                  "N28", "Polykystose rénale"),
        (r"\bhydron[ée]phrose\b",                                                    "N28", "Hydronéphrose"),
        (r"\bobstruction\s+(?:des?\s+)?(?:voies?\s+urinaires?|urinaire|ur[ée]t[ée]rale?)\b|\buropathie\s+obstructive\b", "N28", "Obstruction des voies urinaires"),

        # S72 - femur (femoral neck)
        (r"\bfracture\s+(?:du\s+)?col(?:\s+(?:du\s+f[ée]mur|f[ée]moral))?(?:\s+(?:droit|gauche))?\b", "S72", "Fracture du col fémoral"),
        (r"\bfracture\s+cervico[- ]?c[ée]phalique(?:\s+(?:droit|gauche))?\b",   "S72", "Fracture du col fémoral"),

        # S09 - head trauma, unspecified (non-severe wording)
        (r"\b(?:traumatisme|trauma)\s+cr[âa]nien\b(?!\s+grave)",                "S09", "Traumatisme crânien"),
        # S06 - intracranial injury
        (r"\b(?:tc\s+grave|(?:tc|traumatisme|trauma)\s+cr[âa]nien\s+grave)\b",  "S06", "Lésion intracrânienne"),
        (r"\bh[ée]morragie\s+sous[- ]arachno[iï]dienne\b",                      "S06", "Lésion intracrânienne"),
        (r"\bh[ée]morragie\s+m[ée]ning[ée]e\b",                                 "S06", "Lésion intracrânienne"),
        (r"\bh[ée]morragie\s+intra[- ]?ventriculaire\b",                        "S06", "Lésion intracrânienne"),
        (r"\bp[ée]t[ée]chie\s+h[ée]morragique\b",                               "S06", "Lésion intracrânienne"),
        (r"\bh[ée]matome\s+(?:sous[- ]dural|extra[- ]dural|[ée]pidural)\b",     "S06", "Lésion intracrânienne"),
        (r"\bcontusion\s+c[ée]r[ée]brale\b",                                    "S06", "Lésion intracrânienne"),
        (r"\bl[ée]sion\s+intracr[âa]nienne\b",                                  "S06", "Lésion intracrânienne"),

        # S32 - lumbar spine / pelvis
        # More flexible S32 lumbar vertebra fracture patterns
        (r"\bfracture(?:s)?(?:\s+de)?(?:\s+la)?(?:\s+vert[èe]bre)?(?:\s+lombaire)?(?:\s+corps)?(?:\s+vert[èe]bral)?(?:\s+de)?\s+[L,l][1-5]\b", "S32", "Fracture de L1-L5 (lombaire)"),
        (r"\bfracture[-\s]?tassement\s+du\s+corps\s+vert[ée]bral\s+de\s+l\d+\b", "S32", "Fracture-tassement du corps vertébral lombaire"),
        (r"\bfracture\s+du\s+corps\s+vert[ée]bral\s+de\s+l\d+\b", "S32", "Fracture du corps vertébral lombaire"),
        (r"\bfracture\s+de\s+[L,l][1-5]\b", "S32", "Fracture de L1-L5 (lombaire)"),
        (r"\bfracture\s+[L,l][1-5]\b", "S32", "Fracture L1-L5 (lombaire)"),
        (r"\bfracture\s+de\s+la\s+branche\s+ischio[- ]pubienne\b", "S32", "Fracture de la branche ischio-pubienne"),
        (r"\bfracture\s+comminutive\s+de\s+la\s+branche\s+ilio[- ]pubienne\b", "S32", "Fracture comminutive de la branche ilio-pubienne"),
        (r"\bfracture\s+de\s+la\s+branche\s+ilio[- ]pubienne\b", "S32", "Fracture de la branche ilio-pubienne"),
        (r"\bfracture\s+articulaire\s+compl[èe]te\s+du\s+cotyle\b", "S32", "Fracture articulaire complète du cotyle"),
        (r"\bfracture\s+du\s+cotyle\b", "S32", "Fracture du cotyle"),

        # S42 - clavicle / scapula / proximal humerus
        (r"\bfracture\s+(?:de\s+l['’])?omoplate(?:\s+(?:gauche|droite))?(?:\s+non\s+d[ée]plac[ée]e?)?\b", "S42", "Fracture de l'omoplate"),
        (r"\bfracture\s+(?:de\s+la\s+)?scapula(?:\s+(?:gauche|droite))?(?:\s+non\s+d[ée]plac[ée]e?)?\b", "S42", "Fracture de la scapula"),
        (r"\bfracture\s+(?:de\s+la\s+)?clavicule(?:\s+(?:droite|gauche))?\b",      "S42", "Fracture de la clavicule"),
        (r"\bfracture\s+articulaire\s+de\s+la\s+gl[èe]ne\s+scapulaire\b",         "S42", "Fracture de la glène scapulaire"),
        (r"\bfracture\s+de\s+la\s+gl[èe]ne\s+scapulaire\b",                        "S42", "Fracture de la glène scapulaire"),
        (r"\bfracture\s+de\s+l['’]extr[ée]mit[ée]\s+sup[ée]rieure\s+de\s+l['’]hum[ée]rus\b", "S42", "Fracture de l'extrémité supérieure de l'humérus"),
        (r"\bfracture\s+de\s+l['’]hum[ée]rus\s+proximal\b",                        "S42", "Fracture de l'humérus proximal"),

        # Respiratory
        (r"\bBPCO\b|\bbronchopneumopathie\s+chronique\s+obstructive\b", "J44", "BPCO"),
        (r"\b(insuffisance\s+respiratoire(?:\s+aigu[ëe])?|d[ée]tresse\s+respiratoire|arr[eê]t\s+respiratoire|balancement\s+thoraco[- ]abdominal)\b", "J96", "Insuffisance respiratoire"),
        (r"(?:\b(?:intubation\s+oro[- ]trach[ée]ale|intub(?:é|ée|ation)|ventilation\s+m[ée]canique)\b.{0,60}\b(?:insuffisance\s+respiratoire(?:\s+aigu[ëe])?|d[ée]tresse\s+respiratoire|d[ée]saturation|polypn[ée]ique|hypox[ée]mie)\b|\b(?:insuffisance\s+respiratoire(?:\s+aigu[ëe])?|d[ée]tresse\s+respiratoire|d[ée]saturation|polypn[ée]ique|hypox[ée]mie)\b.{0,60}\b(?:intubation\s+oro[- ]trach[ée]ale|intub(?:é|ée|ation)|ventilation\s+m[ée]canique)\b)","J96","Insuffisance respiratoire"),
        (r"(?:\b(?:vni|ventilation\s+non[- ]invasive|ventilation\s+assist[ée]e|vac)\b.{0,60}\b(?:insuffisance\s+respiratoire(?:\s+aigu[ëe])?|d[ée]tresse\s+respiratoire|d[ée]saturation|polypn[ée]ique|hypox[ée]mie)\b|\b(?:insuffisance\s+respiratoire(?:\s+aigu[ëe])?|d[ée]tresse\s+respiratoire|d[ée]saturation|polypn[ée]ique|hypox[ée]mie)\b.{0,60}\b(?:vni|ventilation\s+non[- ]invasive|ventilation\s+assist[ée]e|vac)\b)","J96","Insuffisance respiratoire"),
        (r"\bpneumopathie\s+d['’]inhalation\b|\bpneumonie\s+d['’]inhalation\b",      "J69", "Pneumopathie d'inhalation"),
        (
            r"(?:\b(?:antibioth[ée]rapie|antibiotique(?:s)?)\b.{0,160}\b(?:vis[ée]e?\s+respiratoire|respiratoire)\b.{0,160}\bfoyer\s+auscultatoire\b|"
            r"\bfoyer\s+auscultatoire\b.{0,160}\b(?:antibioth[ée]rapie|antibiotique(?:s)?)\b.{0,160}\b(?:vis[ée]e?\s+respiratoire|respiratoire)\b|"
            r"\bfoyer\s+auscultatoire\b.{0,120}\b(?:vis[ée]e?\s+respiratoire|respiratoire)\b|"
            r"\b(?:vis[ée]e?\s+respiratoire|respiratoire)\b.{0,120}\bfoyer\s+auscultatoire\b)", "J18", "Pneumopathie"),
        (
            r"(?:\binfection\s+pulmonaire\b.{0,80}\b(?:sous|trait[ée]e?\s+par|ttt\s+par)\b.{0,40}\b(?:rocephine|ceftriaxone|antibioth[ée]rapie|antibiotique(?:s)?)\b|"
            r"\b(?:rocephine|ceftriaxone|antibioth[ée]rapie|antibiotique(?:s)?)\b.{0,80}\binfection\s+pulmonaire\b)", "J18", "Pneumopathie"),
        # Specific: inhalation/aspiration pneumonia (J69) must come before generic J18
        # Robust: match 'pneumopathie ... inhalation', 'pneumopathie ... suite à une inhalation', etc.
        (r"(\bpneumopathie\b|\bpneumonie\b)[^\n]{0,80}?(inhalation|aspiration|suite\s+[aà]\s+une?\s+(inhalation|aspiration))|"
         r"(inhalation|aspiration)[^\n]{0,80}?(\bpneumopathie\b|\bpneumonie\b)", "J69", "Pneumopathie d'inhalation/aspiration"),
        (r"\bpneumopathie\b|\bpneumonie\b",                                            "J18", "Pneumopathie"),
        # Metabolic / nutritional
        (r"\bd[ée]shydrat(?:ation|[ée]e?)\b|\br[ée]hydrat(?:ation|er|[ée]e?)\b|\bhydratation\s*\+{2,}\b", "E86", "Déshydratation"),
        (r"\bd[ée]nutrition\s+s[ée]v[èe]re\b",                                       "E43", "Dénutrition sévère"),
        (r"\bd[ée]nutrition\s+mod[ée]r[ée]e\b",                                      "E44", "Dénutrition modérée"),
        (r"\bhyperkali[ée]mie\b",                                                    "E87", "Hyperkaliémie"),
        (r"\bhypokali[ée]mie\b",                                                     "E87", "Hypokaliémie"),
        (r"\bhyponat[ée]mie\b",                                                      "E87", "Hyponatrémie"),
        (r"\bhypernat[ée]mie\b",                                                     "E87", "Hypernatrémie"),
        (r"\bhypocalc[ée]mie\b",                                                     "E83", "Hypocalcémie"),
        (r"\bhypercalc[ée]mie\b",                                                    "E83", "Hypercalcémie"),
        (r"\bhypoglyc[ée]mie\b",                                                     "E16", "Hypoglycémie"),
        # Haematological
        (r"\ban[ée]mie\b",                                                           "D64", "Anémie"),
        (r"\b(d[ée]globulisation)\b",                                                "D64", "Anémie"),
        (r"\b(chute|baisse)\s+d['’]h[ée]moglobine\b",                                "D64", "Anémie"),
        # Infectious
        (r"\bsepsis\b|\bchoc\s+septique\b",                                          "A41", "Sepsis / Choc septique"),
        (r"\bpy[ée]l[oo]?n[ée]phrite\b(?:\s+aigu[ëe])?",                             "N10", "Pyélonéphrite aiguë"),
        (r"\b(?:infection\s+urinaire|i\.?\s*u\.?|cystite)\b(?:\s+[àa]\s+[a-z0-9\.-]{1,30})?", "N39", "Infection urinaire"),
        # Neurological
        (r"\b(?:d[ée]lirium|syndrome\s+confusionnel(?:\s+aigu)?|[ée]tat\s+confusionnel(?:\s+aigu)?|confusion\s+aigu[ëe])\b", "F05", "Confusion / Syndrome confusionnel"),

        # Functional mobility disorders often missed by UMLS
        (r"\bdifficult[ée]\s+[àa]\s+la\s+marche\b|\btroubles?\s+de\s+la\s+marche\b|\btroubles?\s+de\s+l['’]?[ée]quilibre\b|\binstabilit[ée]\s+posturale\b|\bmarche\b.{0,30}\bdifficilement\b|\bmarche\s+(?:tr[èe]s\s+)?difficile(?:ment)?\b|\b(?:[ée]tat\s+)?grabataire\b|\b(?:reprise\s+de\s+la\s+marche|reprend\s+la\s+marche)\s+[àa]\s+l['’]aide\s+d['’]un\s+d[ée]ambulateur\b|\bmarche\s+avec\s+(?:un\s+)?d[ée]ambulateur\b|\bimpossibilit[ée]\s+de\s+se\s+d[ée]placer(?:\s+dans\s+son\s+propre\s+domicile)?\b|\bincapacit[ée]\s+[àa]\s+se\s+d[ée]placer(?:\s+(?:au\s+domicile|[àa]\s+domicile|dans\s+son\s+domicile|chez\s+(?:elle|lui)))?\b|\bne\s+peut\s+plus\s+se\s+d[ée]placer(?:\s+(?:au\s+domicile|[àa]\s+domicile|dans\s+son\s+domicile|chez\s+(?:elle|lui)))?\b|\br[ée]-?autonomisation\b.{0,40}\b(?:marche|transferts?)\b", "R26", "Difficulté à la marche / mobilité réduite"),
        (r"\bretrouv(?:e|é|ée)\s+au\s+sol\b|\bchutes?\s+s['’]?acc[ée]l[èe]rent\b",  "W19", "Chute accidentelle"),
        (r"\b(?:r[ée]flexe\s+de\s+)?grasping\b|\bsigne\s+de\s+pr[ée]hension\b",     "R29", "Signes neurologiques et ostéomusculaires"),
        (r"\bchutes?\s+[àa]\s+r[ée]p[ée]tition\b|\bchuteur\s+chronique\b",          "R29", "Chutes à répétition"),
        # Bacterial agent (context required)
        (r"\benterobacter\s+cloacae\b|\be\.?\s*cloacae\b",                          "B96", "Enterobacter cloacae comme agent infectieux"),
        (r"\bescherichia\s+coli\b|\be\.?\s*coli\b",                                 "B96", "Escherichia coli comme agent infectieux"),
        (r"\bklebsiella\b",                                                         "B96", "Klebsiella comme agent infectieux"),
        (r"\benterocoque\b|\benterococcus\b",                                       "B95", "Enterococcus comme agent infectieux"),
        (r"\b(?:staphyloco(?:c|que)|staphylococcus|s\.?\s*aureus|mssa|sasm|sams|mrsa|sarm)\b", "B95", "Staphylococcus comme agent infectieux"),
        (r"\bstreptoco(c|que)\b|\bstreptococcus\b",                                 "B95", "Streptococcus comme agent infectieux"),
        # Acute ischemic stroke
        # I63, I69 - acute ischemic stroke (radiology/report wording)
        (r"\b(?:avc\s+(?:isch[ée]mique|r[ée]cent)|aic|accident\s+isch[ée]mique\s+c[ée]r[ée]bral)\b|\b(?:infarctus|isch[ée]mie|hypodensit[ée])\b.{0,80}\bcapsulo[- ]?lenticulaire\b|\bocclusion\s+de\s+m1\b|\bm1\b.{0,30}\bocclu(?:sion|[eé])\b", "I63", "AVC ischémique"),
        (r"accident (c[ée]r[ée]bral|vasculaire) isch[ée]mique", "I63", "Accident vasculaire cérébral ischémique"),
        (r"\bAVC isch[ée]mique\b", "I63", "Accident vasculaire cérébral ischémique"),
        (r"\bisch[ée]mie sylvienne\b", "I63", "Accident vasculaire cérébral ischémique"),
        (r"\bthrombectomie.*isch[ée]mique", "I63", "Accident vasculaire cérébral ischémique"),
        (r"\binfarctus.*c[ée]r[ée]bral", "I63", "Accident vasculaire cérébral ischémique"),
        # I69 - sequelae of cerebrovascular disease (stroke history/sequelae)    
        (r"\bavc\s+(?:ancien|ant[ée]c[ée]dent|en\s+\d{4})\b|\bancien\s+avc\b|\bant[ée]c[ée]dent(?:s)?\s+d['’]?avc\b|\batcd\s*[:\-]?\s*avc\b|\batcd\b.{0,20}\bavc\b", "I69", "Séquelles AVC"),

        #F10 - Alcohol-related disorders (broader, includes chronic alcohol use, dependence, etc.)
        (r"\b(?:oh\b.{0,20}\b(?:chronique|non\s+sevr[ée]?|sevrage)|alcool(?:isme|o[- ]d[ée]pendance)?|[ée]thylisme|exog[ée]nose)(?:\s+(?:chronique|aigu[ëe]?|active|non\s+sevr[ée]?|non\s+sevr[eé]))*\b", "F10", "Troubles liés à l'alcool"),
        # Old stroke / stroke history → sequelae
        (r"\bavc\s+(?:ancien|ant[ée]c[ée]dent|en\s+\d{4})\b|\bancien\s+avc\b|\bant[ée]c[ée]dent(?:s)?\s+d['’]?avc\b|\batcd\s*[:\-]?\s*avc\b|\batcd\b.{0,20}\bavc\b|\bs[ée]quelles?\b.{0,30}\bavc\b|\bavc\b.{0,30}\bs[ée]quelles?\b|\bs[ée]quellaire\b.{0,30}\bavc\b|\b(?:depuis|post[-\s]?|apr[èe]s)\s+avc\b", "I69", "Séquelles AVC"),
        # Cardiac rhythm
        (r"\btachycardie\b|\btachycardique\b",                                       "R00", "Tachycardie"),
        (r"\bbradycardie\b|\bbradycardique\b",                                       "R00", "Bradycardie"),
        
    
    ]

    # Global context flags
    has_alzheimer = bool(re.search(r"\balzheimer\b", normalized))
    has_infection = has_infectious_context_local(seen_codes)
    has_hemorrhagic_ctx = has_hemorrhagic_context(text)
    has_shock_ctx = has_shock_context(text)
    has_rhabdo_ctx = has_rhabdo_context(text)
    has_urinary_ctx = has_urinary_context(text)

    # --- S06/S09 context logic: if S09 present and hemorrhagic context, replace S09 with S06 ---
    """
    s09_idx = next((i for i, item in enumerate(results) if isinstance(item, dict) and item.get("code") == "S09"), None)
    if s09_idx is not None and has_hemorrhagic_ctx:
        # Replace S09 with S06
        results[s09_idx] = {"problem": "Lésion intracrânienne (trauma crânien + contexte hémorragique)", "code": "S06"}
        seen_codes.discard("S09")
        seen_codes.add("S06")
    """
    for pattern, code, label in rules:
        for m in re.finditer(pattern, normalized):
            if is_negated_for_code(m.start(), m.end(), code):
                continue
            actual_code = code
            if code == "R33":
                # Only convert BU-Sang to true hematuria if urinary context exists
                # and there is no strong rhabdomyolysis/myoglobinuria context
                if not has_urinary_ctx or has_rhabdo_ctx:
                    continue
                actual_code = "R31"
            if code == "F03" and re.search(severe_cognitive_pattern, m.group(0), re.IGNORECASE):
                has_chronic_cognitive_context = any(
                    re.search(pattern, normalized, re.IGNORECASE)
                    for pattern in chronic_cognitive_context_patterns
                )
                has_acute_cognitive_context = any(
                    re.search(pattern, normalized, re.IGNORECASE)
                    for pattern in acute_cognitive_context_patterns
                )
                if has_acute_cognitive_context and not has_chronic_cognitive_context and not has_alzheimer:
                    actual_code = "F05"
            # Alzheimer context adjustments
            # If context changes the code, remove the original from results/seen_codes
            if code == "F00" and not has_alzheimer:
                # Remove F00 if present
                results = [r for r in results if r["code"] != "F00"]
                seen_codes.discard("F00")
                actual_code = "F03"
            if code == "F03" and has_alzheimer:
                # Remove F03 if present
                results = [r for r in results if r["code"] != "F03"]
                seen_codes.discard("F03")
                actual_code = "F00"
            # Context/contradiction logic should always be applied
            if actual_code in {"B95", "B96"} and not has_infection:
                # Remove B95/B96 if present
                results = [r for r in results if r["code"] != actual_code]
                seen_codes.discard(actual_code)
                continue
            if actual_code == "D64" and has_hemorrhagic_ctx:
                # Remove D64 if present
                results = [r for r in results if r["code"] != "D64"]
                #print(f"Discarding D64 code due to hemorrhagic context")
                seen_codes.discard("D64")
                continue
            # E83 context adjustments: if corrected calcium is normal or phosphocalcic workup is normal, suppress E83 code    
            if actual_code == "E83":
                corr = re.search(
                    r"calc[ée]mie\s+corrig[ée]e?(?:\s*[:=]?\s*|de\s+)(\d{1,2}(?:[.,]\d{1,2})?)",
                    normalized
                )
                if corr:
                    corr_val = float(corr.group(1).replace(",", "."))
                    if 2.2 <= corr_val < 2.6:
                        continue
                if re.search(r"\bbilan\s+phosphocalcique\s+normal\b", normalized):
                    continue            
            if actual_code not in seen_codes:
                #print(f"Adding code {actual_code}")
                results.append({"problem": label, "code": actual_code})
                seen_codes.add(actual_code)
            #print(f"Matched pattern '{pattern}' for code '{code}' with label '{label}' at position {m.start()}-{m.end()}")
            break  # one match per rule is enough

    # Upstream entities may already inject R33 in existing_codes. Drop it when
    # context only contains rule-out/negated urinary retention mentions.
    if any(extract_code_val(r["code"]) == "R33" for r in results):
        r33_positive_patterns = [
            r"\br[ée]tention\s+(?:d['’]urine|urinaire)\b",
            r"\bglobe\s+v[ée]sical\b",
            r"\bsondage\s+(?:v[ée]sical|urinaire|[àa]\s+demeure)\b",
            r"\bbladder(?:\s+scan)?\b[^0-9]{0,20}(?:[3-9][0-9]{2}|[1-9][0-9]{3,})\s*(?:ml|cc)\b",
            r"\br[ée]sidu\s+post[- ]?mictionnel\b[^0-9]{0,20}(?:[3-9][0-9]{2}|[1-9][0-9]{3,})\s*(?:ml|cc)\b",
        ]
        has_non_negated_r33_signal = False
        for p in r33_positive_patterns:
            for m in re.finditer(p, normalized):
                if not is_negated_for_code(m.start(), m.end(), "R33"):
                    has_non_negated_r33_signal = True
                    break
            if has_non_negated_r33_signal:
                break

        has_r33_ruleout_signal = any(re.search(p, normalized) for p in r33_negation_patterns)
        if has_r33_ruleout_signal and not has_non_negated_r33_signal:
            results = [r for r in results if extract_code_val(r["code"]) != "R33"]

    
    present_codes = {extract_code_val(r["code"]) for r in results}

    # User rule: if G30 is present and severe context is detected, add F00.
    if "G30" in present_codes and severe_context(text) and "F00" not in present_codes:
        results.append({"problem": "Troubles neurocognitifs majeurs (contexte sévère)", "code": "F00"})
        present_codes.add("F00")

    # Remove R79 when a more specific explanatory code is already present
    if "R79" in present_codes and any(c in present_codes for c in {"N39", "E87", "A41", "J18", "N17", "J69"}):
        results = [r for r in results if extract_code_val(r["code"]) != "R79"]
        
    if "N18" in present_codes:
        results = [r for r in results if extract_code_val(r["code"]) != "N19"]
    
    if "N17" in present_codes and "N19" in present_codes:
        results = [r for r in results if extract_code_val(r["code"]) != "N19"]

    # Prefer specific scalp wound over generic unspecified injury.
    if "S01" in present_codes and "T14" in present_codes:
        results = [r for r in results if extract_code_val(r["code"]) != "T14"]
    
    if "G41" in present_codes:
        results = [r for r in results if extract_code_val(r["code"]) not in {"R40", "R41", "R47"}]

    if any(code in present_codes for code in {"F00", "F01", "F03"}):
        results = [r for r in results if extract_code_val(r["code"]) != "R41"]
    if "F05" in present_codes:
        results = [r for r in results if extract_code_val(r["code"]) not in {"R41", "R44", "R45"}]
    
    # F03 ->  F01 if vascular dementia context, but only if no explicit Alzheimer context (G30 or "alzheimer" term) is present. If Alzheimer context is present, F00 is already assigned and F03 should be removed.
    if re.search(r"\bd[ée]mence\s+(?:probablement\s+|de\s+type\s+)?vasculaire\b",normalized, flags=re.IGNORECASE):
        results = [r for r in results if extract_code_val(r["code"]) != "F03"]
        if not any(extract_code_val(r["code"]) == "F01" for r in results):
            results.append({"problem": "Démence vasculaire", "code": "F01"})
    
    if re.search(r"\bd[ée]pression\s+de\s+trac[ée]\b|\bd[ée]pression\b.{0,20}\btrac[ée]\b|\btrac[ée]\b.{0,20}\bd[ée]pression\b", normalized, re.IGNORECASE):
        results = [r for r in results if extract_code_val(r["code"]) not in {"F33", "F32"}]
        
    if "F33" in present_codes:
        recurrent_depression_pattern = (
            r"\br[ée]current(?:e|es|s)?\b|"
            r"\bchronique\b|"
            r"\b[ée]pisodes?\s+r[ée]p[ée]t[ée]s?\b|"
            r"\bant[ée]c[ée]dents?\s+de\s+d[ée]pression\b"
        )
        if not re.search(recurrent_depression_pattern, normalized, re.IGNORECASE):
            # If recurrent/chronic context is absent, recode F33 -> F32.
            results = [r for r in results if extract_code_val(r["code"]) != "F33"]
            if not any(extract_code_val(r["code"]) == "F32" for r in results):
                results.append({"problem": "Dépression", "code": "F32"})
    # F33 is not used in the local 109/HFRS setup, normalize to F32 to keep depression signal.
    if any(extract_code_val(r["code"]) == "F33" for r in results):
        for r in results:
            if extract_code_val(r["code"]) == "F33":
                r["code"] = "F32"
                if r.get("problem") in {"", None}:
                    r["problem"] = "Dépression"
    # If both J18 and J69 are present, J18 will be removed, and only J69 will remain.
    # If only one is present, it is kept
    if "J69" in {extract_code_val(r["code"]) for r in results}:
        results = [r for r in results if extract_code_val(r["code"]) != "J18"]
    if "J18" in present_codes:
        if re.search(r"\bpneumopathie\b[^\n]{0,100}\b(?:inhalation|aspiration)\b", normalized, re.I) \
        or re.search(r"\b(?:inhalation|aspiration)\b[^\n]{0,100}\bpneumopathie\b", normalized, re.I):
            results = [r for r in results if extract_code_val(r["code"]) != "J18"]
            if "J69" not in present_codes:
                results.append({"code": "J69", "label": "Pneumopathie d'inhalation/aspiration"})
    
    # If both J80 (SDRA) and J96 (Insuffisance respiratoire) are present, remove J96 (prefer specific SDRA code)
    if "J80" in present_codes and "J96" in present_codes:
        results = [r for r in results if extract_code_val(r["code"]) != "J96"]
    if "S32" in present_codes and re.search(
        r"\b(?:fracture\s+vert[ée]brale\s+de\s+fatigue|fracture\s+d['’]insuffisance|fracture\s+de\s+fragilit[ée]|tassement\s+vert[ée]bral.{0,20}(?:ost[ée]oporotique|de\s+fatigue|d['’]insuffisance))\b",
        normalized,
        re.I,
    ):
        results = [r for r in results if extract_code_val(r["code"]) != "S32"]
        if "M48" not in {extract_code_val(r["code"]) for r in results}:
            results.append({"code": "M48", "label": "Fracture vertébrale de fatigue/insuffisance"})
    if has_stroke_sequelae_context():
        if not has_acute_stroke_context():
            results = [r for r in results if extract_code_val(r["code"]) != "I63"]
        if not any(extract_code_val(r["code"]) == "I69" for r in results):
            results.append({"problem": "Séquelles AVC", "code": "I69"})
    
    if has_shock_ctx:
        results = [r for r in results if extract_code_val(r["code"]) != "I95"]
    if has_hemorrhagic_ctx:
        results = [r for r in results if r["code"] != "D64"]

    # Prefer recurrent/chronic falls coding over isolated accidental fall,
    # but keep both if an explicit acute fall event is also documented.
    repeated_falls_context = bool(
        re.search(r"\bchutes?\s+[àa]\s+r[ée]p[ée]tition\b|\bchuteur\s+chronique\b", normalized)
    )
    acute_fall_context = bool(
        re.search(
            r"\bretrouv(?:e|é|ée)\s+au\s+sol\b|\bchute\s+accidentelle\b|\bchute\s+(?:avec|suite\s+[àa])\b|\bchute\s+ce\s+jour\b",
            normalized,
        )
    )
    if repeated_falls_context and not acute_fall_context and any(extract_code_val(r["code"]) == "R29" for r in results):
        results = [r for r in results if extract_code_val(r["code"]) != "W19"]
       
        
    # Deduplicate by code (keep first occurrence)
    deduped = []
    seen = set()
    for r in results:
        code_val = extract_code_val(r["code"])
        if code_val not in seen:
            deduped.append(r)
            seen.add(code_val)
    results = deduped

    # Filter to only codes in the 109-code list
    # Load 109-code list from file (or cache if needed)
    if not hasattr(scan_full_text_rule_based_codes, "_cim_109_codes"):
        import csv
        with open("/home/coder/nalfe/data/cim_109_code.csv", encoding="utf-8") as f:
            reader = csv.reader(f)
            scan_full_text_rule_based_codes._cim_109_codes = set(row[0][:3] for row in reader if row)
    valid_codes = scan_full_text_rule_based_codes._cim_109_codes
    results = [r for r in results if extract_code_val(r["code"]) in valid_codes]
    
    return results

def save_to_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False, encoding='utf-8')

def save_to_json(df: pd.DataFrame, path: str):
    df.to_json(path, orient='records', force_ascii=False, indent=2)

def load_cim10_mapping(path: str) -> dict:
    """
    Loads a CIM-10 mapping file. If the file is a combined mapping (list of dicts with 'term' and 'code'),
    returns a dict mapping term to code. If the file is a simple dict, returns as is.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # If data is a list of dicts (combined mapping), convert to term->code dict
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'term' in data[0] and 'code' in data[0]:
        mapping = {}
        for entry in data:
            term = entry['term']
            code = entry['code']
            mapping[term] = code
        return mapping
    return data
