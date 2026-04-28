"""
Monkey-patch QuickUMLS to use a minimal spaCy French pipeline
to avoid 'tok2vec' errors when loading fr_core_news_sm.
"""

# Set QuickUMLS environment variables before any imports
import os
quickumls_home = os.path.abspath("quickumls_installs")
quickumls_folder = "2025AA_fre_lower"
os.environ["QUICKUMLS_HOME"] = quickumls_home
os.environ["QUICKUMLS_FRE_2025AA"] = quickumls_folder

import quickumls
import spacy
if not hasattr(quickumls, '_patched_fr_nlp'):
    nlp_fr = spacy.blank("fr")
    if "sentencizer" not in nlp_fr.pipe_names:
        nlp_fr.add_pipe("sentencizer")
    quickumls._patched_fr_nlp = nlp_fr
    orig_spacy_load = spacy.load
    def patched_spacy_load(name, *args, **kwargs):
        if name in ["fr", "fr_core_news_sm"]:
            return quickumls._patched_fr_nlp
        return orig_spacy_load(name, *args, **kwargs)
    spacy.load = patched_spacy_load
    
from medkit.core.text import TextDocument
from medkit.text.segmentation.section_tokenizer import SectionTokenizer
from medkit.text.segmentation.sentence_tokenizer import SentenceTokenizer
from medkit.text.preprocessing.char_replacer import CharReplacer
from medkit.text.context.negation_detector import NegationDetector
from medkit.text.ner.hf_entity_matcher import HFEntityMatcher
from medkit.text.ner.quick_umls_matcher import QuickUMLSMatcher


import pandas as pd
import json
import re
from typing import List, Dict, Union
# Import Z-code detection and clean_text from utils
from .utils import extract_z_codes_hybrid
from .utils import clean_text
from .utils import rule_based_diagnosis_codes
from .utils import scan_full_text_rule_based_codes
# Import enrichment function for clinically-significant findings detection
from clinical_fragility.helper_function import enrich_text_with_clinically_significant_findings, detect_sex



# Force GPU usage for torch/transformers
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
    if torch.cuda.is_available():
        torch_device = torch.device('cuda:0')
    else:
        raise RuntimeError('CUDA GPU is not available!')
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Optional: For local LLM (Mistral/Qwen) via text-generation-inference or similar
try:
    from transformers import AutoModelForCausalLM, TextGenerationPipeline
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class ClinicalNoteProcessor:
    @staticmethod
    def extract_z_codes_from_hybrid(z_codes_hybrid, cim_109_codes=None):
        # z_codes_hybrid is a list of dicts: {"code": Zxx, "justification": ...}
        import pandas as pd
        if z_codes_hybrid is None or (isinstance(z_codes_hybrid, float) and pd.isna(z_codes_hybrid)):
            return []
        if not isinstance(z_codes_hybrid, list):
            return []
        if cim_109_codes is not None:
            return [item["code"] for item in z_codes_hybrid if isinstance(item, dict) and "code" in item and item["code"][:3] in cim_109_codes]
        else:
            return [item["code"] for item in z_codes_hybrid if isinstance(item, dict) and "code" in item]
    import re

    def normalize_text_for_umls(self, text: str) -> str:
        """
        Normalize the entire cleaned note by replacing known clinical variants and synonyms
        with their canonical forms, while avoiding replacements inside negated contexts.
        """
        if not text:
            return ""

        negation_patterns = [
            r"\babsence\b",
            r"\babsent(?:e)?\b",
            r"\bpas\s+d[e']",
            r"\baucun(?:e)?\b",
            r"\bsans\b",
            r"\bnon\b",
            r"\bn[ée]gatif(?:ve)?\b",
            r"\babsence\s+de\s+signe(?:s)?\s+de\b",
            r"\babsence\s+de\s+signes?\s+de\b",
            r"\bsans\s+signe(?:s)?\s+de\b",
        ]

        def is_negated_in_context(full_text: str, start: int, end: int) -> bool:
            window = full_text[max(0, start - 80): min(len(full_text), end + 40)]
            return any(re.search(pattern, window, flags=re.IGNORECASE) for pattern in negation_patterns)

        canonical_rules = [
            # IRC abbreviation normalization
            (r"\birc\b", "Insuffisance rénale chronique (IRC)"),
            # COVID-19 normalization (sym:normalize_text_for_umls)
            (r"\bcovid(?:[\s\-]?19)?\b", "COVID-19"),
            # Chute normalization (sym:normalize_text_for_umls)
            (r"\bchut[ée]s?\b", "chute"),
            (r"\bchuter\b", "chute"),
            # Nutrition / hydration
            (r"\bsyndrome\s+de\s+d[ée]nutrition\b|\bd[ée]nutrition\b|\bmalnutrition\b|\bcachexie\b", "Malnutrition"),
            (r"\bd[ée]shydrat(?:ation(?:\s+clinique)?|[ée]e?)\b|\br[ée]hydrat(?:ation|er|[ée]e?)\b|\bhydratation\s*\+{2,}\b", "Déshydratation"),

            # Renal
            (r"\binsuffisance\s+r[ée]nale\s+chronique\s+terminale\b", "Insuffisance rénale chronique terminale"),
            (r"\binsuffisance\s+r[ée]nale?\s+aigu[ëe]\b", "Insuffisance rénale aiguë"),
            (r"\binsuffisance\s+r[ée]nale?\s+chronique\b", "Insuffisance rénale chronique"),
            (r"\br[ée]tention\s+azot[ée]e\b", "Insuffisance rénale chronique"),

            # Electrolytes / biology
            (r"\bhyponatr[ée]mie\b", "Hyponatrémie"),
            (r"\bhypernatr[ée]mie\b", "Hypernatrémie"),
            (r"\bhypokali[ée]mie\b", "Hypokaliémie"),
            (r"\bhyperkali[ée]mie\b", "Hyperkaliémie"),
            (r"\ban[ée]mie\b|\bd[ée]globulisation\b", "Anémie"),
            (r"\bcrp\s+[ée]lev[ée]e\b|\bsyndrome\s+inflammatoire\b", "Syndrome inflammatoire"),
            (r"\bcarence\s+en\s+vitamine\s+d\b|\bd[ée]ficit\s+en\s+vitamine\s+d\b", "Déficit en vitamine D"),

            # Respiratory
            (r"\bbpco\b", "Bronchopneumopathie chronique obstructive (BPCO)"),
            (r"\b(insuffisance\s+respiratoire(?:\s+aigu[ëe])?|d[ée]tresse\s+respiratoire|arr[eê]t\s+respiratoire)\b", "Insuffisance respiratoire"),
            (r"\b(pneumopathie|pneumonie)\s+d['’](?:inhalation|aspiration)\b|\bpneumopathie\s+sur\s+fausse\s+route\b", "Pneumopathie d'inhalation"),
            (r"\bpneumopathie\s+bas(?:e|ale)\s+(droite|gauche)\b", "Pneumopathie"),
            (r"\bpneumopathie\b|\bpneumonie\b", "Pneumopathie"),
            (r"\bdyspn[ée]e\b", "Dyspnée"),

            # Neuro / cognition
            (r"\balz[a-z]*heimer\b.*\bd[ée]butant[e]?\b", "Maladie d'Alzheimer débutante"),
            (r"\balz[a-z]*heimer\b", "Maladie d'Alzheimer"),
            (r"\balz[a-z]{3,}mer\b.*\bdebutant[e]?\b", "Maladie d'Alzheimer débutante"),
            (r"\bd[ée]mence\s+[àa]\s+corps\s+de\s+l[ée]w?y\b", "Démence à corps de Lewy"),
            (r"\bd[ée]mence\s+vasculaire\b", "Démence vasculaire"),
            (r"\btroubles?\s+cognitifs?\s+s[ée]v[èe]res?\b|\btrouble\s+neurocognitif\b", "Troubles cognitifs sévères"),
            (r"\bconfusion\b|\bconfusion\s+mentale\b", "Confusion"),
            (r"\bobnubilation\b|\bsomnolence\b|\bcoma\b|\bcomateux\b", "Trouble de la conscience"),

            # Urinary / digestive
            (r"\bglobe\s+v[ée]sical\b|\br[ée]tention\s+urinaire\b|\br[ée]tention\s+aigu[ëe]?\s+d['’]urines?\b", "Rétention urinaire"),
            (r"\bf[ée]calome\b", "Fécalome"),
            (r"\bocclusion\s+intestinale\b", "Occlusion intestinale"),
            (r"\bconstipation\b", "Constipation"),
            (r"\bh[ée]maturie\b", "Hématurie"),

            # Functional / frailty
            (r"\bdifficult[ée]?\s+[àa]\s+la\s+marche\b|\btrouble\s+de\s+la\s+marche\b", "Trouble de la marche"),
            (r"\bchutes?\s+[àa]\s+r[ée]p[ée]tition\b|\bchuteur\s+chronique\b", "Chutes à répétition"),
            (r"\bgrabataire\b|\b[ée]tat\s+grabataire\b", "État grabataire"),
            (r"\bescarre\b", "Escarre"),

            # Cardiometabolic
            (r"\bhta\b|\bhypertension\s+art[ée]rielle\b", "Hypertension artérielle"),
            (r"\bdiab[èe]te\s+de\s+type\s+2\b|\bdnid\b", "Diabète de type 2"),
            (r"\bdiab[èe]te\s+de\s+type\s+1\b", "Diabète de type 1"),
            (r"\bdyslipid[ée]mie\b", "Dyslipidémie"),
        ]

        normalized_text = text

        for pattern, replacement in canonical_rules:
            compiled = re.compile(pattern, flags=re.IGNORECASE)

            def replace_if_not_negated(match):
                if is_negated_in_context(normalized_text, match.start(), match.end()):
                    return match.group(0)
                return replacement

            normalized_text = compiled.sub(replace_if_not_negated, normalized_text)

        return normalized_text

    def select_best_cim10_code(self, sentence, disorder_span, quickumls_codes):
        """
        Select the best CIM-10 code for a disorder detected by QuickUMLS.
        - sentence: the full sentence text
        - disorder_span: (start, end) character indices of the detected disorder in the sentence
        - quickumls_codes: list of codes returned by QuickUMLS (already expanded/filtered)
        Uses classifier, mapping, and context window (word before and after the detected disorder).
        """
        # Load valid codes from file (cache at class level)
        if not hasattr(self, '_valid_cim10_codes'):
            code_path = '/home/coder/nalfe/data/cim_109_code.csv'
            with open(code_path, encoding='utf-8') as f:
                codes = [line.strip() for line in f if line.strip() and not line.lower().startswith('code')]
            self._valid_cim10_codes = set(codes)
        valid_codes = self._valid_cim10_codes

        # Load mapping (cache at class level)
        if not hasattr(self, '_cim10_mapping_lower'):
            mapping_path = '/home/coder/nalfe/data/cim10_mapping_109_filtered.json'
            import json
            with open(mapping_path, encoding='utf-8') as f:
                mapping = json.load(f)
            mapping_lower = {}
            for k, v in mapping.items():
                mapping_lower[k.strip().lower()] = v if isinstance(v, list) else [v]
            self._cim10_mapping_lower = mapping_lower
        mapping = self._cim10_mapping_lower

        # Extract context window: word before, disorder, word after
        start, end = disorder_span
        words = sentence.split()
        # Find the word indices for the span
        char_idx = 0
        disorder_word_idx = None
        for i, word in enumerate(words):
            idx = sentence.find(word, char_idx)
            if idx <= start < idx + len(word):
                disorder_word_idx = i
                break
            char_idx = idx + len(word)
        context_terms = set()
        if disorder_word_idx is not None:
            # Add disorder word
            context_terms.add(words[disorder_word_idx])
            # Add previous word if exists
            if disorder_word_idx > 0:
                context_terms.add(words[disorder_word_idx - 1] + ' ' + words[disorder_word_idx])
            # Add next word if exists
            if disorder_word_idx < len(words) - 1:
                context_terms.add(words[disorder_word_idx] + ' ' + words[disorder_word_idx + 1])
        
        # 1. Try mapping for each context term
        for term in context_terms:
            normalized = term.strip().lower()
            mapping_codes = mapping.get(normalized, [])
            for code in mapping_codes:
                if code in valid_codes:
                    return code

        # 2. QuickUMLS codes (already filtered/expanded)
        for code in quickumls_codes:
            if code in valid_codes:
                return code

        # 3. Fallback: return None or first available
        return quickumls_codes[0] if quickumls_codes else None
    
    def debug_section_and_entities(self, text: str):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force transformers to use GPU for all pipelines/models
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    def extract_active_diagnoses_llm(self, text, prompt_template=None, max_length=512):
        """
        Use a single local LLM pass to extract clinically relevant diagnoses from clean clinical text.
        Pre-enriches text by detecting clinically-significant findings (lab abnormalities, Glasgow/GCS, etc.)
        so they are included if they have diagnostic significance, rather than blindly excluded.
        Returns a list of diagnoses as strings after lightweight cleanup.
        """
        import requests
        if not text:
            return []

        def _parse_llm_diagnosis_output(raw_output: str) -> List[str]:
            diagnoses_local = []
            for line in (raw_output or "").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue

                # Accept bullets, numbered lists and simple dash-like prefixes.
                stripped = re.sub(r"^\s*(?:[-•*]|\d+[\.)])\s*", "", stripped).strip()
                if not stripped:
                    continue
                normalized = self._normalize_diagnosis_term(stripped)
                # Robustly filter out negated/absent diagnoses even if LLM returns them
                if normalized and not self._is_negated_or_absent_diagnosis(normalized):
                    diagnoses_local.append(normalized)
            return diagnoses_local

        # Detect patient sex for enrichment (affects lab thresholds)
        patient_sex = detect_sex(text)
        
        # Pre-enrich text: detect and annotate clinically-significant findings
        # (lab abnormalities, Glasgow/GCS, potassium recharge, etc.)
        #enriched_text = enrich_text_with_clinically_significant_findings(text, sex=patient_sex)

        if prompt_template is None:
            # General, scalable prompt with stricter electrolyte disorder extraction
            prompt_template_general = (
                "Texte clinique :\n{note}\n\n"
                "Extrais uniquement les diagnostics, troubles ou problèmes médicaux ACTUELS OU CHRONIQUES, AFFIRMÉS et CLINIQUEMENT PERTINENTS mentionnés dans le texte.\n\n"
                "Consigne principale :\n"
                "- garde un intitulé aussi proche que possible des termes utilisés dans le texte original ;\n"
                "- n'interprète pas, ne regroupe pas, ne reformule pas en syndrome plus large ;\n"
                "- ne remplace pas un terme du texte par un diagnostic plus général ou plus causal ;\n"
                "- normalise seulement légèrement l'orthographe ou les variantes évidentes si nécessaire.\n\n"
                "Inclure :\n"
                "- les diagnostics explicitement nommés ;\n"
                "- les troubles ou problèmes médicaux réellement présents ;\n"
                "- les anomalies biologiques codifiables si elles correspondent clairement à un problème médical (ex : hyponatrémie, anémie, hyperkaliémie) ;\n"
                "- les troubles cognitifs, neurologiques, psychiatriques ou fonctionnels ;\n"
                "- les troubles de la marche, la mobilité réduite, les chutes à répétition, l'instabilité posturale si explicitement mentionnés.\n\n"
                "Exclure :\n"
                "- les éléments niés, absents, non confirmés, suspects, à exclure ou résolus ;\n"
                "- les examens normaux, observations non pathologiques, éléments administratifs ;\n"
                "- les actes, traitements, dispositifs, examens, résultats isolés ;\n"
                "- les qualificatifs isolés non diagnostiques (ex : normocytaire, arégénérative).\n\n"
                "Règles importantes :\n"
                "- ne pas inférer un diagnostic non explicitement mentionné ;\n"
                "- ne pas transformer un signe ou une expression clinique en un autre diagnostic si ce diagnostic n'est pas écrit dans le texte ;\n"
                "- conserve plusieurs éléments distincts s'ils sont tous mentionnés ;\n"
                "- garde des termes proches du texte source.\n"
                "- N’extrais pas de diagnostic d’hyponatrémie, hypernatrémie, hypokaliémie, hyperkaliémie, etc., sauf si le terme exact est mentionné dans le texte ou si une valeur biologique anormale est clairement indiquée (par exemple : « natrémie à 120 », « kaliémie à 2,8 », etc.).\n\n"
                "Sortie :\n"
                "- une puce par élément ;\n"
                "- intitulé médical court, en français ;\n"
                "- pas d'explication, pas de justification, pas de code CIM-10 ;\n"
                "- pas de doublons ;\n"
                "- si aucun diagnostic actif : Aucun diagnostic actif.\n\n"
                "Réponds uniquement en français."
            )
           
        prompt_template = prompt_template_general 
        #prompt = prompt_template.format(note=enriched_text)
        prompt = prompt_template.format(note=text)
        # If using Ollama (llm_model_name startswith 'ollama:' or is a known Ollama model)
        if self.llm_model_name and (self.llm_model_name.startswith('ollama:') or self.llm_model_name in ["mistral-small:latest", "mistral:latest", "qwen2.5:72b", "gpt-oss:120b"]):
            # Use Ollama HTTP API
            model = self.llm_model_name.replace('ollama:', '')
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": max_length, "temperature": 0.1, "top_p": 0.9},
                    },
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
                output_text = result.get("response", "")
            except Exception as e:
                raise RuntimeError(f"Ollama API error: {e}")
        elif self.llm_pipeline:
            # Force pipeline to use GPU
            output_text = self.llm_pipeline(prompt, max_length=max_length, do_sample=False, device=self.device.index if self.device.type == 'cuda' else -1)[0]['generated_text']
        else:
            raise RuntimeError("No local LLM pipeline is loaded. Set llm_model_name and ensure LLM_AVAILABLE, or use Ollama.")

        combined = _parse_llm_diagnosis_output(output_text)

        # Deduplicate while preserving order.
        deduped = []
        seen = set()
        for diag in combined:
            key = diag.lower().strip()
            key = re.sub(r"\s*\[[^\]]+\]\s*", "", key)
            key = re.sub(r"\s+", " ", key).strip(" .;,:-\t")
            if key and key not in seen:
                deduped.append(diag)
                seen.add(key)
        return deduped

    def _is_negated_or_absent_diagnosis(self, term: str) -> bool:
        """Return True when a diagnosis string explicitly states absence, negation, or non-confirmation."""
        normalized = re.sub(r"\s+", " ", (term or "").strip().lower())
        if not normalized:
            return True

        negation_patterns = [
            r"\babsence\b",
            r"\babsent(?:e)?\b",
            r"\bpas d['’]",
            r"\baucun(?:e)?\b",
            r"\bsans anomalie\b",
            r"\bnon\s+(?:d[ée]celable|document[ée]e?|confirm[ée]e?|retrouv[ée]e?|visible|objectiv[ée]e?)\b",
            r"\bn[ée]gatif(?:ve)?\b",
        ]
        return any(re.search(pattern, normalized) for pattern in negation_patterns)
    
    def _normalize_diagnosis_term(self, term: str) -> str:
        """
        Improved cleanup/filtering for diagnosis candidates.
        Removes qualifiers like 'probable', 'possible', 'non' at start or end, and filters non-diagnostic findings.
        """
        if not term:
            return ""

        cleaned = re.sub(r"^\s*(?:[-•*]|\d+[\.)])\s*", "", term).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" .;,:-\t")
        if not cleaned:
            return ""

        # Remove wrapping brackets: [Déshydratation]
        cleaned = re.sub(r"^\[(.+)\]$", r"\1", cleaned).strip()

        # Remove trailing bracketed justifications: Anémie [Hb 8.4]
        cleaned = re.sub(r"\s*\[[^\]]+\]\s*$", "", cleaned).strip()

        # Normalize frequent verbose suffixes that create duplicates without adding diagnosis value.
        cleaned = re.sub(r"\s+à\s+la\s+bio$", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s+biologique$", "", cleaned, flags=re.IGNORECASE).strip()

        cleaned = cleaned.strip(" .;,:-\t")
        if not cleaned:
            return ""

        if self._is_negated_or_absent_diagnosis(cleaned):
            return ""

        low = cleaned.lower()

        # Drop isolated descriptive qualifiers
        if re.fullmatch(r"(?:normocytaire|macrocytaire|microcytaire|ar[ée]g[ée]n[ée]rative|hypochrome|normochrome)", low):
            return ""

        # Drop obvious non-diagnostic single labels
        if re.fullmatch(
            r"(?:examen(?:\s+clinique)?|bilan|traitement|dispositif(?:\s+médical)?|consultation(?:\s+d['’]urgence)?)",
            low,
        ):
            return ""

        # Filter truly non-diagnostic normal observations / admin noise
        non_diag_patterns = [
            r"\bconsultation d'urgence\b",
            r"\bn[°o]\s*s[ée]jour\b",
            r"\bipp\b",
            r"\bmurmure\s+v[ée]siculaire\b",
            r"\bvibrations?\s+vocales?\b",
            r"\bpupilles?\s+sym[ée]triques?\b",
            r"\br[ée]ponse\s+aux\s+ordres\s+simples?\b",
            r"\bconscience\s+vigilance\s+normale\b",
            r"\bsym[ée]triques?\s+dans\s+les\s+2\s+champs\b",
            r"\blangue\s+r[ôo]tie\b",
            r"\byeux\s+cireux\b",
            r"\bh[ée]matocrite\s+à\s+[\d\.,]+\s*l/l\b",
            # Filter any permeability findings (normal or non) as non-diagnostic
            r"\bperm[éee]abilit[éee] de l['’]art[èe]re",
        ]
        for pat in non_diag_patterns:
            if re.search(pat, low):
                return ""

        # Canonical normalization rules
        canonical_rules = [
            # Nutrition / hydration
            (r"\bsyndrome\s+de\s+d[ée]nutrition\b|\bd[ée]nutrition\b|\bmalnutrition\b|\bcachexie\b", "Malnutrition"),
            (r"\bd[ée]shydrat(?:ation(?:\s+clinique)?|[ée]e?)\b|\br[ée]hydrat(?:ation|er|[ée]e?)\b|\bhydratation\s*\+{2,}\b", "Déshydratation"),

            # Renal
            (r"\binsuffisance\s+r[ée]nale?\s+aigu[ëe]\b", "Insuffisance rénale aiguë"),
            (r"\binsuffisance\s+r[ée]nale?\s+chronique\b", "Insuffisance rénale chronique"),
            (r"\br[ée]tention\s+azot[ée]e\b", "Insuffisance rénale"),

            # Electrolytes / biology
            (r"\bhyponatr[ée]mie\b", "Hyponatrémie"),
            (r"\bhypernatr[ée]mie\b", "Hypernatrémie"),
            (r"\bhypokali[ée]mie\b", "Hypokaliémie"),
            (r"\bhyperkali[ée]mie\b", "Hyperkaliémie"),
            (r"\ban[ée]mie\b|\bd[ée]globulisation\b", "Anémie"),
            (r"\bcrp\s+[ée]lev[ée]e\b|\bsyndrome\s+inflammatoire\b", "Syndrome inflammatoire"),
            (r"\bcarence\s+en\s+vitamine\s+d\b|\bd[ée]ficit\s+en\s+vitamine\s+d\b", "Déficit en vitamine D"),

            # Respiratory
            (r"\b(insuffisance\s+respiratoire(?:\s+aigu[ëe])?|d[ée]tresse\s+respiratoire|arr[eê]t\s+respiratoire)\b", "Insuffisance respiratoire"),
            (r"\b(pneumopathie|pneumonie)\s+d['’](?:inhalation|aspiration)\b|\bpneumopathie\s+sur\s+fausse\s+route\b", "Pneumopathie d'inhalation"),
            (r"\b(pneumopathie|pneumonie)\b", "Pneumopathie"),
            (r"\bdyspn[ée]e\b", "Dyspnée"),

            # Neuro / cognition
            (r"\balzheimer\b.*\bdebutant[e]?\b", "Maladie d'Alzheimer débutante"),
            (r"\balz[a-z]{3,}mer\b.*\bdebutant[e]?\b", "Maladie d'Alzheimer débutante"),
            (r"\bd[ée]mence\s+[àa]\s+corps\s+de\s+l[ée]w?y\b", "Démence à corps de Lewy"),
            (r"\bd[ée]mence\s+vasculaire\b", "Démence vasculaire"),
            (r"\btroubles?\s+cognitifs?\s+s[ée]v[èe]res?\b|\btrouble\s+neurocognitif\b", "Troubles cognitifs sévères"),
            (r"\bconfusion\b|\bconfusion\s+mentale\b", "Confusion"),
            (r"\bobnubilation\b|\bsomnolence\b|\bcoma\b|\bcomateux\b", "Trouble de la conscience"),

            # Urinary / digestive
            (r"\bglobe\s+v[ée]sical\b|\br[ée]tention\s+urinaire\b|\br[ée]tention\s+aigu[ëe]?\s+d['’]urines?\b", "Rétention urinaire"),
            (r"\bf[ée]calome\b", "Fécalome"),
            (r"\bocclusion\s+intestinale\b", "Occlusion intestinale"),
            (r"\bconstipation\b", "Constipation"),
            (r"\bh[ée]maturie\b", "Hématurie"),

            # Functional / frailty
            (r"\bdifficult[ée]?\s+[àa]\s+la\s+marche\b|\btrouble\s+de\s+la\s+marche\b", "Trouble de la marche"),
            (r"\bchutes?\s+[àa]\s+r[ée]p[ée]tition\b|\bchuteur\s+chronique\b", "Chutes à répétition"),
            (r"\bgrabataire\b|\b[ée]tat\s+grabataire\b", "État grabataire"),
            (r"\bescarre\b", "Escarre"),

            # Cardiometabolic
            (r"\bhta\b|\bhypertension\s+art[ée]rielle\b", "Hypertension artérielle"),
            (r"\bdiab[èe]te\s+de\s+type\s+2\b|\bdnid\b", "Diabète de type 2"),
            (r"\bdiab[èe]te\s+de\s+type\s+1\b", "Diabète de type 1"),
            (r"\bdyslipid[ée]mie\b", "Dyslipidémie"),

            # Fractures / trauma
            #(r"\bfractures?\s+costales?\b|\bfractures?\s+des?\s+c[ôo]tes\b|\bfractures?.*\barcs?.*\bc[ôo]tes\b", "Fracture costale"),
            #(r"\btassement\s+vert[ée]bral\b|\btassements?\s+vert[ée]braux\b", "Tassement vertébral"),
        ]

        for pattern, replacement in canonical_rules:
            if re.search(pattern, low, flags=re.IGNORECASE):
                cleaned = replacement
                low = cleaned.lower()
                break

        # Final cleanup
        cleaned = cleaned.strip(" .;,:-\t")
        if len(cleaned) < 3:
            return ""

        return cleaned

    def _filter_unsupported_electrolyte_diagnoses(self, llm_diagnoses, text):
        """
        Remove LLM-extracted electrolyte diagnoses (Hyponatrémie, Hypernatrémie, Hypokaliémie, Hyperkaliémie)
        if not supported by explicit mention in text or by abnormal lab value.
        """
        # Electrolyte disorder names
        electrolyte_terms = {
            "Hyponatrémie": r"hyponatr[ée]mie",
            "Hypernatrémie": r"hypernatr[ée]mie",
            "Hypokaliémie": r"hypokali[ée]mie",
            "Hyperkaliémie": r"hyperkali[ée]mie",
        }
        # Find all numeric rule-based codes
        from nalfe.hfrs_hybrid_pipeline.utils import extract_numeric_rule_based_codes
        numeric_codes = set(d["code"] for d in extract_numeric_rule_based_codes(text) if "code" in d)
        filtered = []
        for diag in llm_diagnoses:
            keep = True
            for term, regex in electrolyte_terms.items():
                if diag.strip().lower() == term.lower():
                    # If not in text and not in numeric codes, remove
                    if not re.search(regex, text, flags=re.IGNORECASE) and "E87" not in numeric_codes:
                        keep = False
                        break
            if keep:
                filtered.append(diag)
        return filtered

    def preprocess_and_cache_notes(self, df: pd.DataFrame, text_col: str = 'text', cache_path: str = 'preprocessed_notes.json') -> pd.DataFrame:
        """
        Preprocess each document: run clean_text (with LLM/diagnostics) and extract_z_codes_hybrid (with LLM).
        Save results to cache_path as JSON. If cache exists, load and return.
        """
        import os
        import json
        if cache_path == 'preprocessed_notes.json':
            cache_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', 'output', 'preprocessed_notes.json')
            )
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
            return pd.DataFrame(cached)
        preprocessed_rows = []
        for _, row in df.iterrows():
            raw_text = row[text_col]
            # Apply canonical normalization to the entire cleaned note
            text_cleaned = self.normalize_text_for_umls(raw_text)
            text_cleaned = clean_text(text_cleaned, insert_diagnostics=False, processor=self)
            
            # Try to load section definitions from YAML to get all possible headers
            try:
                rules = SectionTokenizer.load_section_definition("/home/coder/nalfe/data/default_section_definition_updated_v2.yml", encoding='utf-8')
                section_dict = rules[0]
                section_headers = section_dict.get("diagnostics", [])
            except Exception:
                print("Warning: Failed to load section definitions from YAML. Using default diagnostics headers.")
                section_headers = ["DIAGNOSTICS", "DIAGNOSTIC", "AU TOTAL", "AU TOTAL :", "AU TOTAL:"]
            # Build regex to match any diagnostics header (case-insensitive, with/without colon)
            header_pattern = r"|".join([re.escape(h.strip(": ")) + r"\s*:" for h in section_headers])
            diagnostics_pattern = re.compile(rf"(?:{header_pattern})\s*(.*?)(?=\n[A-ZÉÈÀÂÎÔÛÇ\- ]+\s*:|\Z)", re.DOTALL | re.IGNORECASE)
            match = diagnostics_pattern.search(text_cleaned)
            if match:
                diagnostics_block = match.group(1)
                existing_disorders = set()
                for line in diagnostics_block.splitlines():
                    line = line.strip()
                    if line.startswith("-"):
                        disorder = line.lstrip("-• ").strip()
                        if disorder and not self._is_negated_or_absent_diagnosis(disorder):
                            existing_disorders.add(disorder)
            else:
                existing_disorders = set()
            llm_diagnostics = self.extract_active_diagnoses_llm(text_cleaned)
            # Filter unsupported electrolyte diagnoses
            llm_diagnostics = self._filter_unsupported_electrolyte_diagnoses(llm_diagnostics, raw_text)
            new_disorders = [d for d in llm_diagnostics if d not in existing_disorders]
            # Use canonical header for insertion/appending
            diagnostics_header = "Diagnostics :"
            if match and new_disorders:
                start, end = match.span(1)
                before = text_cleaned[:start]
                original = diagnostics_block.rstrip()
                appended = "\n" + "\n".join(f"- {d}" for d in new_disorders)
                after = text_cleaned[end:]
                text_cleaned = before + original + appended + after
            elif not match and llm_diagnostics:
                diagnostics_text = diagnostics_header + "\n- " + "\n- ".join(llm_diagnostics)
                text_cleaned = text_cleaned.strip() + "\n" + diagnostics_text
            text = text_cleaned
            preprocessed_rows.append({
                "raw_text": raw_text,
                "cleaned_text": text,
                "z_codes_hybrid": extract_z_codes_hybrid(raw_text),
            })
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(preprocessed_rows, f, ensure_ascii=False, indent=2)
        return pd.DataFrame(preprocessed_rows)
        # No error raised here; error handling is done in extract_active_diagnoses_llm
        # (If neither HuggingFace nor Ollama is available, extract_active_diagnoses_llm will raise.)

    def _extract_cim10_codes(self, cim10_codes):
        """
        Recursively extract all CIM-10 codes from any structure (dict, list, string).
        Expands code ranges (e.g., 'S00-S09') into individual 3-char codes.
        Keep raw QuickUMLS codes here and let downstream contextual filtering
        (scan_full_text_rule_based_codes) apply the final 109-code allowlist.
        Returns a set of code strings.
        """
        codes = set()

        def normalize_code(code: str) -> str:
            return str(code).strip().upper()

        def expand_code_range(code_range):
            # Only handle ranges like 'S00-S09'
            if '-' in code_range and len(code_range) <= 9:
                start, end = code_range.split('-')
                start = normalize_code(start)
                end = normalize_code(end)
                if not start or not end or start[0] != end[0]:
                    return []
                prefix = start[0]
                try:
                    start_num = int(start[1:])
                    end_num = int(end[1:])
                except ValueError:
                    return []
                expanded = [f"{prefix}{str(i).zfill(2)}" for i in range(start_num, end_num+1)]
                return expanded
            return []

        if isinstance(cim10_codes, dict):
            code = cim10_codes.get('code')
            if code:
                code = normalize_code(code)
                if '-' in code:
                    codes.update(expand_code_range(code))
                else:
                    codes.add(code)
        elif isinstance(cim10_codes, str):
            cim10_codes = normalize_code(cim10_codes)
            if '-' in cim10_codes:
                codes.update(expand_code_range(cim10_codes))
            else:
                codes.add(cim10_codes)
        elif isinstance(cim10_codes, list):
            for item in cim10_codes:
                codes.update(self._extract_cim10_codes(item))
        return codes

    def _load_cim10_classifier(self, model_dir="/home/coder/nalfe/results/best_model"):
        """
        Load fine-tuned CamemBERT classifier for CIM-10 code prediction.
        Uses the selected device (auto GPU or CPU).
        """
        from transformers import CamembertTokenizer, CamembertForSequenceClassification
        import torch
        self.cim10_tokenizer = CamembertTokenizer.from_pretrained(model_dir)
        self.cim10_model = CamembertForSequenceClassification.from_pretrained(model_dir)
        self.cim10_model.eval()
        # Convert id2label keys to int for correct lookup
        self.cim10_id2label = {int(k): v for k, v in self.cim10_model.config.id2label.items()}
        # Use the same device as selected in __init__
        if hasattr(self, "device") and self.device and self.device != "cpu":
            self.cim10_device = torch.device(self.device)
        else:
            self.cim10_device = torch.device("cpu")
        self.cim10_model.to(self.cim10_device)

    # Load CUI to CIM10 mapping once (class-level cache)
    _cui_to_cim10 = None

    def _get_cui_to_cim10(self):
        if ClinicalNoteProcessor._cui_to_cim10 is None:
            import json
            mapping_path = os.path.abspath("/home/coder/nalfe/data/cui_to_cim10.json")
            with open(mapping_path, encoding="utf-8") as f:
                ClinicalNoteProcessor._cui_to_cim10 = json.load(f)
        return ClinicalNoteProcessor._cui_to_cim10

    def medkit_structuring(self, text: str) -> dict:
        """
        Run medkit pipeline: cleaning, sectioning, sentence splitting, negation (on sentences), NER (with negation/section attrs copied to entities), normalization.
        Returns structured dict with sections, sentences, and entities.
        """
        # Ensure QuickUMLS French index path is set for entity extraction
        if not os.environ.get("QUICKUMLS_FRE_2025AA"):
            # Set default path if not already set
            os.environ["QUICKUMLS_FRE_2025AA"] = os.path.abspath("quickumls_installs/2025AA_fre_lower")
        # Try to load section definitions from YAML to get all possible headers
        try:
            rules = SectionTokenizer.load_section_definition("/home/coder/nalfe/data/default_section_definition_updated_v2.yml", encoding='utf-8')
            section_dict = rules[0]
            section_headers = section_dict.get("diagnostics", [])
        except Exception:
            print("Warning: Failed to load section definitions from YAML. Using default diagnostics headers.")
            section_headers = ["DIAGNOSTICS", "DIAGNOSTIC", "AU TOTAL", "AU TOTAL :", "AU TOTAL:"]
        # Clean and normalize the input text using LLM and diagnostics insertion
        # Step 1: Clean the input text without inserting diagnostics
        text_cleaned = clean_text(text, insert_diagnostics=False, processor=self)
        diagnostics_header = "Diagnostics :"
        # Step 2: Find the Diagnostics section in the cleaned text
        import re
        diagnostics_pattern = re.compile(r"Diagnostics\s*:\s*(.*?)(?=\n[A-ZÉÈÀÂÎÔÛÇ\- ]+\s*:|\Z)", re.DOTALL | re.IGNORECASE)
        match = diagnostics_pattern.search(text_cleaned)
        if match:
            # Step 3: Extract existing disorders from Diagnostics section (lines starting with '-')
            diagnostics_block = match.group(1)
            existing_disorders = []
            seen = set()
            for line in diagnostics_block.splitlines():
                line = line.strip()
                if line.startswith("-"):
                    disorder = line.lstrip("-• ").strip()
                    if disorder and not self._is_negated_or_absent_diagnosis(disorder) and disorder not in seen:
                        existing_disorders.append(disorder)
                        seen.add(disorder)
        else:
            existing_disorders = []
            seen = set()
        # Step 4: Always extract disorders with LLM from the full cleaned text.
        # Even if a Diagnostics section already exists, the LLM can recover disorders
        # mentioned elsewhere in the note that should be consolidated into Diagnostics.
        llm_diagnostics = self.extract_active_diagnoses_llm(text_cleaned)
        # Step 5: Merge and deduplicate, preserving order: existing first, then new unique
        all_disorders = existing_disorders[:]
        for d in llm_diagnostics:
            if d not in seen:
                all_disorders.append(d)
                seen.add(d)
        # Step 6: Rebuild Diagnostics section with deduplicated disorders
        # Final deduplication for output (preserve order)
        final_disorders = []
        final_seen = set()
        for d in all_disorders:
            if d not in final_seen:
                final_disorders.append(d)
                final_seen.add(d)
        if match:
            start, end = match.span(1)
            before = text_cleaned[:start]
            after = text_cleaned[end:]
            diagnostics_text = "\n" + "\n".join(f"- {d}" for d in final_disorders)
            text_cleaned = before + diagnostics_text + after
        elif final_disorders:
            diagnostics_text = diagnostics_header + "\n- " + "\n- ".join(final_disorders)
            text_cleaned = text_cleaned.strip() + "\n" + diagnostics_text
        # Step 7: Use the merged/augmented text for further processing
        text = text_cleaned
        try:
            import unicodedata
            text = unicodedata.normalize("NFC", text)
        except ImportError:
            pass
        # --- Ensure diagnostics section header is on its own line for tokenizer ---
        # If diagnostics section was appended, ensure header is on a separate line
        # Normalize all diagnostics headers to canonical form for sectionizer
        for h in section_headers:
            # Replace any variant header with canonical header on its own line
            pattern = re.compile(rf"\s*{re.escape(h.strip(': '))}\s*:\s*", re.IGNORECASE)
            text = pattern.sub("\nDiagnostics :\n", text)
        # Now proceed with section splitting
        # --- Extra deduplication: re-extract Diagnostics section and deduplicate again ---
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        diagnostics_start = None
        for idx, line in enumerate(lines):
            if line.lower().startswith('diagnostics'):
                diagnostics_start = idx
                break
        if diagnostics_start is not None:
            header = lines[diagnostics_start]
            disorders = []
            for l in lines[diagnostics_start+1:]:
                if l.startswith('-'):
                    disorder_line = l.lstrip('-• ').strip()
                    if disorder_line and not self._is_negated_or_absent_diagnosis(disorder_line):
                        # Split if multiple disorders joined by ' - '
                        split_disorders = [d.strip() for d in disorder_line.split(' - ') if d.strip()]
                        disorders.extend(split_disorders)
            # Deduplicate: if both 'X' and 'X (details)' exist, keep only the one with parenthesis
            base_to_full = {}
            for d in disorders:
                # Extract base (before parenthesis)
                import re
                m = re.match(r"^(.*?)(?:\s*\(.*\))?$", d)
                base = m.group(1).strip() if m else d.strip()
                # Prefer the one with parenthesis if both exist
                if base not in base_to_full or (base_to_full[base] == base and d != base):
                    base_to_full[base] = d
            # Preserve order
            seen = set()
            deduped = []
            for d in disorders:
                m = re.match(r"^(.*?)(?:\s*\(.*\))?$", d)
                base = m.group(1).strip() if m else d.strip()
                best = base_to_full[base]
                if best not in seen:
                    deduped.append(best)
                    seen.add(best)
            # Rebuild lines
            new_diag_lines = [header] + [f"- {d}" for d in deduped]
            # Replace in lines
            lines = lines[:diagnostics_start] + new_diag_lines
        text2 = "\n".join(lines)
        # Create medkit doc
        doc = TextDocument(text=text2)
        char_replacer = CharReplacer(output_label="char_replaced")
        doc.raw_segment = char_replacer.run([doc.raw_segment])[0]
        # Section segmentation
        try:
            # Try to load section definitions from YAML file
            rules = SectionTokenizer.load_section_definition("/home/coder/nalfe/data/default_section_definition_updated_v2.yml", encoding='utf-8')
            section_tokenizer = SectionTokenizer(section_dict=rules[0], section_rules=rules[1], output_label="section")
        except Exception as e:
            # Fallback to default section dictionary if YAML loading fails
            print(f"Warning: Failed to load section definitions from YAML. Using default section dictionary. Error: {e}")
            section_dict = {
                "Antécédents": ["ANTÉCÉDENTS", "ANTECEDENTS", "HISTORIQUE"],
                "Motif de consultation": ["MOTIF DE CONSULTATION", "RAISON DE CONSULTATION"],
                "Examen clinique": ["EXAMEN CLINIQUE", "EXAMEN PHYSIQUE", "EXAMEN"],
                "Traitement": ["TRAITEMENT", "PRESCRIPTION", "ORDONNANCE", ],
                "Diagnostics": ["DIAGNOSTICS", "DIAGNOSTIC", "AU TOTAL", "AU TOTAL :", "AU TOTAL:"],
                "Conclusion": ["CONCLUSION"]
            }
            section_tokenizer = SectionTokenizer(section_dict=section_dict, output_label="section")
        # Segment the document into sections
        sections = section_tokenizer.run([doc.raw_segment])
        # Sentence segmentation (copy section attr)
        sentence_tokenizer = SentenceTokenizer(
            keep_punct=True, 
            split_on_newlines=True, 
            output_label="sentence", 
            attrs_to_copy=["section"]
            )
        section_sentences = []
        for section in sections:
            # Explicitly copy section attribute to each sentence
            sents = sentence_tokenizer.run([section])
            from medkit.core import Attribute
            for sent in sents:
                # Remove any pre-existing section attr without breaking AttributeCollection
                # Safely remove all 'section' attributes using medkit API
                attrs_to_remove = [attr for attr in sent.attrs if attr.label == "section"]
                for attr in attrs_to_remove:
                    try:
                        sent.attrs.remove(attr)
                    except Exception:
                        pass
                sent.attrs.add(Attribute(label="section", value=section.metadata.get("name", section.label)))
            section_sentences.append((section, sents))
        # Negation detection on sentences
        neg_detector = NegationDetector(output_label="negation")
        all_sentences = [sent for _, sents in section_sentences for sent in sents]

        # Debug: print section attribute of each sentence before entity extraction
        # Debug print removed
        if not all_sentences:
            negated_sentences = []
        else:
            negated_sentences = neg_detector.run(all_sentences)
            if negated_sentences is None:
                negated_sentences = all_sentences
        # Only use QuickUMLSMatcher for entity extraction (no HFEntityMatcher)
        try:
            umls_matcher = QuickUMLSMatcher(
                version="2025AA",
                language="FRE",
                lowercase=True,
                normalize_unicode=False,
                window=8,
                threshold=0.95,
                similarity="cosine",
            )
            quickumls_entities = []
            for sent in all_sentences:
                entities = umls_matcher.run([sent])
                section_val = None
                negation_val = None
                for attr in sent.attrs:
                    if attr.label == "section":
                        section_val = attr.value
                    if attr.label == "negation":
                        negation_val = attr.value
                for ent in entities:
                    if not self._filter_quickumls_entity(ent.text):
                        continue
                    from medkit.core import Attribute
                    ent.attrs = [a for a in ent.attrs if a.label not in ("section", "negation")]
                    ent.attrs.append(Attribute(label="section", value=section_val))
                    if negation_val is not None:
                        ent.attrs.append(Attribute(label="negation", value=negation_val))
                    quickumls_entities.append(ent)
        except Exception as e:
            # If QuickUMLSMatcher fails, return empty entity list
            quickumls_entities = []
        if quickumls_entities is None:
            quickumls_entities = []

        # Build structured output for QuickUMLS only
        output = {"sections": [], "quickumls_entities": []}
        output["sections"] = []
        for section in sections:
            label = section.metadata.get("name", section.label)
            if label.lower() == "diagnostics":
                lines = [l.strip() for l in section.text.splitlines() if l.strip()]
                header = lines[0] if lines and (lines[0].lower().startswith('diagnostics')) else None
                disorders = []
                for l in lines[1:]:
                    if l.startswith('-'):
                        disorder = l.lstrip('-• ').strip()
                        if disorder and not self._is_negated_or_absent_diagnosis(disorder):
                            disorders.append(disorder)
                seen = set()
                deduped = []
                for d in disorders:
                    if d not in seen:
                        deduped.append(d)
                        seen.add(d)
                text = ''
                if header:
                    text += header + "\n"
                text += "\n".join(f"- {d}" for d in deduped)
                output["sections"].append({"label": label, "text": text})
            else:
                output["sections"].append({"label": label, "text": section.text})

        # Only QuickUMLS entities are included
        for section, sents in section_sentences:
            section_json = {
                "label": section.metadata.get("section", section.label),
                "text": section.text,
                "sentences": []
            }
            section_quickumls_entities = []
            section_quickumls_seen = set()
            def get_span_start_end(s):
                if hasattr(s, 'start') and hasattr(s, 'end'):
                    return s.start, s.end
                elif hasattr(s, 'replaced_spans') and s.replaced_spans:
                    return s.replaced_spans[0].start, s.replaced_spans[0].end
                else:
                    raise AttributeError('Span object has no start/end or replaced_spans')
            for sent in sents:
                sent_json = {"text": sent.text, "quickumls_entities": []}
                for ent in quickumls_entities:
                    if ent.label != "disorder":
                        continue
                    span = ent.spans[0]
                    sent_span = sent.spans[0]
                    span_start, span_end = get_span_start_end(span)
                    sent_start, sent_end = get_span_start_end(sent_span)
                    if span_start >= sent_start and span_end <= sent_end:
                        disorder_text = ent.text
                        if disorder_text not in section_quickumls_seen:
                            umls_cui = next((attr.value for attr in ent.attrs if attr.label == "NORMALIZATION"), None)
                            if umls_cui and umls_cui.startswith("umls:"):
                                umls_cui = umls_cui[5:]
                            cim10_codes = []
                            if umls_cui:
                                cui_to_cim10 = self._get_cui_to_cim10()
                                cim10_codes = cui_to_cim10.get(umls_cui, [])

                            code3_set = set()
                            for code in cim10_codes:
                                if isinstance(code, str) and len(code) >= 3:
                                    code3_set.add(code[:3])
                            code3_list = list(code3_set)

                            if code3_list:
                                # If only one unique 3-char code, use it directly
                                if len(code3_list) == 1:
                                    selected_codes = code3_list
                                # If multiple unique 3-char codes, use select_best_cim10_code to choose
                                else:
                                    best_code = self.select_best_cim10_code(
                                        sent.text,
                                        (span_start - sent_start, span_end - sent_start),
                                        code3_list
                                    )
                                    selected_codes = [best_code] if best_code else code3_list
                            else:
                                # --- Only if QuickUMLS returns no code, fallback to classifier/mapping ---
                                selected_codes = []

                            ent_json = {
                                "text": ent.text,
                                "label": ent.label,
                                "negated": any(attr.value for attr in ent.attrs if attr.label == "negation"),
                                "umls_cui": umls_cui,
                                "cim10_codes": selected_codes,
                                "section": next((attr.value for attr in ent.attrs if attr.label == "section"), None)
                            }
                            sent_json["quickumls_entities"].append(ent_json)
                            output["quickumls_entities"].append(ent_json)
                            section_quickumls_entities.append(ent_json)
                            section_quickumls_seen.add(disorder_text)
                section_json["sentences"].append(sent_json)
        return output

    def __init__(self, cim10_mapping: Dict[str, str], ner_model_name: str = None, llm_model_name: str = None, device: str = None,
                 hfentity_prob_threshold: float = 0.9, quickumls_prob_threshold: float = 0.8, classifier_prob_threshold: float = 0.99):
        self.cim10_mapping = cim10_mapping
        self.ner_model_name = ner_model_name
        self.llm_model_name = llm_model_name
        self.llm_pipeline = None
        # Only load HuggingFace pipeline if llm_model_name does NOT start with 'ollama:'
        if self.llm_model_name and LLM_AVAILABLE and not self.llm_model_name.startswith('ollama:'):
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
                tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
                model = AutoModelForCausalLM.from_pretrained(self.llm_model_name)
                self.llm_pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
            except Exception as e:
                print(f"Warning: Could not load LLM pipeline for {self.llm_model_name}: {e}")
                self.llm_pipeline = None
        # Auto-select GPU with most free memory if device is not specified
        if device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        device_count = pynvml.nvmlDeviceGetCount()
                        max_free = 0
                        best_gpu = 0
                        for i in range(device_count):
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            if meminfo.free > max_free:
                                max_free = meminfo.free
                                best_gpu = i
                        self.device = f"cuda:{best_gpu}"
                        # Info: Using GPU with most free memory
                        pynvml.nvmlShutdown()
                    except Exception as e:
                        # Warn: Could not auto-select GPU, using default cuda:0
                        self.device = "cuda:0"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
        self.ner_pipeline = None
        self.llm_pipeline = None
        self.hfentity_prob_threshold = hfentity_prob_threshold
        self.quickumls_prob_threshold = quickumls_prob_threshold
        self.classifier_prob_threshold = classifier_prob_threshold
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        tokenizer_kwargs = {"use_fast": True}
        from_pretrained_kwargs = {}
        private_models = ["Dr-BERT/DrBERT-4GB-ner", "almanach/medcamembert-ner"]
        if hf_token and ner_model_name in private_models:
            from_pretrained_kwargs["use_auth_token"] = hf_token
            tokenizer_kwargs["use_auth_token"] = hf_token
        def get_device_index(device):
            if device is None or device == "cpu":
                return -1
            if device.startswith("cuda:"):
                return int(device.split(":")[1])
            if device == "cuda":
                return 0
            return -1

        device_index = get_device_index(self.device)

        if ner_model_name and TRANSFORMERS_AVAILABLE:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            try:
                tokenizer = AutoTokenizer.from_pretrained(ner_model_name, **tokenizer_kwargs)
            except Exception:
                tokenizer_kwargs["use_fast"] = False
                tokenizer = AutoTokenizer.from_pretrained(ner_model_name, **tokenizer_kwargs)
            model = AutoModelForTokenClassification.from_pretrained(ner_model_name, **from_pretrained_kwargs)
            self.ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=device_index
            )
        # Only support Ollama LLMs for llm_model_name; do not load HuggingFace LLM pipeline
        self.llm_pipeline = None

        self.spacy_nlp = spacy.blank("fr")
        self.spacy_nlp.add_pipe("sentencizer")

        self.umls_matcher = QuickUMLSMatcher(
            version="2025AA",
            language="FRE",
            lowercase=True,
            normalize_unicode=False,
        )

    def read_data(self, path: str) -> pd.DataFrame:
        if path.endswith('.csv'):
            return pd.read_csv(path, encoding='utf-8')
        elif path.endswith('.xlsx') or path.endswith('.xls'):
            return pd.read_excel(path)
        elif path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        else:
            raise ValueError('Unsupported file format. Use .csv, .xlsx, .xls, or .json')


    def _mapping_fallback_codes(self, disorder_text: str) -> List[str]:
        """Return CIM-10 codes from the term mapping when classifier has no confident prediction."""
        if not isinstance(self.cim10_mapping, dict):
            return []

        normalized = (disorder_text or "").strip().lower()
        if not normalized:
            return []

        # Build a case-insensitive index lazily for robust exact-match lookup.
        if not hasattr(self, "_cim10_mapping_lower"):
            mapping_lower = {}
            for term, code in self.cim10_mapping.items():
                term_key = str(term).strip().lower()
                if not term_key:
                    continue
                if isinstance(code, str):
                    mapping_lower.setdefault(term_key, set()).add(code)
                elif isinstance(code, (list, tuple, set)):
                    for c in code:
                        if c:
                            mapping_lower.setdefault(term_key, set()).add(str(c))
            self._cim10_mapping_lower = mapping_lower

        matched = self._cim10_mapping_lower.get(normalized, set())
        return sorted(matched)

    def _extract_diagnostics_section_terms(self, medkit_struct: dict) -> List[str]:
        """Extract unique diagnosis strings from the reconstructed Diagnostics section."""
        sections = medkit_struct.get("sections", []) if isinstance(medkit_struct, dict) else []
        diagnoses = []
        seen = set()

        for section in sections:
            label = str(section.get("label", "")).strip().lower()
            if label != "diagnostics":
                continue

            text = section.get("text", "") or ""
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped or stripped.lower().startswith("diagnostics"):
                    continue
                if stripped.startswith("-"):
                    stripped = stripped.lstrip("-• ").strip()
                if not stripped:
                    continue

                split_disorders = [item.strip() for item in stripped.split(" - ") if item.strip()]
                for disorder in split_disorders:
                    if self._is_negated_or_absent_diagnosis(disorder):
                        continue
                    key = disorder.lower()
                    if key not in seen:
                        diagnoses.append(disorder)
                        seen.add(key)

        return diagnoses

    def scan_full_text_rule_based_codes(self, text: str, existing_codes: list = None) -> list:
        """Delegate to utils.scan_full_text_rule_based_codes, supporting existing_codes for context-aware filtering."""
        return scan_full_text_rule_based_codes(text, existing_codes)

    def _filter_quickumls_entity(self, entity_text: str) -> bool:
        """
        Stricter filter for QuickUMLS entities to exclude non-specific, process, or incomplete terms.
        Returns True if entity should be kept, False if it should be discarded.
        
        Filters out:
        - Very short text (< 4 chars)
        - Generic/process terms (expanded blacklist)
        - Known non-diagnostic or administrative terms
        - Single-character fragments
        """
        text = (entity_text or "").strip().lower()
        # Filter 1: Very short matches (< 4 characters)
        if len(text) < 4:
            return False

        # Expanded blacklist of generic/process/administrative terms (add as needed)
        generic_terms = {
            'syndrome', 'distension', 'dilatation', 'urgences', 'ampoule', 'douleur', 'fievre', 'malaise', 'fatigue',
            'hemorragie', 'perforation', 'rupture', 'inflammation', 'œdeme', 'edema', 'hyper', 'hypo', 'blanche',
            'purge', 'aspiration', 'lésions', 'lésion', 'fractures', 'fracture', 'saignements', 'saignement',
            'trauma', 'traumatisme', 'trouble', 'troubles', 'défaillance', 'défaut', 'anomalie', 'modification',
            'altération', 'changement', 'dysfonction', 'dysfonctionnement', 'problème', 'problèmes', 'complication',
            'complications', 'manifestation', 'manifestations', 'événement', 'événements', 'procédure', 'procédures',
            'soin', 'soins', 'prise en charge', 'prise en charge médicale', 'prise en charge chirurgicale',
            'consultation', 'consultations', 'examen', 'examens', 'bilan', 'bilans', 'traitement', 'traitements',
            'dispositif', 'dispositifs', 'hospitalisation', 'hospitalisations', 'sortie', 'entrée', 'décès',
            'cause', 'causes', 'facteur', 'facteurs', 'risque', 'risques', 'antécédent', 'antécédents',
            'contexte', 'conseil', 'conseils', 'information', 'informations', 'mesure', 'mesures', 'contrôle',
            'contrôles', 'surveillance', 'suivi', 'suivis', 'prise', 'prise en charge', 'prise en charge médicale',
            'prise en charge chirurgicale', 'soin', 'soins', 'protocole', 'protocoles', 'plan', 'plans',
            'objectif', 'objectifs', 'décision', 'décisions', 'orientation', 'orientations', 'projet', 'projets',
            'programme', 'programmes', 'service', 'services', 'unité', 'unités', 'secteur', 'secteurs',
            'zone', 'zones', 'structure', 'structures', 'établissement', 'établissements', 'centre', 'centres',
            'groupe', 'groupes', 'population', 'populations', 'collectif', 'collectifs', 'individu', 'individus',
            'personne', 'personnes', 'patient', 'patients', 'malade', 'malades', 'enfant', 'enfants', 'adulte',
            'adultes', 'femme', 'femmes', 'homme', 'hommes', 'âge', 'âges', 'année', 'années', 'mois', 'jours',
            'heure', 'heures', 'minute', 'minutes', 'seconde', 'secondes', 'fois', 'nombre', 'quantité', 'volume',
            'poids', 'taille', 'indice', 'score', 'degré', 'niveau', 'taux', 'valeur', 'mesure', 'résultat', 'donnée',
            'paramètre', 'élément', 'aspect', 'aspects', 'type', 'types', 'forme', 'formes', 'nature', 'natures',
            'catégorie', 'catégories', 'classe', 'classes', 'groupe', 'groupes', 'sous-groupe', 'sous-groupes',
            'ensemble', 'ensembles', 'liste', 'listes', 'tableau', 'tableaux', 'fiche', 'fiches', 'dossier', 'dossiers',
            'document', 'documents', 'rapport', 'rapports', 'note', 'notes', 'lettre', 'lettres', 'message',
            'messages', 'appel', 'appels', 'contact', 'contacts', 'rendez-vous', 'rendez vous', 'visite', 'visites',
            'consultation', 'consultations', 'examen', 'examens', 'bilan', 'bilans', 'traitement', 'traitements',
            'protocole', 'protocoles', 'plan', 'plans', 'objectif', 'objectifs', 'décision', 'décisions',
            'orientation', 'orientations', 'projet', 'projets', 'programme', 'programmes', 'service', 'services',
            'unité', 'unités', 'secteur', 'secteurs', 'zone', 'zones', 'structure', 'structures', 'établissement',
            'établissements', 'centre', 'centres', 'groupe', 'groupes', 'population', 'populations', 'collectif',
            'collectifs', 'individu', 'individus', 'personne', 'personnes', 'patient', 'patients', 'malade',
            'malades', 'enfant', 'enfants', 'adulte', 'adultes', 'femme', 'femmes', 'homme', 'hommes', 'âge',
            'âges', 'année', 'années', 'mois', 'jours', 'heure', 'heures', 'minute', 'minutes', 'seconde',
            'secondes', 'fois', 'nombre', 'quantité', 'volume', 'poids', 'taille', 'indice', 'score', 'degré',
            'niveau', 'taux', 'valeur', 'mesure', 'résultat', 'donnée', 'paramètre', 'élément', 'aspect', 'type',
            'forme', 'nature', 'catégorie', 'classe', 'groupe', 'sous-groupe', 'ensemble', 'liste', 'tableau',
            'fiche', 'dossier', 'document', 'rapport', 'note', 'lettre', 'message', 'appel', 'contact',
            'rendez-vous', 'visite', 'ischémie', 'crises', 'crise', 'retraite',
        }
        if text in generic_terms:
            return False

        # Filter 3: Single-character fragments
        if len(text) == 1:
            return False

        return True  # Keep entity

    def compute_hospital_fragility_score(self, codes: List[str]) -> float:
        # Load points from hfrs_points.csv
        points_map = {}
        import csv
        with open('/home/coder/nalfe/data/hfrs_points.csv', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                points_map[row['code']] = float(row['points'])
        score = 0.0
        # Only count unique codes for scoring
        for code in set(codes):
            score += points_map.get(code, 0.0)
        return score

    def process_notes(self, df: pd.DataFrame, text_col: str = 'text', preprocessed_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Process a DataFrame of clinical notes, using preprocessed clean_text and Z-code results if provided.
        Extract codes (with classifier fallback), compute HFRS scores, and return all relevant columns.
        """
        hfrs_code_set = set()
        import csv
        with open('/home/coder/nalfe/data/hfrs_points.csv', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                hfrs_code_set.add(row['code'])
        results = []
        if preprocessed_df is not None:
            df = df.copy()
            preprocessed_df = preprocessed_df.copy()
            # Normalize text columns for robust merging
            def normalize_text(s):
                return s.astype(str).str.strip().str.replace('\r\n', '\n').str.replace('\r', '\n').str.replace(' +', ' ', regex=True)
            df[text_col + '_norm'] = normalize_text(df[text_col])
            preprocessed_df['raw_text_norm'] = normalize_text(preprocessed_df['raw_text'])
            # Merge on normalized columns
            df = df.merge(preprocessed_df, left_on=text_col + '_norm', right_on='raw_text_norm', how='left')
        for _, row in df.iterrows():
            raw_text = row[text_col]
            cleaned_text = row.get('cleaned_text', None)
            z_codes = row.get('z_codes_hybrid', None)
            # Robustly check for NaN/missing values before passing to clean_text/medkit_structuring
            import pandas as pd
            def is_valid_text(val):
                return isinstance(val, str) and val.strip() != '' and not pd.isna(val)

            # Use preprocessed cleaned_text if available and valid, else fallback to valid raw_text, else empty string
            if is_valid_text(cleaned_text):
                medkit_input = cleaned_text
            elif is_valid_text(raw_text):
                medkit_input = raw_text
            else:
                medkit_input = ''
            medkit_struct = self.medkit_structuring(medkit_input)
            diagnostics_terms = self._extract_diagnostics_section_terms(medkit_struct)
            # Use preprocessed z_codes if available, else fallback
            z_codes = z_codes if z_codes is not None else extract_z_codes_hybrid(raw_text)
            # Do not add Z-codes to diag_codes_hf/diag_codes_quickumls here
            diag_codes_quickumls = []
            quickumls_classifier_codes = set()
           
            # QuickUMLSMatcher: get codes from 'diagnostics' section
            for ent in medkit_struct.get("quickumls_entities", []):
                section = ent.get("section") or ent.get("section_label") or None
                section_norm = (section or "").strip().lower()
                if ent["label"] == "disorder" and not ent["negated"] : #and section_norm == "diagnostics":
                    cim10_codes = ent.get("cim10_codes")
                    # --- If QuickUMLS returns any code, use it (even if not in 109 list), skip fallback ---
                    found_code = False
                    if cim10_codes:
                        all_codes = self._extract_cim10_codes(cim10_codes)
                        if all_codes:
                            for code in all_codes:
                                code3 = code[:3]
                                diag_codes_quickumls.append({"problem": ent["text"], "code": code3})
                                found_code = True
                        else:
                            # If cim10_codes exists but _extract_cim10_codes returns nothing, still treat as "codes returned" and skip fallback
                            # Use the raw codes (truncated to 3 chars) as a fallback for output
                            for code in cim10_codes:
                                code3 = str(code)[:3]
                                diag_codes_quickumls.append({"problem": ent["text"], "code": code3})
                                found_code = True
                    # Fallback: only if QuickUMLS returns no code at all
                    if not cim10_codes or not found_code:
                        codes = rule_based_diagnosis_codes(ent["text"], diagnostics_terms, [item["code"] for item in diag_codes_quickumls])
                        if not codes:
                            codes = self._mapping_fallback_codes(ent["text"])
                        for code in codes:
                            code3 = code[:3]
                            diag_codes_quickumls.append({"problem": ent["text"], "code": code3})
                            quickumls_classifier_codes.add(code3)

            quickumls_seen_terms = {item["problem"].strip().lower() for item in diag_codes_quickumls}
            context_codes = [item["code"] for item in diag_codes_quickumls]
            
            # For each diagnosis term in the Diagnostics section, assign a CIM-10 code if not already processed
            for disorder in diagnostics_terms:
                disorder_key = disorder.strip().lower()
                # Skip empty terms or terms already processed (avoid duplicates)
                if not disorder_key or disorder_key in quickumls_seen_terms:
                    continue
                # Step 1: Try to assign codes using the rule-based method (context-aware, uses other terms and codes)
                codes = rule_based_diagnosis_codes(disorder, diagnostics_terms, context_codes)
                # Step 2: If rule-based method fails, try to assign codes using the mapping fallback (dictionary lookup)
                if not codes:
                    codes = self._mapping_fallback_codes(disorder)
                # Step 3: If still no code, use the classifier to predict the code (ML model, only if above threshold)
                #if not codes:
                for code in codes:
                    code3 = code[:3]
                    diag_codes_quickumls.append({"problem": disorder, "code": code3})
                    quickumls_classifier_codes.add(code3)
                    context_codes.append(code3)
                # Mark this disorder as processed to avoid duplicate assignment
                quickumls_seen_terms.add(disorder_key)

            # Pass the full diag_codes_quickumls list to scan_full_text_rule_based_codes for context-aware filtering
            diag_codes_quickumls = self.scan_full_text_rule_based_codes(medkit_input, diag_codes_quickumls)
           
            # After filtering, update code sets to only include codes present in the final lists
            final_quickumls_codes = {c['code'] for c in diag_codes_quickumls}
            # Add Z-codes (if any) to the final code sets (no redundancy, Z-codes are unique)
            if z_codes:
                # Load 109-code list from file (or cache if needed)
                if not hasattr(self, '_valid_cim10_codes'):
                    code_path = '/home/coder/nalfe/data/cim_109_code.csv'
                    with open(code_path, encoding='utf-8') as f:
                        import csv
                        reader = csv.reader(f)
                        self._valid_cim10_codes = set(row[0][:3] for row in reader if row)
                cim_109_codes = self._valid_cim10_codes
                z_code_strs = self.extract_z_codes_from_hybrid(z_codes, cim_109_codes)
                final_quickumls_codes.update(z_code_strs)
                # Also add Z-codes to diagnostics_cim10_codes_quickumls for display
                for z_code in z_code_strs:
                    # Use Z-code description as problem if available
                    z_code_desc = None
                    # Try to get description from extract_z_codes_hybrid output (row['z_codes_hybrid'])
                    if isinstance(z_codes, list):
                        for z_item in z_codes:
                            if isinstance(z_item, dict) and z_item.get("code") == z_code:
                                z_code_desc = z_item.get("problem")
                                break
                    diag_codes_quickumls.append({"problem": z_code_desc if z_code_desc else z_code, "code": z_code})
            # HFRS scores should be computed from the final code lists (including Z-codes)
            hfrs_quickumls = self.compute_hospital_fragility_score([code for code in final_quickumls_codes if code in hfrs_code_set])
            fragility_category = self.compute_fragility_category(hfrs_quickumls)
            results.append({
                'raw_text': raw_text,
                'diagnostics_cim10_codes_quickumls': diag_codes_quickumls,
                'UMLS_CIM10': list(final_quickumls_codes),
                'HFRS_score_quickumls': hfrs_quickumls,
                'fragility_category': fragility_category,
                'structured': medkit_struct,
            })
        return pd.DataFrame(results)

    @staticmethod
    def compute_fragility_category(score: float) -> str:
        if score >= 5 and score < 15:
            return 'intermediate'
        elif score >= 15:
            return 'high'
        else:
            return 'low'
