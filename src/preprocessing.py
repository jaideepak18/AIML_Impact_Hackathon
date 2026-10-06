"""Text preprocessing and normalization module for Business Entity Resolution.

Provides comprehensive, rule-based text cleaning for business names and addresses,
handling abbreviations, international legal suffixes (US, India, France),
numeric components, and tokenization without external geocoding dependencies.
"""

import re
import unicodedata
from typing import List, Optional, Pattern, Set, Tuple
import pandas as pd

# -----------------------------------------------------------------------------
# Precompiled Regexes & Substitution Tables
# -----------------------------------------------------------------------------
RE_WHITESPACE: Pattern[str] = re.compile(r"\s+")
RE_PUNCTUATION: Pattern[str] = re.compile(r"[^\w\s]")
RE_DIGITS: Pattern[str] = re.compile(r"\b\d+\b")
RE_AMPERSAND: Pattern[str] = re.compile(r"\s*&\s*")

# DBA / Trade Name patterns
RE_DBA: Pattern[str] = re.compile(
    r"\b(?:d/?b/?a|t/?a|trading\s+as|doing\s+business\s+as|a/?k/?a)\b",
    flags=re.IGNORECASE,
)

# Common business abbreviation mappings (US, India, France)
NAME_ABBREVIATIONS = {
    r"\bmfg\b": "manufacturing",
    r"\bintl\b": "international",
    r"\bsvc\b": "services",
    r"\bsvcs\b": "services",
    r"\btech\b": "technology",
    r"\btechnol\b": "technology",
    r"\bmgmt\b": "management",
    r"\bassoc\b": "associates",
    r"\bctr\b": "center",
    r"\bcentre\b": "center",
    r"\bdept\b": "department",
    r"\bgrp\b": "group",
    r"\bhosp\b": "hospital",
    r"\bmed\b": "medical",
    r"\buniv\b": "university",
    r"\bcomm\b": "communications",
    r"\bsols\b": "solutions",
    r"\bsoln\b": "solution",
    r"\bsolns\b": "solutions",
    r"\bengg\b": "engineering",
    r"\bent\b": "enterprises",
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bco\b": "company",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    # French entity abbreviations
    r"\bassoc\b": "association",
    r"\bets\b": "etablissements",
    r"\bcie\b": "compagnie",
    r"\bste\b": "societe",
}

# Legal suffixes across jurisdictions (US, India, France/EU)
LEGAL_SUFFIX_PATTERNS = [
    # Multi-word suffixes first
    r"\bprivate limited\b",
    r"\bpvt limited\b",
    r"\bpvt ltd\b",
    r"\bpublic limited company\b",
    r"\blimited liability company\b",
    r"\blimited liability partnership\b",
    r"\bprofessional corporation\b",
    r"\bjoint stock company\b",
    # French multi-word
    r"\bsociete anonyme\b",
    r"\bsociete par actions simplifiee\b",
    r"\bsociete a responsabilite limitee\b",
    r"\bgroupement d interet economique\b",
    # Single-word suffixes (US / UK / International)
    r"\bcorporation\b",
    r"\bincorporated\b",
    r"\bcompany\b",
    r"\blimited\b",
    r"\bprivate\b",
    r"\bllc\b",
    r"\bllp\b",
    r"\bplc\b",
    r"\bcorp\b",
    r"\binc\b",
    r"\bltd\b",
    r"\bpvt\b",
    r"\bco\b",
    r"\bpc\b",
    # French legal entities
    r"\bsas\b",
    r"\bsarl\b",
    r"\bsasu\b",
    r"\beurl\b",
    r"\bsci\b",
    r"\bsnc\b",
    r"\bsa\b",
    r"\bscop\b",
    r"\bgie\b",
]

RE_LEGAL_SUFFIXES: Pattern[str] = re.compile(
    r"\b(?:" + "|".join(LEGAL_SUFFIX_PATTERNS) + r")\b", flags=re.IGNORECASE
)

# Comprehensive address abbreviations (US, India, France) & landmarks
ADDRESS_ABBREVIATIONS = {
    # US / General Street Suffixes
    r"\bst\b": "street",
    r"\bstr\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bbvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bhwy\b": "highway",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bfl\b": "floor",
    r"\bflr\b": "floor",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bunit\b": "suite",
    r"\bpkwy\b": "parkway",
    r"\bbldg\b": "building",
    r"\bp\.?o\.?\s*box\b": "pobox",
    # Indian Landmark & Municipal Formats
    r"\bopp\b": "opposite",
    r"\bopposite\b": "opposite",
    r"\bnr\b": "near",
    r"\bnear\b": "near",
    r"\bbeside\b": "beside",
    r"\bbehind\b": "behind",
    r"\badj\b": "adjacent",
    r"\badjacent\b": "adjacent",
    r"\bh\.?no\.?\b": "house",
    r"\bd\.?no\.?\b": "door",
    r"\bdoor\s*no\.?\b": "door",
    r"\bplot\s*no\.?\b": "plot",
    r"\bflat\s*no\.?\b": "flat",
    r"\bsy\.?\s*no\.?\b": "survey",
    r"\bsurvey\s*no\.?\b": "survey",
    r"\bsec\b": "sector",
    r"\bsector\b": "sector",
    r"\bph\b": "phase",
    r"\bblk\b": "block",
    r"\bno\.?\b": "number",
    # French Address Patterns (Official Challenge Test Country: France)
    r"\brue\b": "rue",
    r"\bbd\b": "boulevard",
    r"\bbld\b": "boulevard",
    r"\ball\b": "allee",
    r"\ballee\b": "allee",
    r"\bimp\b": "impasse",
    r"\bimpasse\b": "impasse",
    r"\brte\b": "route",
    r"\bchem\b": "chemin",
    r"\bch\b": "chemin",
    r"\bquai\b": "quai",
    r"\bfbg\b": "faubourg",
    r"\bfaubourg\b": "faubourg",
    r"\bcedex\b": "cedex",
    r"\bbp\b": "boitepostale",
    r"\bb\.?p\.?\b": "boitepostale",
    r"\bbat\b": "batiment",
    r"\bbatiment\b": "batiment",
    r"\betg\b": "etage",
    r"\betage\b": "etage",
}

STOPWORDS: Set[str] = {
    # English
    "the", "a", "an", "and", "of", "in", "at", "by", "for", "with", "on", "to",
    "from", "dba", "trade", "name",
    # French
    "de", "la", "le", "les", "du", "des", "en", "et", "au", "aux", "sur", "sous",
    "l", "d", "un", "une",
}


def unicode_to_ascii(text: str) -> str:
    """Normalize Unicode characters to ASCII representation."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("utf-8")


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace and strip margins."""
    return RE_WHITESPACE.sub(" ", text).strip()


def normalize_name(name: str) -> str:
    """Normalize business name preserving semantic tokens.

    Applies:
    - Unicode ASCII normalization
    - Lowercase
    - '&' -> 'and'
    - Common business abbreviation expansion
    - Punctuation removal
    - Whitespace normalization
    """
    if not name or not isinstance(name, str):
        return ""

    text = unicode_to_ascii(name).lower()
    text = RE_AMPERSAND.sub(" and ", text)
    text = RE_DBA.sub(" dba ", text)
    text = RE_PUNCTUATION.sub(" ", text)

    for pattern, replacement in NAME_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)

    return normalize_whitespace(text)


def strip_legal_suffixes(normalized_name: str) -> str:
    """Strip recognized legal entity suffixes to produce core name."""
    if not normalized_name:
        return ""
    text = RE_LEGAL_SUFFIXES.sub(" ", normalized_name)
    return normalize_whitespace(text)


def normalize_address(address: str) -> str:
    """Normalize business address preserving numbers and expanding abbreviations.

    Applies:
    - Unicode ASCII normalization
    - Lowercase
    - Common street/building abbreviation expansion
    - Punctuation removal (while preserving digits)
    - Whitespace normalization
    """
    if not address or not isinstance(address, str):
        return ""

    text = unicode_to_ascii(address).lower()
    text = RE_AMPERSAND.sub(" and ", text)
    text = RE_PUNCTUATION.sub(" ", text)

    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)

    return normalize_whitespace(text)


def extract_address_digits(address: str) -> str:
    """Extract ordered sequence of numeric tokens (house numbers, PINs, zip codes)."""
    if not address or not isinstance(address, str):
        return ""
    digits = RE_DIGITS.findall(address)
    # Filter out single-digit noise if multiple digits present, but keep distinctive numbers
    meaningful = [d for d in digits if len(d) >= 1]
    return " ".join(meaningful)


def extract_tokens(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize normalized text into clean word list."""
    if not text:
        return []
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def normalize_country(country: str) -> str:
    """Standardize country label (open-set, case-insensitive)."""
    if not country or not isinstance(country, str):
        return "UNKNOWN"
    return country.strip().upper()


class Preprocessor:
    """End-to-end preprocessor for source DataFrames."""

    def __init__(self) -> None:
        pass

    def process_dataframe(
        self, df: pd.DataFrame, verbose: bool = False
    ) -> pd.DataFrame:
        """Add standardized preprocessing columns to input DataFrame.

        Preserves:
        - business_name (original)
        - business_address (original)

        Adds:
        - normalized_name
        - name_without_legal_suffix
        - normalized_address
        - address_digits
        - normalized_country
        """
        out_df = df.copy()

        # Name processing
        raw_names = out_df["business_name"].fillna("").astype(str).tolist()
        norm_names = [normalize_name(n) for n in raw_names]
        stripped_names = [strip_legal_suffixes(n) for n in norm_names]

        # Address processing
        raw_addrs = out_df["business_address"].fillna("").astype(str).tolist()
        norm_addrs = [normalize_address(a) for a in raw_addrs]
        addr_digits = [extract_address_digits(a) for a in raw_addrs]

        # Country processing
        raw_countries = out_df["country"].fillna("").astype(str).tolist()
        norm_countries = [normalize_country(c) for c in raw_countries]

        out_df["normalized_name"] = norm_names
        out_df["name_without_legal_suffix"] = stripped_names
        out_df["normalized_address"] = norm_addrs
        out_df["address_digits"] = addr_digits
        out_df["normalized_country"] = norm_countries

        return out_df
