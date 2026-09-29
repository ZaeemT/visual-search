"""Facts about a match, extracted before any text is written.

An explanation has to be grounded in something actually observed, so this
module produces the only material an explanation is allowed to use:

  query side  — dominant colours measured from the uploaded pixels, and the
                category the embedding model itself predicts for the image
  item side   — the catalogue's own annotations for the retrieved product
  shared      — what the two have in common, and how close the match is

Nothing here is invented or copied from a product listing. Colours come from
the pixels, the predicted category comes from the model's text tower, and the
item facts come from the dataset's annotations.

Colours, fit, length, materials and styles are mapped to English through the
dataset's own closed vocabularies (13 colours, 5 fits, 4 lengths), so those are
exact lookups rather than translation guesses — and the English colour names
are what the groundedness check verifies against later. Category names are a
long open-ended list, so they are left in the original wording and translated
at generation time instead.

"другая"/"другой" means *unspecified*; it is dropped rather than reported,
because "other fit" would turn a missing value into a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

UNSPECIFIED = {"другая", "другое", "другой"}

# Anchor colours for naming pixels, keyed by the dataset's own colour words so
# a measured colour and an annotated colour are directly comparable.
COLOR_ANCHORS: dict[str, tuple[int, int, int]] = {
    "белый": (245, 245, 245),
    "черный": (25, 25, 25),
    "серый": (128, 128, 128),
    "красный": (200, 40, 40),
    "розовый": (240, 150, 180),
    "оранжевый": (240, 140, 40),
    "желтый": (235, 215, 70),
    "зеленый": (70, 150, 70),
    "голубой": (120, 190, 230),
    "синий": (45, 70, 165),
    "фиолетовый": (140, 70, 170),
    "коричневый": (125, 85, 55),
}

COLOR_EN = {
    "белый": "white",
    "черный": "black",
    "серый": "grey",
    "красный": "red",
    "розовый": "pink",
    "оранжевый": "orange",
    "желтый": "yellow",
    "зеленый": "green",
    "голубой": "light blue",
    "синий": "blue",
    "фиолетовый": "purple",
    "коричневый": "brown",
}

FIT_EN = {
    "свободная": "loose",
    "облегающая": "fitted",
    "приталенная": "tailored",
    "оверсайз": "oversized",
}

LENGTH_EN = {"длинная": "long", "средняя": "mid-length", "короткая": "short"}

SEX_EN = {"женский": "women's", "мужской": "men's", "унисекс": "unisex"}

MATERIAL_EN = {
    "хлопок": "cotton",
    "ткань": "fabric",
    "полиэстер": "polyester",
    "металл": "metal",
    "кожа": "leather",
    "шелк": "silk",
    "трикотаж": "knit",
    "джинсовая ткань": "denim",
    "синтепух": "padded synthetic",
    "кружево": "lace",
    "замша": "suede",
    "шерсть": "wool",
    "лен": "linen",
    "атлас": "satin",
    "резина": "rubber",
    "пластик": "plastic",
    "стекло": "glass",
    "дерево": "wood",
    "мех": "fur",
    "шифон": "chiffon",
    "вельвет": "corduroy",
}

STYLE_EN = {
    "современный": "modern",
    "повседневный": "casual",
    "классический": "classic",
    "вечерний": "evening",
    "спортивный": "sporty",
    "женственный": "feminine",
    "пляжный": "beach",
    "деловой": "business",
    "восточный": "traditional",
    "минимализм": "minimal",
    "винтажный": "vintage",
    "романтический": "romantic",
    "элегантный": "elegant",
    "богемный": "bohemian",
    "уличный": "streetwear",
}


def _translate(values, lookup: dict[str, str]) -> list[str]:
    """Map annotation words to English, dropping unspecified ones and duplicates."""
    out = []
    for value in values or []:
        word = (value or "").strip().lower()
        if not word or word in UNSPECIFIED:
            continue
        english = lookup.get(word, word)
        if english not in out:
            out.append(english)
    return out


def _translate_one(value, lookup: dict[str, str]) -> str | None:
    result = _translate([value], lookup)
    return result[0] if result else None


def dominant_colors(image: Image.Image, top: int = 2) -> list[str]:
    """Name the main colours of the garment, measured from the pixels.

    Two deliberate choices:

    * Only the central half of the frame is sampled. These are studio product
      shots on a plain backdrop, so the full frame's dominant colour is usually
      the backdrop, not the product.
    * Pixels are snapped to the dataset's own colour vocabulary, so a measured
      colour can be compared directly against an annotated one.
    """
    width, height = image.size
    box = (width // 4, height // 4, width - width // 4, height - height // 4)
    crop = image.crop(box).convert("RGB")
    crop.thumbnail((64, 64))  # ~4k pixels is plenty, and keeps this near 1ms

    pixels = np.asarray(crop, dtype=np.float32).reshape(-1, 3)
    anchors = np.array(list(COLOR_ANCHORS.values()), dtype=np.float32)
    names = list(COLOR_ANCHORS)

    # Nearest anchor per pixel. Plain Euclidean RGB distance is crude next to a
    # perceptual space, but the anchors are far apart, so it is enough to tell
    # "mostly red" from "mostly blue".
    distances = ((pixels[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2)
    counts = np.bincount(distances.argmin(axis=1), minlength=len(names))

    share = counts / max(counts.sum(), 1)
    ranked = [(names[i], float(share[i])) for i in np.argsort(-share) if share[i] >= 0.15]
    return [COLOR_EN[name] for name, _ in ranked[:top]]


def closeness(score: float) -> str:
    """Describe how close a match is without exposing the number.

    Thresholds come from the observed score distribution for this model: a
    near-duplicate lands near 0.97, a same-category match near 0.85, a loose
    visual match near 0.6. They are model-specific, so they live beside the
    code rather than in the prompt.
    """
    if score >= 0.90:
        return "almost identical"
    if score >= 0.80:
        return "very close"
    if score >= 0.65:
        return "close"
    return "loose"


@dataclass
class QuerySignals:
    """What was observed about the uploaded image itself."""

    dominant_colors: list[str] = field(default_factory=list)
    predicted_category: str | None = None


@dataclass
class MatchSignals:
    """Everything an explanation for one result is allowed to draw on."""

    query: QuerySignals
    category: str
    colors: list[str]
    fit: str | None
    length: str | None
    sex: str | None
    materials: list[str]
    styles: list[str]
    shared_colors: list[str]
    same_category: bool
    similarity: float

    def as_facts(self) -> dict:
        """The fact sheet handed to the phrasing step. No scores, no internals."""
        return {
            "photo_colours": self.query.dominant_colors,
            "photo_looks_like": self.query.predicted_category,
            "item_category": self.category,
            "item_colours": self.colors,
            "item_fit": self.fit,
            "item_length": self.length,
            "item_materials": self.materials,
            "item_styles": self.styles,
            "shared_colours": self.shared_colors,
            "same_category": self.same_category,
            "match_strength": closeness(self.similarity),
        }


def build_match_signals(query: QuerySignals, item: dict, score: float) -> MatchSignals:
    """Combine what was seen in the photo with what the catalogue records."""
    colors = _translate(item.get("colors"), COLOR_EN)
    category = (item.get("name") or "item").strip()

    return MatchSignals(
        query=query,
        category=category,
        colors=colors,
        fit=_translate_one(item.get("fit"), FIT_EN),
        length=_translate_one(item.get("length"), LENGTH_EN),
        sex=_translate_one(item.get("sex"), SEX_EN),
        materials=_translate(item.get("materials"), MATERIAL_EN),
        styles=_translate(item.get("styles"), STYLE_EN),
        shared_colors=[c for c in colors if c in query.dominant_colors],
        same_category=bool(
            query.predicted_category
            and query.predicted_category.strip().lower() == category.lower()
        ),
        similarity=score,
    )
