from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ai_people import CastPeopleMatcher, PersonMatch
from .ai_ocr import extract_ocr_text
from ._caption_lmstudio import CaptionDetails
from ._caption_qwen import QwenLocalCaptioner
from ._caption_prompts import _build_qwen_prompt, _build_combined_qwen_prompt


class EnhancedCaptioner:
    """Enhanced captioner that integrates Cast database lookup with AI name extraction."""
    
    def __init__(
        self,
        *,
        cast_store_dir: str | Path,
        min_similarity: float = 0.40,
        min_margin: float = 0.06,
        min_face_size: int = 40,
        max_faces: int = 40,
        skip_artwork: bool = True,
        min_face_quality: float = 0.20,
        min_sample_count: int = 2,
        ignore_similarity: float = 0.88,
        review_top_k: int = 5,
    ):
        self.cast_matcher = CastPeopleMatcher(
            cast_store_dir=cast_store_dir,
            min_similarity=min_similarity,
            min_margin=min_margin,
            min_face_size=min_face_size,
            max_faces=max_faces,
            skip_artwork=skip_artwork,
            min_face_quality=min_face_quality,
            min_sample_count=min_sample_count,
            ignore_similarity=ignore_similarity,
            review_top_k=review_top_k,
        )
        self.qwen_captioner = QwenLocalCaptioner()
    
    def extract_names_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract potential names from OCR text using simple heuristics."""
        import re
        
        if not text.strip():
            return []
        
        suggestions = []
        text_lower = text.lower()
        
        # Look for hyphen-separated names (common in Chinese albums)
        hyphen_names = re.findall(r'([a-zA-Z\u4e00-\u9fff]+(?:-[a-zA-Z\u4e00-\u9fff]+)+)', text)
        for name in hyphen_names:
            suggestions.append({
                "name": name.replace('-', ' ').title(),
                "confidence": 0.8,
                "source": "visible_text",
                "context": f"Hyphen-separated name pattern in OCR text: {name}"
            })
        
        # Look for capitalized words that might be names
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        for word in words:
            # Filter out common non-name words
            if word.lower() not in {'time', 'date', 'place', 'city', 'country', 'province'}:
                suggestions.append({
                    "name": word,
                    "confidence": 0.6,
                    "source": "visible_text",
                    "context": f"Capitalized word in OCR text: {word}"
                })
        
        # Look for Chinese characters that might be names
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        for chars in chinese_chars:
            if len(chars) >= 2:  # At least 2 characters for a name
                suggestions.append({
                    "name": chars,
                    "confidence": 0.7,
                    "source": "visible_text",
                    "context": f"Chinese characters in OCR text: {chars}"
                })
        
        return suggestions
    
    def get_cast_name_suggestions(self, image_path: str | Path) -> List[Dict[str, Any]]:
        """Get name suggestions from Cast database based on detected faces."""
        try:
            face_matches = self.cast_matcher.match_image(image_path)
            suggestions = []
            
            for match in face_matches:
                suggestions.append({
                    "name": match.name,
                    "confidence": match.certainty,
                    "source": "cast_database",
                    "context": f"Face match with score {match.score:.3f}"
                })
            
            return suggestions
        except Exception:
            return []
    
    def get_enhanced_caption(
        self,
        image_path: str | Path,
        *,
        source_path: str | Path | None = None,
        album_title: str = "",
        printed_album_title: str = "",
        photo_count: int = 1,
        is_cover_page: bool = False,
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Get enhanced caption with AI name extraction and Cast database integration."""
        # First, extract OCR text
        try:
            ocr_text = extract_ocr_text(image_path)
        except Exception:
            ocr_text = ""
        
        # Get Cast database suggestions
        cast_suggestions = self.get_cast_name_suggestions(image_path)
        
        # Get text-based name suggestions
        text_suggestions = self.extract_names_from_text(ocr_text)
        
        # Combine all suggestions
        all_suggestions = cast_suggestions + text_suggestions
        
        # Get AI caption with name extraction
        prompt = _build_qwen_prompt(
            people=[s["name"] for s in cast_suggestions],
            objects=[],
            ocr_text=ocr_text,
            source_path=source_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            photo_count=photo_count,
            is_cover_page=is_cover_page,
        )
        
        caption_details = self.qwen_captioner.describe(image_path, prompt=prompt)
        
        # Combine AI suggestions with our suggestions
        combined_suggestions = list(caption_details.name_suggestions) + all_suggestions
        
        # Remove duplicates and sort by confidence
        seen = set()
        unique_suggestions = []
        for suggestion in combined_suggestions:
            name = suggestion.get("name", "").lower()
            if name and name not in seen:
                seen.add(name)
                unique_suggestions.append(suggestion)
        
        unique_suggestions.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return caption_details.text, caption_details.location_name, unique_suggestions
    
    def get_combined_ocr_caption(
        self,
        image_path: str | Path,
        *,
        source_path: str | Path | None = None,
        album_title: str = "",
        printed_album_title: str = "",
        photo_count: int = 1,
        is_cover_page: bool = False,
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Get combined OCR and caption with name extraction in a single inference."""
        # Get Cast database suggestions for hinting
        cast_suggestions = self.get_cast_name_suggestions(image_path)
        cast_names = [s["name"] for s in cast_suggestions]
        
        # Get combined OCR and caption
        ocr_text, caption, ai_suggestions = self.qwen_captioner.describe_combined(
            image_path,
            people=cast_names,
            objects=[],
            source_path=source_path,
            album_title=album_title,
            printed_album_title=printed_album_title,
            photo_count=photo_count,
            is_cover_page=is_cover_page,
        )
        
        # Get text-based name suggestions from OCR
        text_suggestions = self.extract_names_from_text(ocr_text)
        
        # Combine all suggestions
        combined_suggestions = list(ai_suggestions) + cast_suggestions + text_suggestions
        
        # Remove duplicates and sort by confidence
        seen = set()
        unique_suggestions = []
        for suggestion in combined_suggestions:
            name = suggestion.get("name", "").lower()
            if name and name not in seen:
                seen.add(name)
                unique_suggestions.append(suggestion)
        
        unique_suggestions.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return ocr_text, caption, unique_suggestions


def create_enhanced_captioner(
    cast_store_dir: str | Path,
    **kwargs: Any
) -> EnhancedCaptioner:
    """Create an enhanced captioner with Cast database integration."""
    return EnhancedCaptioner(cast_store_dir=cast_store_dir, **kwargs)