# AI Name Extraction Features Implementation

## Overview

This implementation adds AI-powered name extraction capabilities to the photoalbums system, leveraging AI text processing to extract names from visible text, labels, and contextual clues in photos. This enhances the face matching and caption generation workflows by providing additional name suggestions beyond what's available in the Cast database.

## Features Implemented

### 1. Enhanced Prompt Templates with Name Extraction

**Files Modified:**
- `photoalbums/lib/_caption_prompts.py`

**Changes:**
- Updated `_build_qwen_prompt()` to include name extraction instructions in the JSON schema
- Updated `_build_combined_qwen_prompt()` to include name extraction in combined OCR+caption workflows
- Added detailed instructions for AI models to extract names from visible text, labels, and contextual clues
- Specified confidence scoring and source attribution requirements

**Key Instructions Added:**
```
name_suggestions: an array of objects with 'name', 'confidence', 'source', and 'context' fields.
Extract names from visible text, labels, or contextual clues.
Include names that appear in signs, documents, clothing, or other visible elements.
Set confidence between 0.0 and 1.0 based on clarity and context.
Set source to 'visible_text', 'contextual_clue', or 'label'.
Set context to describe where the name was found.
```

### 2. Extended JSON Schemas

**Files Modified:**
- `photoalbums/lib/_caption_lmstudio.py`
- `photoalbums/lib/_caption_qwen.py`

**Changes:**
- Extended LM Studio JSON schema to include `name_suggestions` field
- Extended Qwen JSON schema to include `name_suggestions` field
- Added proper validation for name suggestion objects with required fields
- Updated parsing functions to handle the new `name_suggestions` field

**Schema Structure:**
```json
{
  "name_suggestions": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "confidence": {"type": "number"},
        "source": {"type": "string"},
        "context": {"type": "string"}
      },
      "required": ["name", "confidence", "source"],
      "additionalProperties": false
    }
  }
}
```

### 3. Enhanced CaptionDetails Class

**Files Modified:**
- `photoalbums/lib/_caption_lmstudio.py`

**Changes:**
- Added `name_suggestions` field to the `CaptionDetails` dataclass
- Updated `__post_init__` to initialize empty list if None
- Updated `__eq__` method to include name_suggestions in comparison
- Updated parsing functions to populate the name_suggestions field

### 4. Enhanced Face Matching with AI Hints

**Files Modified:**
- `photoalbums/lib/ai_people.py`

**Changes:**
- Added `match_image_with_ai_hints()` method to `CastPeopleMatcher` class
- Enhanced hint text generation using AI-derived name suggestions
- Improved face matching accuracy by incorporating AI name suggestions as hints
- Added filtering for high-confidence suggestions (confidence > 0.5)

**Integration Points:**
- AI name suggestions are converted to enhanced hint text
- Combined with existing hint text for improved face matching
- High-confidence suggestions are prioritized in the matching process

### 5. EnhancedCaptioner Class

**Files Created:**
- `photoalbums/lib/ai_name_suggestions.py`

**Features:**
- Integrated Cast database lookup with AI name extraction
- Text-based name extraction from OCR content using regex patterns
- Hyphen-separated name pattern recognition (common in Chinese albums)
- Chinese character name extraction
- Capitalized word filtering for potential names
- Combined suggestion deduplication and confidence-based sorting

**Key Methods:**
- `extract_names_from_text()`: Extracts names from OCR text using heuristics
- `get_cast_name_suggestions()`: Gets face-based name suggestions from Cast database
- `get_enhanced_caption()`: Generates captions with integrated name suggestions
- `get_combined_ocr_caption()`: Combined OCR and caption with name extraction

### 6. Text-Based Name Extraction

**Implementation Details:**
- **Hyphen-separated names**: Detects patterns like "leslie-tommy-robert" and converts to "Leslie Tommy Robert"
- **Capitalized words**: Identifies potential names while filtering common non-name words
- **Chinese characters**: Extracts Chinese character sequences as potential names
- **Confidence scoring**: Assigns confidence based on pattern type and context

**Confidence Levels:**
- Hyphen-separated patterns: 0.8
- Chinese character sequences: 0.7
- Capitalized words: 0.6

### 7. Comprehensive Test Suite

**Files Created:**
- `test_basic_ai_name_suggestions.py`
- `test_simple_ai_name_suggestions.py`
- `test_ai_name_suggestions.py`

**Test Coverage:**
- Enhanced prompt template validation
- JSON schema structure verification
- CaptionDetails class functionality
- Qwen and LM Studio JSON parsing
- Integration testing of all components

## Usage Examples

### Basic Usage

```python
from photoalbums.lib.ai_name_suggestions import create_enhanced_captioner

# Create enhanced captioner with Cast database integration
captioner = create_enhanced_captioner(cast_store_dir="path/to/cast/store")

# Get enhanced caption with name extraction
caption, location, name_suggestions = captioner.get_enhanced_caption(
    image_path="photo.jpg",
    source_path="album/page.jpg",
    album_title="Family Photos"
)

# Get combined OCR and caption
ocr_text, caption, name_suggestions = captioner.get_combined_ocr_caption(
    image_path="photo.jpg",
    source_path="album/page.jpg"
)
```

### Enhanced Face Matching

```python
from photoalbums.lib.ai_people import CastPeopleMatcher

# Create face matcher
matcher = CastPeopleMatcher(cast_store_dir="path/to/cast/store")

# Get AI name suggestions first
ai_suggestions = extract_ai_name_suggestions(image_path)

# Use enhanced matching with AI hints
face_matches = matcher.match_image_with_ai_hints(
    image_path="photo.jpg",
    ai_name_suggestions=ai_suggestions
)
```

## Benefits

1. **Improved Face Matching Accuracy**: AI-derived name suggestions provide additional hints for face matching algorithms
2. **Enhanced Caption Quality**: Captions include extracted names from visible text and contextual clues
3. **Better OCR Utilization**: Leverages OCR text beyond simple text extraction
4. **Pattern Recognition**: Handles Chinese name patterns and hyphenated names common in photo albums
5. **Confidence-Based Filtering**: Only high-confidence suggestions are used to avoid noise
6. **Integration Ready**: Seamlessly integrates with existing Cast database and face matching workflows

## Technical Architecture

```
AI Name Extraction System
├── Enhanced Prompt Templates
│   ├── Qwen prompts with name extraction
│   └── LM Studio prompts with name extraction
├── JSON Schema Extensions
│   ├── name_suggestions field
│   └── Structured name suggestion objects
├── Enhanced CaptionDetails
│   ├── name_suggestions field
│   └── Updated parsing logic
├── EnhancedCaptioner
│   ├── Cast database integration
│   ├── Text-based name extraction
│   └── Combined suggestion processing
└── Enhanced Face Matching
    ├── AI hint integration
    └── Improved matching accuracy
```

## Future Enhancements

1. **Machine Learning Integration**: Train models specifically for name extraction from photo text
2. **Multi-language Support**: Expand name extraction for different languages and writing systems
3. **Contextual Understanding**: Improve context analysis for better name disambiguation
4. **User Feedback Loop**: Allow users to confirm/correct AI suggestions to improve accuracy
5. **Batch Processing**: Process multiple images in batch for improved efficiency

## Testing

All features have been thoroughly tested with:
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Schema validation tests
- Error handling tests

Run tests with:
```bash
python tests/ai_name_suggestions/test_basic_ai_name_suggestions.py
```

## Conclusion

This implementation successfully adds AI-powered name extraction capabilities to the photoalbums system, providing enhanced face matching accuracy and improved caption generation through intelligent text analysis and Cast database integration.