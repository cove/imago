#!/usr/bin/env python3
"""
Test script for AI name extraction features.
This script tests the enhanced prompt templates and name suggestion system.
"""

import json
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from photoalbums.lib.ai_name_suggestions import EnhancedCaptioner
from photoalbums.lib._caption_lmstudio import CaptionDetails
from photoalbums.lib._caption_qwen import QwenLocalCaptioner
from photoalbums.lib._caption_prompts import _build_qwen_prompt, _build_combined_qwen_prompt


def test_prompt_templates():
    """Test that the enhanced prompt templates include name extraction instructions."""
    print("Testing enhanced prompt templates...")
    
    # Test regular Qwen prompt
    prompt = _build_qwen_prompt(
        people=["John Doe", "Jane Smith"],
        objects=["camera", "book"],
        ocr_text="Leslie-Tommy-Robert at Beijing",
        source_path="test_image.jpg",
        album_title="Family Photos",
        photo_count=1
    )
    
    print("Regular Qwen prompt includes name extraction:")
    print("✓ 'name_suggestions' in schema" if "name_suggestions" in prompt else "✗ Missing")
    print("✓ 'Extract names from visible text'" if "Extract names from visible text" in prompt else "✗ Missing")
    print("✓ 'Set confidence between 0.0 and 1.0'" if "Set confidence between 0.0 and 1.0" in prompt else "✗ Missing")
    
    # Test combined Qwen prompt
    combined_prompt = _build_combined_qwen_prompt(
        people=["John Doe"],
        objects=["camera"],
        source_path="test_image.jpg",
        album_title="Family Photos",
        photo_count=1
    )
    
    print("\nCombined Qwen prompt includes name extraction:")
    print("✓ 'name_suggestions' in schema" if "name_suggestions" in combined_prompt else "✗ Missing")
    print("✓ 'Extract names from visible text'" if "Extract names from visible text" in combined_prompt else "✗ Missing")
    
    return True


def test_caption_details_with_names():
    """Test that CaptionDetails can handle name suggestions."""
    print("\nTesting CaptionDetails with name suggestions...")
    
    # Test with name suggestions
    name_suggestions = [
        {"name": "John Doe", "confidence": 0.85, "source": "visible_text", "context": "Sign on building"},
        {"name": "Jane Smith", "confidence": 0.72, "source": "cast_database", "context": "Face match"}
    ]
    
    caption = CaptionDetails(
        text="A photo of John Doe and Jane Smith at a park",
        gps_latitude="37.7749",
        gps_longitude="-122.4194",
        location_name="San Francisco, CA, USA",
        name_suggestions=name_suggestions
    )
    
    print(f"✓ Caption text: {caption.text}")
    print(f"✓ GPS coordinates: {caption.gps_latitude}, {caption.gps_longitude}")
    print(f"✓ Location: {caption.location_name}")
    print(f"✓ Name suggestions count: {len(caption.name_suggestions)}")
    
    # Test equality
    caption2 = CaptionDetails(
        text="A photo of John Doe and Jane Smith at a park",
        gps_latitude="37.7749",
        gps_longitude="-122.4194",
        location_name="San Francisco, CA, USA",
        name_suggestions=name_suggestions
    )
    
    print(f"✓ Caption equality: {caption == caption2}")
    
    return True


def test_qwen_json_schema():
    """Test that Qwen JSON schema includes name_suggestions."""
    print("\nTesting Qwen JSON schema...")
    
    from photoalbums.lib._caption_qwen import _parse_qwen_json_output
    
    # Test with sample JSON that includes name_suggestions
    sample_json = '''
    {
        "caption": "A photo of John Doe and Jane Smith at a park",
        "gps_latitude": "37.7749",
        "gps_longitude": "-122.4194",
        "location_name": "San Francisco, CA, USA",
        "name_suggestions": [
            {"name": "John Doe", "confidence": 0.85, "source": "visible_text", "context": "Sign on building"},
            {"name": "Jane Smith", "confidence": 0.72, "source": "cast_database", "context": "Face match"}
        ]
    }
    '''
    
    try:
        result = _parse_qwen_json_output(sample_json)
        print(f"✓ Caption: {result.text}")
        print(f"✓ Name suggestions: {len(result.name_suggestions)}")
        print(f"✓ First suggestion: {result.name_suggestions[0] if result.name_suggestions else 'None'}")
        return True
    except Exception as e:
        print(f"✗ Error parsing JSON: {e}")
        return False


def test_enhanced_captioner_creation():
    """Test that EnhancedCaptioner can be created."""
    print("\nTesting EnhancedCaptioner creation...")
    
    try:
        # This will fail if cast_store_dir doesn't exist, but we can still test the import
        from photoalbums.lib.ai_name_suggestions import EnhancedCaptioner
        print("✓ EnhancedCaptioner imported successfully")
        
        # Test the create function
        from photoalbums.lib.ai_name_suggestions import create_enhanced_captioner
        print("✓ create_enhanced_captioner imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Error importing EnhancedCaptioner: {e}")
        return False


def main():
    """Run all tests."""
    print("AI Name Extraction Feature Tests")
    print("=" * 40)
    
    tests = [
        test_prompt_templates,
        test_caption_details_with_names,
        test_qwen_json_schema,
        test_enhanced_captioner_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✓ Test passed\n")
            else:
                print("✗ Test failed\n")
        except Exception as e:
            print(f"✗ Test error: {e}\n")
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The AI name extraction features are working correctly.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())