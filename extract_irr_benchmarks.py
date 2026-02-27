#!/usr/bin/env python3
"""Extract IRR benchmark data from images using OpenAI Vision API."""
import json
import base64
from pathlib import Path
import openai
from config import OPENAI_API_KEY, OPENAI_MODEL

# Map image files to asset classes - IRR images only
# Each asset class has 2 images: one for IRRs, one for Multiples
# We need the IRR images which show "IRRs by vintage" tables
# If first image fails, try the alternative
IMAGE_MAPPINGS = {
    "venture_capital": [
        "/Users/carolynenewman/.cursor/projects/Users-carolynenewman-Updates/assets/Screenshot_2026-01-29_at_12.48.23_PM-514b21c1-5228-46b7-abf8-17c4d5bd02c9.png",
        "/Users/carolynenewman/.cursor/projects/Users-carolynenewman-Updates/assets/Screenshot_2026-01-29_at_12.48.23_PM-a22b2f51-5571-48c7-bbcc-4b19298bd2f2.png"
    ],
    "real_estate": "/Users/carolynenewman/.cursor/projects/Users-carolynenewman-Updates/assets/Screenshot_2026-01-29_at_12.48.41_PM-f0eb6a73-76bd-4999-ac15-75cd4bd20785.png",
    "private_debt": "/Users/carolynenewman/.cursor/projects/Users-carolynenewman-Updates/assets/Screenshot_2026-01-29_at_12.49.13_PM-062399a6-a4a3-45b3-800a-f7665c9b6b18.png"
}

EXTRACTION_PROMPT = """Extract benchmark IRR (Internal Rate of Return) data from this investment performance table.

The table shows IRR benchmarks by vintage year (2000-2023). Extract the percentile values for each vintage:
- Top decile (90th percentile)
- Top quartile (75th percentile)  
- Median (50th percentile)
- Bottom quartile (25th percentile)
- Bottom decile (10th percentile)

Return ONLY valid JSON in this exact format:
{
  "2000": {
    "top_decile": 29.56,
    "top_quartile": 22.51,
    "median": 12.50,
    "bottom_quartile": 5.20,
    "bottom_decile": -3.02
  },
  "2001": { ... },
  ...
  "2023": { ... }
}

Rules:
- Values are numbers (floats), not strings
- Remove % signs (15.5% becomes 15.5)
- Negative values stay negative
- Include all vintages 2000-2023
- Return ONLY JSON, no markdown or explanation"""


def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_irr_data_from_image(image_path, asset_class):
    """Extract IRR benchmark data from an image using OpenAI Vision API."""
    print(f"\n📸 Processing {asset_class} IRR image: {Path(image_path).name}")
    
    try:
        # Encode image
        base64_image = encode_image(image_path)
        
        # Call OpenAI Vision API
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",  # Use vision-capable model
            messages=[
                {
                    "role": "system",
                    "content": "You are a data extraction specialist. Extract table data accurately and return only valid JSON."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4000
        )
        
        # Parse JSON response
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from markdown code blocks if present
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            response_text = response_text[json_start:json_end].strip()
        elif '```' in response_text:
            json_start = response_text.find('```') + 3
            json_end = response_text.find('```', json_start)
            response_text = response_text[json_start:json_end].strip()
        
        # Try to find JSON object boundaries if not in code blocks
        if not response_text.startswith('{'):
            json_start = response_text.find('{')
            if json_start != -1:
                json_end = response_text.rfind('}') + 1
                response_text = response_text[json_start:json_end]
        
        if not response_text or not response_text.startswith('{'):
            print(f"  ⚠️  No valid JSON found in response")
            print(f"  Response preview: {response_text[:500]}")
            return None
        
        irr_data = json.loads(response_text)
        print(f"  ✅ Extracted {len(irr_data)} vintages")
        return irr_data
        
    except Exception as e:
        print(f"  ❌ Error extracting data: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_benchmarks_json(asset_class, irr_data):
    """Update benchmarks.json with extracted IRR data."""
    benchmarks_file = Path("benchmarks.json")
    
    # Load existing benchmarks
    with open(benchmarks_file, 'r') as f:
        benchmarks = json.load(f)
    
    # Update IRR data
    if asset_class not in benchmarks:
        benchmarks[asset_class] = {"irrs_by_vintage": {}, "multiples_by_vintage": {}}
    
    benchmarks[asset_class]["irrs_by_vintage"] = irr_data
    
    # Save updated benchmarks
    with open(benchmarks_file, 'w') as f:
        json.dump(benchmarks, f, indent=2)
    
    print(f"  ✅ Updated benchmarks.json with {asset_class} IRRs")


def main():
    """Extract IRR data from all images and update benchmarks.json."""
    print("🔍 Extracting IRR benchmark data from images...")
    print("=" * 60)
    
    for asset_class, image_path_or_list in IMAGE_MAPPINGS.items():
        # Handle both single paths and lists of paths (for fallback)
        if isinstance(image_path_or_list, list):
            image_paths = image_path_or_list
        else:
            image_paths = [image_path_or_list]
        
        irr_data = None
        for image_path in image_paths:
            if not Path(image_path).exists():
                print(f"\n⚠️  Image not found: {Path(image_path).name}")
                continue
            
            # Extract IRR data
            irr_data = extract_irr_data_from_image(image_path, asset_class)
            
            if irr_data:
                break
            else:
                print(f"  ⚠️  Extraction failed, trying next image...")
        
        if irr_data:
            update_benchmarks_json(asset_class, irr_data)
        else:
            print(f"  ❌ All attempts failed for {asset_class}")
    
    print("\n" + "=" * 60)
    print("✅ Extraction complete!")
    print("\nRun 'python3 benchmark_lookup.py' to verify the data.")


if __name__ == "__main__":
    main()
