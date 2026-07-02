#!/usr/bin/env python3
"""
Classify model architecture type in OpenVINO IR model.

Classifies models as:
  - Dense (traditional feedforward)
  - MOE (Mixture of Experts)
  - Multimodal (vision-language, etc.)

Usage:
    python classify_model_architecture.py -m /path/to/model/dir
    python classify_model_architecture.py -m /path/to/openvino_language_model.xml
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import NamedTuple


class ModelTypeResult(NamedTuple):
    model_type: str  # "MOE", "Dense", "Multimodal", "Unknown"
    confidence: str  # "high", "medium", "low"
    indicators: list[str]
    model_path: str
    details: dict  # additional metadata


MOE_KEYWORDS = {
    'shared_expert': r'\.mlp\.shared_expert',
    'experts_layer': r'\.experts\b',
    'router_gate': r'\.router|\.gate(?!_proj)',
    'expert_gate': r'expert.*gate|gate.*expert',
    'variadic_split': r'VariadicSplit',  # Pattern for splitting to multiple experts
    'moe_layer': r'MoE|mixture_of_experts',
}

MULTIMODAL_KEYWORDS = {
    'vision_encoder': r'vision.*encoder|image.*encoder|clip|vision_',
    'audio_encoder': r'audio.*encoder|whisper',
    'embedding_vision': r'embed.*vision|vision.*embed',
}


def extract_model_name(model_path: Path) -> str:
    """Extract model name from path."""
    # Try to get from path components
    path_str = str(model_path)
    
    # Look for common model names in the path
    for part in reversed(path_str.split('/')):
        part_lower = part.lower()
        if any(x in part_lower for x in ['qwen', 'gemma', 'llama', 'mistral', 'falcon', 'phi']):
            return part
    
    return model_path.name or 'unknown'


def find_xml_file(model_path: Path) -> Path | None:
    """Find openvino_language_model.xml in model directory."""
    if model_path.is_file() and model_path.name.endswith('.xml'):
        return model_path
    
    if model_path.is_dir():
        xml_path = model_path / 'openvino_language_model.xml'
        if xml_path.exists():
            return xml_path
    
    return None


def find_config_file(model_path: Path) -> Path | None:
    """Find config.json in model directory."""
    if model_path.is_file():
        config_path = model_path.parent / 'config.json'
    else:
        config_path = model_path / 'config.json'
    
    if config_path.exists():
        return config_path
    
    return None


def read_config_json(config_path: Path) -> dict:
    """Read and parse config.json."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {}


def detect_moe_from_config(config: dict) -> tuple[bool, str]:
    """Detect MOE from config.json fields."""
    # Check explicit enable_moe_block flag
    if config.get('enable_moe_block', False):
        return True, 'enable_moe_block=true'
    
    # Check nested text_config (for Qwen, Llama, Gemma models)
    text_config = config.get('text_config', {})
    
    # Check for num_experts field (MOE indicator)
    num_experts = config.get('num_experts') or text_config.get('num_experts')
    
    # Try different field names for active experts
    num_active = (config.get('num_experts_per_tok') or 
                  text_config.get('num_experts_per_tok') or
                  text_config.get('top_k_experts'))  # Gemma uses top_k_experts
    
    if num_experts and num_active:
        return True, f'experts: {num_experts} total, {num_active} active'
    
    if num_experts:
        return True, f'num_experts: {num_experts}'
    
    return False, ''


def detect_multimodal_from_config(config: dict) -> bool:
    """Detect multimodal model from config."""
    # Check for vision/audio specific fields
    vision_fields = ['vision_config', 'image_encoder', 'vision_encoder_config']
    audio_fields = ['audio_config', 'audio_encoder', 'audio_processor']
    
    has_vision = any(k in config for k in vision_fields)
    has_audio = any(k in config for k in audio_fields)
    
    return has_vision or has_audio


def classify_model_architecture(model_path: Path) -> ModelTypeResult:
    """Classify complete model architecture type."""
    
    model_name = extract_model_name(model_path)
    details = {'model_name': model_name}
    indicators = []
    
    # Step 1: Try to read config.json
    config_path = find_config_file(model_path)
    config = read_config_json(config_path) if config_path else {}
    
    # Step 2: Check for MOE from config
    is_moe_config, moe_reason = detect_moe_from_config(config)
    if is_moe_config:
        indicators.append(f'config.json: {moe_reason}')
    
    # Step 3: Check for multimodal from config
    is_multimodal_config = detect_multimodal_from_config(config)
    if is_multimodal_config:
        indicators.append('config.json: multimodal fields found')
    
    # Step 4: Analyze XML for additional patterns
    xml_path = find_xml_file(model_path)
    xml_indicators = []
    
    if xml_path:
        try:
            with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
                xml_content = f.read()
            
            # Check MOE keywords in XML
            moe_count = 0
            for keyword_name, pattern in MOE_KEYWORDS.items():
                matches = re.findall(pattern, xml_content, re.IGNORECASE)
                if matches:
                    moe_count += 1
                    xml_indicators.append(f'{keyword_name}: {len(matches)} occ.')
            
            # Check multimodal keywords in XML
            mm_count = 0
            for keyword_name, pattern in MULTIMODAL_KEYWORDS.items():
                matches = re.findall(pattern, xml_content, re.IGNORECASE)
                if matches:
                    mm_count += 1
                    xml_indicators.append(f'{keyword_name}: {len(matches)} occ.')
            
            if xml_indicators:
                indicators.extend(xml_indicators)
        
        except Exception as e:
            indicators.append(f'XML parse error: {str(e)[:50]}')
    
    # Step 5: Determine final model type and confidence
    model_type = 'Unknown'
    confidence = 'low'
    
    # Build type string based on detected features
    type_parts = []
    
    if is_moe_config or any('expert' in ind.lower() or 'router' in ind.lower() for ind in indicators):
        type_parts.append('MOE')
    
    if is_multimodal_config or any('vision' in ind.lower() or 'audio' in ind.lower() for ind in indicators):
        type_parts.append('Multimodal')
    
    # Combine type parts or default to Dense
    if type_parts:
        model_type = '+'.join(type_parts)
        # Confidence scoring
        if is_moe_config and is_multimodal_config:
            confidence = 'high'
        elif is_moe_config or is_multimodal_config:
            confidence = 'high' if len(indicators) >= 2 else 'medium'
        else:
            confidence = 'medium' if indicators else 'low'
    else:
        model_type = 'Dense'
        confidence = 'high' if config else 'medium'
    
    # Store additional details
    if config:
        text_config = config.get('text_config', {})
        
        if 'num_hidden_layers' in text_config:
            details['hidden_layers'] = text_config.get('num_hidden_layers')
        if 'hidden_size' in text_config:
            details['hidden_size'] = text_config.get('hidden_size')
        if is_moe_config:
            details['num_experts'] = config.get('num_experts') or text_config.get('num_experts')
            # Try different field names for active experts
            active = (config.get('num_experts_per_tok') or 
                     text_config.get('num_experts_per_tok') or
                     text_config.get('top_k_experts'))
            if active:
                details['num_experts_per_tok'] = active
        if config.get('architectures'):
            details['architecture'] = config['architectures'][0]
    
    return ModelTypeResult(
        model_type=model_type,
        confidence=confidence,
        indicators=indicators,
        model_path=str(model_path),
        details=details
    )


def main():
    parser = argparse.ArgumentParser(
        description='Classify model architecture type in OpenVINO IR model'
    )
    parser.add_argument(
        '-m', '--model',
        type=Path,
        required=True,
        help='Path to model directory or openvino_language_model.xml file'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed analysis'
    )
    
    args = parser.parse_args()
    
    # Classify model architecture
    result = classify_model_architecture(args.model)
    
    # Output
    print(f"\n{'='*70}")
    print(f"Model Architecture Type Classification")
    print(f"{'='*70}")
    print(f"Model Name: {result.details.get('model_name', 'unknown')}")
    print(f"Model Type: {result.model_type}")
    print(f"Confidence: {result.confidence.upper()}")
    print(f"\nIndicators Found ({len(result.indicators)}):")
    for indicator in result.indicators:
        print(f"  • {indicator}")
    
    if result.details:
        print(f"\nAdditional Details:")
        for key, value in result.details.items():
            if key != 'model_name' and value is not None:
                print(f"  {key}: {value}")
    
    print(f"{'='*70}")
    
    # Summary with emoji
    type_symbol = {
        'MOE': '🔀',
        'Dense': '🧠',
        'Multimodal': '📸',
        'MOE+Multimodal': '🔀📸',
        'Multimodal+MOE': '📸🔀',
        'Unknown': '❓'
    }
    
    # Try exact match first, then partial match
    symbol = type_symbol.get(result.model_type, '?')
    if symbol == '?':
        for key in type_symbol:
            if key in result.model_type:
                symbol = type_symbol[key]
                break
    
    print(f"\n✓ {symbol} {result.model_type} model ({result.confidence} confidence)\n")
    
    return 0 if result.model_type != 'Unknown' else 1


if __name__ == '__main__':
    exit(main())
