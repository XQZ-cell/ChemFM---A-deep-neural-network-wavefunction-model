# Standard library imports
import sys
import yaml
import ml_collections
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
import argparse

# Insert project root path
sys.path.insert(0, str(Path(__file__).parent))

# Local module imports
from src.main import main
from src.config import base_config
from src.utils import system


def load_yaml_config(yaml_path: str) -> ml_collections.ConfigDict:
    """Load configuration from YAML file as ConfigDict."""
    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Convert to ConfigDict to maintain consistent type with cfg
    return dict_to_configdict(config_data)


def dict_to_configdict(data: Any) -> Any:
    """Recursively convert dictionary to ConfigDict."""
    if isinstance(data, dict):
        # Create ConfigDict
        config_dict = ml_collections.ConfigDict()
        
        for key, value in data.items():
            # Recursively process nested structures
            config_dict[key] = dict_to_configdict(value)
        
        return config_dict
    elif isinstance(data, list):
        # Recursively process elements in the list
        return [dict_to_configdict(item) for item in data]
    else:
        # Basic types directly return
        return data


def configdict_to_dict(config_dict: ml_collections.ConfigDict) -> Dict[str, Any]:
    """Recursively convert ConfigDict to dictionary for YAML serialization."""
    result = {}
    
    for key in config_dict.keys():
        value = config_dict[key]
        
        if isinstance(value, ml_collections.ConfigDict):
            result[key] = configdict_to_dict(value)
        elif isinstance(value, list):
            # Recursively process ConfigDict in the list
            result[key] = [
                configdict_to_dict(item) if isinstance(item, ml_collections.ConfigDict) 
                else item for item in value
            ]
        else:
            result[key] = value
    
    return result


def apply_configdict_to_config(cfg: ml_collections.ConfigDict, yaml_config: ml_collections.ConfigDict) -> None:
    """
    Apply YAML ConfigDict to existing config object.
    Handles special cases like molecule structure separately.
    """
    # First, process the molecule section (if exists)
    if 'molecule' in yaml_config:
        process_molecule_section(cfg, yaml_config.molecule)
        # Remove molecule from yaml_config to avoid duplicate processing
        del yaml_config['molecule']
    
    # Recursively update other configurations
    merge_configdicts(cfg, yaml_config)


def merge_configdicts(target: ml_collections.ConfigDict, source: ml_collections.ConfigDict) -> None:
    """
    Recursively merge source ConfigDict into target ConfigDict.
    """
    for key in source.keys():
        source_value = source[key]
        
        # If the key does not exist in target, add it directly
        if key not in target:
            target[key] = source_value
            continue
        
        target_value = target[key]
        
        # If both are ConfigDict, merge recursively
        if isinstance(source_value, ml_collections.ConfigDict) and isinstance(target_value, ml_collections.ConfigDict):
            merge_configdicts(target_value, source_value)
        else:
            # Otherwise, directly overwrite
            target[key] = source_value


def process_molecule_section(
        cfg: ml_collections.ConfigDict, 
        molecule_data: Union[Dict, ml_collections.ConfigDict]
        ) -> None:
    """Process molecule section from YAML."""
    # Ensure molecule_data is in dictionary form
    if isinstance(molecule_data, ml_collections.ConfigDict):
        molecule_dict = configdict_to_dict(molecule_data)
    else:
        molecule_dict = molecule_data
    
    # Modified: Check for required keys, raise error if missing
    required_keys = ['coords', 'atoms']
    for key in required_keys:
        if key not in molecule_dict:
            raise KeyError(f"Missing required key in molecule section: '{key}'")
    
    coords = molecule_dict['coords']
    atoms_symbols = molecule_dict['atoms']
    unit = molecule_dict.get('unit', 'bohr')
    
    # Validate input
    if len(coords) == 0:
        raise ValueError("Coordinates list is empty")
    
    if len(atoms_symbols) == 0:
        raise ValueError("Atoms list is empty")
    
    if len(coords) != len(atoms_symbols):
        raise ValueError(f"Mismatch between number of coordinates ({len(coords)}) and atoms ({len(atoms_symbols)})")
    
    # Convert units to bohr (internal representation)
    if unit == 'angstrom':
        conversion = 1.0 / 0.529177  # angstrom to bohr
    elif unit == 'bohr':
        conversion = 1.0  # bohr to bohr (no conversion)
    else:
        raise ValueError(f"Unknown unit: {unit}")
    
    # Create atom objects
    atoms = []
    for i, (symbol, coord) in enumerate(zip(atoms_symbols, coords)):
        if len(coord) != 3:
            raise ValueError(f"Coordinate at position {i} must have 3 values, got {len(coord)}: {coord}")
        coord_bohr = [c * conversion for c in coord]
        atoms.append(system.Atom(symbol, coord_bohr))
    
    # Calculate total number of electrons from atoms
    ne = sum([atom.atomic_number for atom in atoms])
    
    # Adjust for charge
    charge = molecule_dict.get('charge', 0)
    ne -= charge
    
    # Handle spin
    spin = molecule_dict.get('spin', 0)
    na = (ne + spin) // 2
    nb = (ne - spin) // 2
    
    # Set molecule configuration
    cfg.system.units = 'bohr'
    cfg.system.electrons = (na, nb)
    cfg.system.molecule = atoms


def create_config_from_yaml(yaml_path: str) -> ml_collections.ConfigDict:
    """Create config object from YAML file."""
    # Load YAML data as ConfigDict
    yaml_config = load_yaml_config(yaml_path)
    
    # Create base config
    cfg = base_config.default()
    
    # Apply YAML configuration to config object
    apply_configdict_to_config(cfg, yaml_config)
    
    # Resolve any references in the config
    cfg = base_config.resolve(cfg)
    
    return cfg


def save_config_to_yaml(cfg: ml_collections.ConfigDict, yaml_path: str) -> None:
    """Save config object to YAML file."""
    # Convert ConfigDict to dictionary
    config_dict = configdict_to_dict(cfg)
    
    # Save as YAML
    with open(yaml_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VMC calculation with YAML config")
    parser.add_argument("config_file", type=str, 
                       help="Path to YAML configuration file")
    parser.add_argument("--save-default", action="store_true",
                       help="Save default configuration to a YAML file and exit")
    parser.add_argument("--output", type=str, default="default_config.yaml",
                       help="Output file for --save-default option")
    
    args = parser.parse_args()
    
    # If --save-default is specified, save default configuration and exit
    if args.save_default:
        cfg = base_config.default()
        save_config_to_yaml(cfg, args.output)
        print(f"Default configuration saved to {args.output}")
        sys.exit(0)
    
    # Create config from YAML file
    cfg = create_config_from_yaml(args.config_file)
    
    # Run main function
    main(cfg)