# NSMace — Target-Aware Force Field Selection

NSMace now supports a `--target` flag through the `nsmace_target.py` wrapper.
The flag selects the appropriate MACE model for the simulation domain.

## Supported targets

| Target        | Recommended model | Use case                                   |
|---------------|-------------------|--------------------------------------------|
| crystal       | mace-mp-0         | Inorganic crystals / materials             |
| fusion-plasma | mace-mp-0         | Plasma-facing materials / wall components  |
| atmospheric   | mace-mp-0         | Aerosol / dust / mineral particles         |
| organic       | mace-off23        | Small organic molecules                    |
| biomolecular  | mace-off23        | Proteins, lipids, RNA, DNA                 |
| protein       | mace-off23        | Protein structures                         |
| abiogenesis   | mace-off23        | Lipid + RNA + water prebiotic mixtures     |
| aging         | mace-off23        | Senolytic small molecules + telomerase     |
| clean-water   | mace-off23        | Membrane, water, salt, desalination        |
| nitrogen-fixation | mace-mp-0     | Transition-metal nitride/sulfide catalysts |
| carbon-capture | mace-mp-0        | Porous MOF/CO2 capture materials           |
| soil-microbiome | mace-off23       | Protein/lipid/RNA soil microbiome          |
| atmospheric-turbulence | mace-mp-0 | k-epsilon RANS turbulence fields           |
| atmospheric-radiation  | mace-mp-0 | CO2 longwave radiation transfer            |
| coupled-thermal-structural | mace-mp-0 | Multi-physics thermal + stress waves    |
| coupled-em-thermal   | mace-mp-0     | Multi-physics EM + heat conduction         |

## Usage

```bash
python3 ../nsmace_target.py --target biomolecular --unit-cell input.cif
```

Bella may set the `MACE_MODEL` environment variable directly; the wrapper
preserves all other original `build/NSMace` command-line arguments.
