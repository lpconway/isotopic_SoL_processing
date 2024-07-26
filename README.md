# isotopic_SoL_processing
Python implementation of Dizco Processing Pipeline from Wozniak et al., Nat. Chem. Biol., 2024,  20, 823–834 as a Proteome Discoverer Node

# Usage

In the Proteome Discoverer consensus workflow, create a scripting node. The "Path to Executable" field should point to a Python interpreter executable (tested on 3.9) with the numpy, pandas, and statsmodels libraries, while the "Command Line Arguments" field should be the path to the Isotopic Sol Processor python file with %NODEARGS% as the argument.

The workflow should have MultiPSM search enabled, and the heavy and light modification names should include the substrings "heavy" and "light" as appropriate.

The output of the script is a tsv file with the suffix "_Processed_SOL__{timestamp}.tsv".
