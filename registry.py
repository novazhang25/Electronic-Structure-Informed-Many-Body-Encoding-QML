"""Production molecules, AO order and package-local output locations."""
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
DEFAULT_INITIAL_DIR = PACKAGE / 'initialdata'
DEFAULT_DESCRIPTOR_DIR = PACKAGE / 'descriptors_native_mp2'
MOLECULES = {
    'HF': (('H', 'F'), (0, 2, 3, 4, 5)),
    'H2O': (('O', 'H', 'H'), (1, 2, 3, 4, 5, 6)),
    'NH3': (('N', 'H', 'H', 'H'), (1, 2, 3, 4, 5, 6, 7)),
    'N2': (('N', 'N'), (1, 2, 3, 4, 6, 7, 8, 9)),
    'CO': (('C', 'O'), (1, 2, 3, 4, 6, 7, 8, 9)),
}
MOLECULE_NAMES = tuple(MOLECULES)


def nh3_selected_aos(symbols, ao_labels):
    """Resolve N(2s,2px,2py,2pz), then all three H(1s), from full AO labels."""
    symbols = list(symbols)
    if sorted(symbols) != ['H', 'H', 'H', 'N']:
        raise ValueError('Expected one N atom and three H atoms')
    labels = [str(label).split() for label in ao_labels]
    nitrogen = symbols.index('N')
    hydrogens = [i for i, symbol in enumerate(symbols) if symbol == 'H']
    expected = [(nitrogen, 'N', orbital) for orbital in ('1s', '2s', '2px', '2py', '2pz')]
    expected += [(atom, 'H', '1s') for atom in hydrogens]
    parsed = [(int(label[0]), label[1], label[2]) for label in labels]
    if len(parsed) != 8 or set(parsed) != set(expected):
        raise ValueError('Unexpected NH3 STO-3G AO labels or duplicated orbitals')
    return [parsed.index(label) for label in expected[1:]]
