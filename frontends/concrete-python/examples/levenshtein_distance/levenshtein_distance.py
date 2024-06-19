# Computing Levenstein distance between strings, https://en.wikipedia.org/wiki/Levenshtein_distance

import time
import argparse
import random
from functools import lru_cache

import numpy

from concrete import fhe


class Alphabet:

    letters: str = None
    mapping_to_int: dict = {}

    @staticmethod
    def lowercase():
        """Set lower case alphabet."""
        return Alphabet("abcdefghijklmnopqrstuvwxyz")

    @staticmethod
    def uppercase():
        """Set upper case alphabet."""
        return Alphabet("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    @staticmethod
    def anycase():
        """Set any-case alphabet."""
        return Alphabet.lowercase() + Alphabet.uppercase()

    @staticmethod
    def dna():
        """Set DNA alphabet."""
        return Alphabet("ATGC")

    def __init__(self, letters: str):
        self.letters = letters

        for i, c in enumerate(self.letters):
            self.mapping_to_int[c] = i

    def __add__(self, other: "Alphabet") -> "Alphabet":
        return Alphabet(self.letters + other.letters)

    def return_available_alphabets() -> list:
        """Return available alphabets."""
        return ["string", "STRING", "StRiNg", "ACTG"]

    @staticmethod
    def init_by_name(alphabet_name: str) -> "Alphabet":
        """Set the alphabet."""
        assert (
            alphabet_name in Alphabet.return_available_alphabets()
        ), f"Unknown alphabet {alphabet_name}"

        if alphabet_name == "string":
            return Alphabet.lowercase()
        if alphabet_name == "STRING":
            return Alphabet.uppercase()
        if alphabet_name == "StRiNg":
            return Alphabet.anycase()
        if alphabet_name == "ACTG":
            return Alphabet.dna()

    def random_pick_in_values(self) -> int:
        """Pick the integer-encoding of a random char in an alphabet."""
        return numpy.random.randint(len(self.mapping_to_int))

    def _random_string(self, length: int) -> str:
        """Pick a random string in the alphabet."""
        return "".join([random.choice(list(self.mapping_to_int)) for _ in range(length)])

    def prepare_random_patterns(self, len_min: int, len_max: int, nb_strings: int) -> list:
        """Prepare random patterns of different lengths."""
        assert len(self.mapping_to_int) > 0, "Mapping not defined"

        list_patterns = []
        for _ in range(nb_strings):
            for length_1 in range(len_min, len_max + 1):
                for length_2 in range(len_min, len_max + 1):
                    list_patterns += [
                        (
                            self._random_string(length_1),
                            self._random_string(length_2),
                        )
                        for _ in range(1)
                    ]

        return list_patterns

    def encode(self, string: str) -> tuple:
        """Encode a string, ie map it to integers using the alphabet."""

        assert len(self.mapping_to_int) > 0, "Mapping not defined"

        for si in string:
            if si not in self.mapping_to_int:
                raise ValueError(
                    f"Char {si} of {string} is not in alphabet {list(self.mapping_to_int.keys())}, please choose the right --alphabet"
                )

        return tuple([self.mapping_to_int[si] for si in string])


class LevenshteinDistance:
    alphabet: Alphabet
    module: fhe.module

    def __init__(self, alphabet: Alphabet, args):
        self.alphabet = alphabet

        self._compile_module(args)

    def calculate(self, a: str, b: str, mode: str, show_distance: bool = False):
        """Compute a distance between two strings, either in fhe or in simulate."""
        if mode == "simulate":
            self._compute_in_simulation([(a, b)])
        else:
            assert mode == "fhe", "Only 'simulate' and 'fhe' mode are available"
            self._compute_in_fhe([(a, b)], show_distance=show_distance)

    def calculate_list(self, l: list, mode: str):
        """Compute a distance between strings of a list, either in fhe or in simulate."""
        for (a, b) in l:
            self.calculate(a, b, mode)

    def _encode_and_encrypt_strings(self, a: str, b: str) -> tuple:
        """Encode a string, ie map it to integers using the alphabet, and then encrypt the integers."""
        a_as_int = self.alphabet.encode(a)
        b_as_int = self.alphabet.encode(b)

        a_enc = tuple(self.module.equal.encrypt(ai, None)[0] for ai in a_as_int)
        b_enc = tuple(self.module.equal.encrypt(None, bi)[1] for bi in b_as_int)

        return a_enc, b_enc

    def _compile_module(self, args):
        """Compile the FHE module."""
        assert len(self.alphabet.mapping_to_int) > 0, "Mapping not defined"

        inputset_equal = [
            (
                self.alphabet.random_pick_in_values(),
                self.alphabet.random_pick_in_values(),
            )
            for _ in range(1000)
        ]
        inputset_mix = [
            (
                numpy.random.randint(2),
                numpy.random.randint(args.max_string_length),
                numpy.random.randint(args.max_string_length),
                numpy.random.randint(args.max_string_length),
                numpy.random.randint(args.max_string_length),
            )
            for _ in range(1000)
        ]

        self.module = LevenshsteinModule.compile(
            {
                "equal": inputset_equal,
                "mix": inputset_mix,
                "constant": [i for i in range(len(self.alphabet.mapping_to_int))],
            },
            show_mlir=args.show_mlir,
            p_error=10**-20,
            show_optimizer=args.show_optimizer,
            comparison_strategy_preference=fhe.ComparisonStrategy.ONE_TLU_PROMOTED,
            min_max_strategy_preference=fhe.MinMaxStrategy.ONE_TLU_PROMOTED,
        )

    def _compute_in_simulation(self, list_patterns: list):
        """Check equality between distance in simulation and clear distance."""
        for a, b in list_patterns:

            print(f"    Computing Levenshtein between strings '{a}' and '{b}'", end="")

            a_as_int = self.alphabet.encode(a)
            b_as_int = self.alphabet.encode(b)

            l1_simulate = levenshtein_simulate(self.module, a_as_int, b_as_int)
            l1_clear = levenshtein_clear(a_as_int, b_as_int)

            assert l1_simulate == l1_clear, f"    {l1_simulate=} and {l1_clear=} are different"
            print(" - OK")

    def _compute_in_fhe(self, list_patterns: list, show_distance: bool = False):
        """Check equality between distance in FHE and clear distance."""
        self.module.keygen()

        # Checks in FHE
        for a, b in list_patterns:

            print(f"    Computing Levenshtein between strings '{a}' and '{b}'", end="")

            a_enc, b_enc = self._encode_and_encrypt_strings(a, b)

            time_begin = time.time()
            l1_fhe_enc = levenshtein_fhe(self.module, a_enc, b_enc)
            time_end = time.time()

            l1_fhe = self.module.mix.decrypt(l1_fhe_enc)

            l1_clear = levenshtein_clear(a, b)

            assert l1_fhe == l1_clear, f"    {l1_fhe=} and {l1_clear=} are different"

            if not show_distance:
                print(f" - OK in {time_end - time_begin:.2f} seconds")
            else:
                print(f" - distance is {l1_fhe}, computed in {time_end - time_begin:.2f} seconds")


# Module FHE
@fhe.module()
class LevenshsteinModule:
    @fhe.function({"x": "encrypted", "y": "encrypted"})
    def equal(x, y):
        """Assert equality between two chars of the alphabet."""
        return x == y

    @fhe.function({"x": "clear"})
    def constant(x):
        return fhe.zero() + x

    @fhe.function(
        {
            "is_equal": "encrypted",
            "if_equal": "encrypted",
            "case_1": "encrypted",
            "case_2": "encrypted",
            "case_3": "encrypted",
        }
    )
    def mix(is_equal, if_equal, case_1, case_2, case_3):
        """Compute the min of (case_1, case_2, case_3), and then return `if_equal` if `is_equal` is
        True, or the min in the other case."""
        min_12 = numpy.minimum(case_1, case_2)
        min_123 = numpy.minimum(min_12, case_3)

        return fhe.if_then_else(is_equal, if_equal, 1 + min_123)

    # There is a single output in mix: it can go to
    #   - input 1 of mix
    #   - input 2 of mix
    #   - input 3 of mix
    #   - input 4 of mix
    # or just be the final output
    #
    # There is a single output of equal, it goes to input 0 of mix
    composition = fhe.Wired(
        [
            fhe.Wire(fhe.AllOutputs(equal), fhe.Input(mix, 0)),
            fhe.Wire(fhe.AllOutputs(mix), fhe.Input(mix, 1)),
            fhe.Wire(fhe.AllOutputs(mix), fhe.Input(mix, 2)),
            fhe.Wire(fhe.AllOutputs(mix), fhe.Input(mix, 3)),
            fhe.Wire(fhe.AllOutputs(mix), fhe.Input(mix, 4)),
            fhe.Wire(fhe.AllOutputs(constant), fhe.Input(mix, 1)),
            fhe.Wire(fhe.AllOutputs(constant), fhe.Input(mix, 2)),
            fhe.Wire(fhe.AllOutputs(constant), fhe.Input(mix, 3)),
            fhe.Wire(fhe.AllOutputs(constant), fhe.Input(mix, 4)),
        ]
    )


@lru_cache
def levenshtein_clear(x: str, y: str):
    """Compute the distance in clear, for reference and comparison."""
    if len(x) == 0:
        return len(y)
    if len(y) == 0:
        return len(x)

    if x[0] == y[0]:
        return levenshtein_clear(x[1:], y[1:])

    case_1 = levenshtein_clear(x[1:], y)
    case_2 = levenshtein_clear(x, y[1:])
    case_3 = levenshtein_clear(x[1:], y[1:])

    return 1 + min(case_1, case_2, case_3)


@lru_cache
def levenshtein_simulate(module: fhe.module, x: str, y: str):
    """Compute the distance in simulation."""
    if len(x) == 0:
        return len(y)
    if len(y) == 0:
        return len(x)

    if_equal = levenshtein_simulate(module, x[1:], y[1:])
    case_1 = levenshtein_simulate(module, x[1:], y)
    case_2 = levenshtein_simulate(module, x, y[1:])
    case_3 = if_equal

    is_equal = module.equal(x[0], y[0])
    returned_value = module.mix(is_equal, if_equal, case_1, case_2, case_3)

    return returned_value


@lru_cache
def levenshtein_fhe(module: fhe.module, x: str, y: str):
    """Compute the distance in FHE."""
    if len(x) == 0:
        return module.constant.run(module.constant.encrypt(len(y)))
    if len(y) == 0:
        return module.constant.run(module.constant.encrypt(len(x)))

    if_equal = levenshtein_fhe(module, x[1:], y[1:])
    case_1 = levenshtein_fhe(module, x[1:], y)
    case_2 = levenshtein_fhe(module, x, y[1:])
    case_3 = if_equal

    is_equal = module.equal.run(x[0], y[0])
    returned_value = module.mix.run(is_equal, if_equal, case_1, case_2, case_3)

    return returned_value


def manage_args():
    """Manage user arguments."""
    parser = argparse.ArgumentParser(description="Levenshtein distance in Concrete.")
    parser.add_argument(
        "--show_mlir",
        dest="show_mlir",
        action="store_true",
        help="Show the MLIR",
    )
    parser.add_argument(
        "--show_optimizer",
        dest="show_optimizer",
        action="store_true",
        help="Show the optimizer outputs",
    )
    parser.add_argument(
        "--autotest",
        dest="autotest",
        action="store_true",
        help="Run random tests",
    )
    parser.add_argument(
        "--autoperf",
        dest="autoperf",
        action="store_true",
        help="Run benchmarks",
    )
    parser.add_argument(
        "--distance",
        dest="distance",
        nargs=2,
        type=str,
        action="store",
        help="Compute a distance",
    )
    parser.add_argument(
        "--alphabet",
        dest="alphabet",
        choices=Alphabet.return_available_alphabets(),
        default="string",
        help="Setting the alphabet",
    )
    parser.add_argument(
        "--max_string_length",
        dest="max_string_length",
        type=int,
        default=4,
        help="Setting the maximal size of strings",
    )
    args = parser.parse_args()

    # At least one option
    assert (
        args.autoperf + args.autotest + (args.distance != None) > 0
    ), "must activate one option --autoperf or --autotest or --distance"

    return args


def main():
    """Main function."""
    print()

    # Options by the user
    args = manage_args()

    # Do what the user requested
    if args.autotest:

        alphabet = Alphabet.init_by_name(args.alphabet)
        levenshtein_distance = LevenshteinDistance(alphabet, args)

        print(f"Making random tests with alphabet {args.alphabet}")
        print(f"Letters are {alphabet.letters}\n")

        list_patterns = alphabet.prepare_random_patterns(0, args.max_string_length, 1)
        print("Computations in simulation\n")
        levenshtein_distance.calculate_list(list_patterns, mode="simulate")
        print("\nComputations in FHE\n")
        levenshtein_distance.calculate_list(list_patterns, mode="fhe")
        print("")

    if args.autoperf:
        for alphabet_name in ["ACTG", "string", "STRING", "StRiNg"]:
            print(
                f"Typical performances for alphabet {alphabet_name}, with string of maximal length:\n"
            )

            alphabet = Alphabet.init_by_name(alphabet_name)
            levenshtein_distance = LevenshteinDistance(alphabet, args)
            list_patterns = alphabet.prepare_random_patterns(
                args.max_string_length, args.max_string_length, 3
            )
            levenshtein_distance.calculate_list(list_patterns, mode="fhe")
            print("")

    if args.distance != None:
        print(
            f"Running distance between strings '{args.distance[0]}' and '{args.distance[1]}' for alphabet {args.alphabet}:\n"
        )

        if max(len(args.distance[0]), len(args.distance[1])) > args.max_string_length:
            args.max_string_length = max(len(args.distance[0]), len(args.distance[1]))
            print(
                "Warning, --max_string_length was smaller than lengths of the input strings, fixing it"
            )

        alphabet = Alphabet.init_by_name(args.alphabet)
        levenshtein_distance = LevenshteinDistance(alphabet, args)
        levenshtein_distance.calculate(
            args.distance[0], args.distance[1], mode="fhe", show_distance=True
        )
        print("")

    print("Successful end\n")


if __name__ == "__main__":
    main()
