# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the Apache 2.0 License.
# (See accompanying file README.md file or copy at https://opensource.org/license/apache-2-0/)


def pytest_addoption(parser):
    parser.addoption(
        "--custom",
        action="store",
        help="Custom SMILES (.smi / .smi.gz) or Scaffolds (.scaff) file to test",
    )
