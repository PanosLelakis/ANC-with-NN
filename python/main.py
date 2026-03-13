import sys
from pathlib import Path

EXPECTED_FOLDER_NAME = "ANC-with-NN"

def ensure_correct_cwd(expected_name: str = EXPECTED_FOLDER_NAME) -> Path:
    """
    Require that the current working directory is the project root folder.
    If not, print an error and exit with non-zero code.
    """
    cwd = Path.cwd().resolve()

    if cwd.name != expected_name:
        print("\nERROR: Wrong working directory.\n")
        print(f"Expected current working directory folder name: {expected_name}")
        print(f"Actual current working directory: {cwd}\n")
        print("Fix:")
        print(f"1) In terminal: cd <path-to>/{expected_name}")
        print(f"2) Then run: python main.py\n")
        sys.exit(1)

    return cwd

def main():
    ensure_correct_cwd()
    from gui.gui_common import build_and_run
    build_and_run()

if __name__ == "__main__":
    main()