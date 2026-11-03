"""Prepare pinned GSM8K references and 1,000 structured arithmetic traces on CPU."""
import argparse
import sys
import yaml
from state_inference.data.preparation import prepare_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--download", action="store_true", help="Allow pinned source download if not already cached")
    args = parser.parse_args(argv)
    try:
        output = prepare_data(args.config, allow_download=args.download)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
