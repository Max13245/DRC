import argparse

ACTIONS = ["train", "config", "use"]


def main():
    parser = argparse.ArgumentParser(description="Config control")

    parser.add_argument(
        "action",
        metavar="action",
        choices=ACTIONS,
        help="Perform action: " + ", ".join(ACTIONS),
    )


if __name__ == "__main__":
    main()

"""
Train a new network (to 95%)
Configure a network (to 100% by overfitting)
Use network 
"""
