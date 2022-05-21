from torch import nn


def main():
    m = nn.Sequential(
        nn.Conv2d(512, 81, (3, 3), (1, 1), 1),
        nn.Linear(81, 1)
    )


main()
