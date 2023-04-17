from importlib_resources import files

data_text = files('selfrel.resources').joinpath('description.txt').read_text()


def main() -> None:
    print(data_text)
    print("Hello World")


if __name__ == "__main__":
    main()
