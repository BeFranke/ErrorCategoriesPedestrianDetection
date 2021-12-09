def test(base_dir, sequence, company):
    print("This test is not implemented yet!")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--company", required=True, type=str)
    parser.add_argument("--sequence", required=True, type=int)
    args = parser.parse_args()
    test(args.data_path, args.sequence, args.company)


if __name__ == "__main__":
    main()
