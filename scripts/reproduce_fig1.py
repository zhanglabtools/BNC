from bnc_repro.cli import main

if __name__ == "__main__":
    main(["plot", "--figure", "fig1", *__import__("sys").argv[1:]])

