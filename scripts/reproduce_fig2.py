from bnc_repro.cli import main

if __name__ == "__main__":
    main(["plot", "--figure", "fig2", *__import__("sys").argv[1:]])

