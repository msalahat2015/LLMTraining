def some_function():
    print("Hello")
   
def main(args):
    some_function()
    print("Hello")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])