import sys
from new_neural import load_dataset, build_model, train_model, test_model
from input import main

def main():
    if len(sys.argv) < 2:
        print('Error! Filename not specified.')

        return

    path = sys.argv[1]

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    model = build_model()
    history = train_model(model, X_train, y_train, X_val, y_val)
    test_model(X_test, y_test)



    

if __name__ == "__main__":
    main()