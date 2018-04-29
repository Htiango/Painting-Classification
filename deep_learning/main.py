import argparse
import numpy as np
import model

def run(args):
    X = np.loadtxt(args.X_path)

    y = np.loadtxt(args.Y_path, dtype=int)

    X = X[(y==5) | (y==6)]
    y = y[(y==5) | (y==6)]

    y[(y==5)] = 0
    y[(y==6)] = 1

    print("Loaded data!")
    print("Data_size = " + str(y.shape[0]))
    print("label 0: " + str(y[(y==0)].shape[0]))
    print("label 1: " + str(y[(y==1)].shape[0]))

    if args.mode == "train":
        iteration_num = 3000
        print("training...")
        model.train(X,y, iteration_num)
    else:
        print("genrating...")
        model.test(X, y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-X', '--X_path', type=str, 
        required=True, help='input path of the feature X.')
    parser.add_argument('-Y', '--Y_path', type=str, 
        required=True, help='input path of the feature Y.')
    parser.add_argument("-m", "--mode", help = "select mode by 'train' or test",
        choices = ["train", "test"], default = "test")

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()