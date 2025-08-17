import pickle

if __name__ == "__main__":
    # Load the pickle file
    with open('ファイル名.pkl', 'rb') as f:
        data = pickle.load(f)

    print(type(data))
    print(data)
