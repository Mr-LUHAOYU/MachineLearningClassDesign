from models import test_model

if __name__ == '__main__':
    test_model("MLP")
    print('\n---------------------------------------\n')
    test_model("XGBoost")
