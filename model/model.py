import pickle 

class Model:
    def load_model(path):
        if path.endswith('.pkl'):
            model = pickle.load(open(path, 'rb'))
        else: 
            raise Exception('Could not load model')
        return model