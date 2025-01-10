
class DataSet(object):
    def __init__(self,dataset_name):
        if dataset_name=='people':
            self.init_people()
        elif dataset_name=='movies':
            pass

    def init_people(self):
        self.train_file = 'data/people/train.csv'
        self.dev_file = 'data/people/dev.csv'
        self.label2id_file = 'data/people/label2id.txt'
        self.suffix = ["ORG", "PER", "LOC"]

    def init_weibo(self):
        self.train_file = 'data/weibo/train.txt'
        self.dev_file = 'data/weibo/dev.txt'
        self.label2id_file = 'data/weibo/label2id.txt'
        self.suffix = ["PER", "LOC", "ORG"]

    def init_emergency(self):
        self.train_file = 'data/emergency/train.csv'
