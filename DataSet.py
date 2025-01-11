
class DataSet(object):
    def __init__(self,dataset_name):
        if dataset_name=='people':
            self.init_people()
        elif dataset_name=='people2':
            self.init_people2()
        else:
            print('Unknown dataset',dataset_name)
        # elif dataset_name=='weibo':
        #     self.init_weibo()
        # elif dataset_name=='emergency':
        #     self.init_emergency()

    def init_people(self):
        self.train_file = 'data/people/train.csv'
        self.dev_file = 'data/people/dev.csv'
        self.label2id_file = 'data/people/label2id.txt'
        self.suffix = ["ORG", "PER", "LOC"]
    def init_people2(self):
        self.train_file = 'data/people2/train.csv'
        self.dev_file = 'data/people2/dev.csv'
        self.label2id_file = 'data/people2/label2id.txt'
        self.suffix = ["ORG", "PER", "LOC"]

    def init_weibo(self):
        self.train_file = 'data/weibo/train.txt'
        self.dev_file = 'data/weibo/dev.txt'
        self.label2id_file = 'data/weibo/label2id.txt'
        self.suffix = ["PER", "LOC", "ORG"]

    def init_emergency(self):
        self.train_file = 'data/emergency/train.csv'
