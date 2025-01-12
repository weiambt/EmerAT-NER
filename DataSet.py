
class DataSet(object):
    def __init__(self,dataset_name):
        if dataset_name=='people':
            self.init_people()
        elif dataset_name=='people2':
            self.init_people2()
        elif dataset_name=='emergency_2024_4_14':
            self.init_emergency_2024_4_14()
        else:
            print('Unknown dataset',dataset_name)
        # elif dataset_name=='weibo':
        #     self.init_weibo()


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

    def init_emergency_2024_4_14(self):
        self.train_file = 'data/emergency_2024_4_14/train.csv'
        self.dev_file = 'data/emergency_2024_4_14/dev.csv'
        self.label2id_file = 'data/emergency_2024_4_14/label2id.txt'
        # self.suffix = ["TYPE","CLOSS","PLOSS"]
        self.suffix = ["TIME", "LOC", "CLOSS", "OORG", "DATE", "OPER", "TYPE", "PORG", "LPER", "PLOSS", "PPER"]
