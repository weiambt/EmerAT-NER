
class DataSet(object):
    def __init__(self,dataset_name):
        self.name = dataset_name
        if dataset_name=='people':
            self.init_people()
        elif dataset_name=='people2':
            self.init_people2()
        elif dataset_name=='emergency_2024_4_14':
            self.init_emergency_2024_4_14()
        elif dataset_name=='emergency_2025_2_1':
            self.init_emergency_2025_2_1()
        elif dataset_name=='emergency_2025_2_4':
            self.init_emergency_2025_2_4()
        elif dataset_name=='emergency_2025_2_8':
            self.init_emergency_2025_2_8()
        elif dataset_name=='weibo':
            self.init_weibo()
        elif dataset_name=='cluener':
            self.init_cluener()
        else:
            print('Unknown dataset',dataset_name)


    def __str__(self):
        return f"Dataset: {self.name}"


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

    def init_emergency_2024_4_14(self):
        self.train_file = 'data/emergency_2024_4_14/train.csv'
        self.dev_file = 'data/emergency_2024_4_14/dev.csv'
        self.label2id_file = 'data/emergency_2024_4_14/label2id.txt'
        # self.suffix = ["TYPE","CLOSS","PLOSS"]
        self.suffix = ["TIME", "DLOC", "CLOSS", "OORG", "DATE", "OPER", "TYPE", "PORG", "LPER", "PLOSS", "PPER","LORG"]
    def init_emergency_2025_2_1(self):
        self.train_file = 'data/emergency_2025_2_1train.csv'
        self.dev_file = 'data/emergency_2025_2_1/dev.csv'
        self.label2id_file = 'data/emergency_2025_2_1/label2id.txt'
        # self.suffix = ["TYPE","CLOSS","PLOSS"]
        self.suffix = ["TIME", "DLOC", "CLOSS", "OORG", "DATE", "OPER", "TYPE", "PORG", "LPER", "PLOSS", "PPER","LORG"]

    def init_emergency_2025_2_4(self):
        self.train_file = 'data/emergency_2025_2_4/train.csv'
        self.dev_file = 'data/emergency_2025_2_4/dev.csv'
        self.label2id_file = 'data/emergency_2025_2_4/label2id.txt'
        # self.suffix = ["TYPE","CLOSS","PLOSS"]
        self.suffix = ["TIME", "DLOC", "CLOSS", "OORG", "DATE", "OPER", "TYPE", "PORG", "LPER", "PLOSS", "PPER","LORG"]

    def init_emergency_2025_2_8(self):
        self.train_file = 'data/emergency_2025_2_8/train.csv'
        self.dev_file = 'data/emergency_2025_2_8/dev.csv'
        self.label2id_file = 'data/emergency_2025_2_8/label2id.txt'
        # self.suffix = ["TYPE","CLOSS","PLOSS"]
        self.suffix = ["TIME", "CLOSS", "OORG", "DATE", "OPER", "TYPE", "PORG", "LPER", "PLOSS", "PPER","LORG"]

    def init_weibo(self):
        self.train_file = 'data/weibo/train.csv'
        self.dev_file = 'data/weibo/dev.csv'
        self.label2id_file = 'data/weibo/label2id.txt'
        self.suffix = ["PER.NAM", "LOC.NAM", "ORG.NAM","GPE.NAM"]

    def init_cluener(self):
        # https: // github.com / CLUEbenchmark / CLUENER2020
        self.train_file = 'data/cluener/train.csv'
        self.dev_file = 'data/cluener/dev.csv'
        self.label2id_file = 'data/cluener/label2id.txt'
        self.suffix = ["address", "book", "company","game","government","movie","name","organization","position","scene"]


if __name__ == '__main__':
    ds = DataSet('emergency_2024_4_14')
    print(ds)