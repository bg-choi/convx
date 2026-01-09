import os
import json

def get_stop_ids(args):
    with open(os.path.join(args.data_dir, 'stopids_v2.json'), 'r') as fin:
        stop_ids = json.load(fin)
    print(f"# Stop IDs: {len(stop_ids)}")
    
    return stop_ids


class AverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self, name: str, round_digits: int = 3):
        self.name = name
        self.round_digits = round_digits
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return '{}: {}'.format(self.name, round(self.avg, self.round_digits))