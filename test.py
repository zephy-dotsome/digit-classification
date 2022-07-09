from parse_config import ConfigParser
import model.model as module_arch
import model.metric as module_metric
import model.loss as module_loss
import data_loader.data_loaders as module_data
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import argparse


def make_submission(preds):
    cols = ['ImageId', 'Label']
    sub = pd.DataFrame(columns=cols)
    sub['ImageId'] = np.arange(1, preds.size(0)+1)
    sub['Label'] = preds.numpy()
    sub.to_csv('submission.csv', index=False)

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))


    pred = None

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data = data.to(device)
            output = model(data).cpu()
            #
            # save sample images, or do something with output here
            #
            _, preds = torch.max(output, dim=1)
            preds = preds.view(-1, 1)
            if pred is not None:
                pred = torch.cat((pred, preds), 0)
            else:
                pred = preds
            
        make_submission(pred)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
