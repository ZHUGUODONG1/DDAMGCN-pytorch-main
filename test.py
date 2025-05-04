import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
import os
import shutil
import pickle
import random
from train import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/ShenZhen_City',help='data path')
parser.add_argument('--adjdata',type=str,default='data/ShenZhen_City/adj_mat.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=627,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.5,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=200,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--force', type=str, default=False,help="remove params dir", required=False)
parser.add_argument('--save',type=str,default='./garage/ShenZhen_City',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--model',type=str,default='DDAMGCN',help='model type')
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate ')
parser.add_argument('--l', type=int,default=3,help='block layers')
parser.add_argument('--k_num', type=int,default=50,help='number of key nodes')
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # multi-GPU


seed = 1
setup_seed(seed)


def main():
    device = torch.device(args.device)
    adj_mx = util.load_adj(args.adjdata, args.adjtype)
    with open('adj_mx1.pkl', 'wb') as file:
        pickle.dump(adj_mx, file)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    # scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]


    if args.model=='DDAMGCN':
        engine = trainer( args.in_dim, args.seq_length, args.num_nodes, args.k_num, args.nhid, args.l,args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.decay
                         )

    params_path = args.save + "/" + args.model

    engine.model.load_state_dict(torch.load("DDAMGCN_exp1_best_174.pth"))
    

    # testing
    outputs = []
    # dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    realy = torch.Tensor(dataloader['y_test']).to(device)

    realy = realy.transpose(1, 3)[:, 0, :, :]
    engine.model.eval()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    prediction = yhat
    for i in range(12):
        pred = prediction[:, :, i]
        # pred = scaler.inverse_transform(yhat[:,:,i])
        # prediction.append(pred)
        real = realy[:, :, i]
        # print("Pred shape:", pred.shape)
        # print("Labels shape:", real.shape)
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    # torch.save(engine.model.state_dict(),
    #            params_path + "/" + args.model + "_exp" + str(args.expid) + "_best_" + str(bestid + 1) + ".pth")
    prediction_path = params_path + "/" + args.model + "_prediction_results"
    ground_truth = realy.cpu().detach().numpy()
    prediction = prediction.cpu().detach().numpy()

    np.savez_compressed(
        os.path.normpath(prediction_path),
        prediction=prediction,
        ground_truth=ground_truth
    )

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
