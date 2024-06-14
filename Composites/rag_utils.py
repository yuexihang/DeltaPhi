from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import tqdm
from utils import count_params,LpLoss,UnitGaussianNormalizer

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class ReferenceDataset(Dataset):
    def __init__(self, refered_a, u, trainset):
        self.refered_a = refered_a
        self.u = u
        self.trainset = trainset

    def __len__(self):
        return len(self.refered_a)

    def __getitem__(self, idx):
        ax = self.refered_a[idx]['x']
        refer_sets = self.refered_a[idx]['ref_x']
        sel_idx = random.sample( list( range( len( refer_sets ) ) ) , 1 )[0]
        ele = refer_sets[ sel_idx ][0]
        return { 'x': ax, 'ref_score': ele[0], 'ref_x': self.trainset[0][int(ele[1])], 'ref_y':  self.trainset[1][ int(ele[1]) ] }, self.u[idx]


def retreieval_from_trainset( retrieval_term, a, u ,trainset_input_vectors, topk = 5,  erase_same_idx=True, ref_number = 1 ):
        num = a.shape[0]
        ntrain = trainset_input_vectors.shape[0]
        
        pair_a = []
        co_u   = []
        for i in tqdm.tqdm( range( num ) ):
            vector_i = retrieval_term[i].reshape( 1, -1 ) # x, y, tout, tin
            # vector_i = a[i].reshape( 1, -1 ) # x, y, tout, tin
            # vector_i = u[i][ :, : ].reshape( 1, -1 ) # x, y, tout, tin

            cosine_sim = cosine_similarity( vector_i , trainset_input_vectors )[0]
            if erase_same_idx:
                cosine_sim[ i ] = -1 # self score = -1
            scores_idxes = sorted( list( zip( list( cosine_sim), list( range( ntrain ) ) ) ) , reverse=True )

            refer_sets = []
            avg_score = 0
            for idx in range(topk):
                avg_score += np.array(  [ scores_idxes[idx+rid][0] for rid in range( ref_number ) ] ).mean()
                refer_sets.append( [ scores_idxes[idx+rid] for rid in range( ref_number ) ] )
            avg_score /= topk
            pair_a.append( { "x": a[i], "ref_x": np.array( refer_sets ), "avg_score": avg_score } )  # ref_x: topk * ref_number * 2
            co_u.append( u[i] )

        mean_score = np.array( [ ele['avg_score'] for ele in pair_a ] ).mean()

        return pair_a, co_u, mean_score

def get_rag_dataloader(xtrain, ytrain, xtest, ytest, batch_size, rag_configs, train_shuffle = True):
    ntrain = xtrain.shape[0]
    x_s, y_s = xtrain.shape[1], ytrain.shape[1]

    training_refer_range = rag_configs['training_refer_range']
    refer_num            = rag_configs['refer_num']

    trainset_input_vectors  = xtrain.reshape(ntrain, -1)
    trainset_output_vectors = ytrain.reshape(ntrain, -1)

    trainset = ( xtrain, ytrain )

    pair_train_a, co_train_u, train_meanscore = retreieval_from_trainset( xtrain, xtrain, ytrain, trainset_input_vectors,  training_refer_range , erase_same_idx=True, ref_number = refer_num )
    train_loader = torch.utils.data.DataLoader(ReferenceDataset(pair_train_a, co_train_u, trainset), batch_size=batch_size, shuffle=train_shuffle)

    pair_test_a, co_test_u, test_meanscore = retreieval_from_trainset(xtest,  xtest, ytest, trainset_input_vectors,  1 ,  erase_same_idx=False , ref_number= refer_num )
    test_loader = torch.utils.data.DataLoader(ReferenceDataset(pair_test_a, co_test_u, trainset), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


