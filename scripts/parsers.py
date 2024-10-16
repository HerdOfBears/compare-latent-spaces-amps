import argparse
from transvae.transformer_models import TransVAE
from transvae.rnn_models import RNN, RNNAttn

'''
Added the aae support lines 37 & 49
need to add sample parser and attn_parser
'''
from transvae.aae_models import AAE
from transvae.wae_models import WAE




def model_init(args, params={}):
    print("parser model_init called /n")
    ### Model Name
    if args.save_name is None:
        if args.model == 'transvae':
            save_name = 'trans{}x-{}_{}'.format(args.d_feedforward // args.d_model,
                                                args.d_model,
                                                args.data_source)
        else:
            save_name = '{}-{}_{}'.format(args.model,
                                          args.d_model,
                                          args.data_source)
    else:
        save_name = args.save_name

    ### Load Model
    if args.model == 'transvae':
        vae = TransVAE(params=params, name=save_name, d_model=args.d_model,
                       d_ff=args.d_feedforward, d_latent=args.d_latent,
                       property_predictor=args.property_predictor, d_pp=args.d_property_predictor,
                       depth_pp=args.depth_property_predictor, type_pp=args.type_property_predictor)
    elif args.model == 'rnnattn':
        vae = RNNAttn(params=params, name=save_name, d_model=args.d_model,
                      d_latent=args.d_latent, property_predictor=args.property_predictor,
                      d_pp=args.d_property_predictor, depth_pp=args.depth_property_predictor, type_pp=args.type_property_predictor)
    elif args.model == 'rnn':
        vae = RNN(params=params, name=save_name, d_model=args.d_model,
                  d_latent=args.d_latent, property_predictor=args.property_predictor,
                  d_pp=args.d_property_predictor, depth_pp=args.depth_property_predictor, type_pp=args.type_property_predictor)
        
    elif args.model == 'aae':
        vae = AAE(params=params, name=save_name, d_model=args.d_model,
                  d_latent=args.d_latent, property_predictor=args.property_predictor,
                  d_pp=args.d_property_predictor, depth_pp=args.depth_property_predictor, type_pp=args.type_property_predictor)
    
    elif args.model == 'wae':
        vae = WAE(params=params, name=save_name, d_model=args.d_model,
                  d_latent=args.d_latent, property_predictor=args.property_predictor,
                  d_pp=args.d_property_predictor, depth_pp=args.depth_property_predictor, type_pp=args.type_property_predictor)
    return vae

def train_parser():
    print("train_parser function called /n")
    parser = argparse.ArgumentParser()
    ### Architecture Parameters
    parser.add_argument('--model', choices=['transvae', 'rnnattn', 'rnn', 'aae', 'wae'],
                        required=True, type=str)
    parser.add_argument('--d_model', default=128, type=int)
    parser.add_argument('--d_feedforward', default=128, type=int)
    parser.add_argument('--d_latent', default=128, type=int)
    parser.add_argument('--property_predictor', choices=['ON', 'OFF'], default='OFF', type=str)
    parser.add_argument('--d_property_predictor', default=2, type=int)
    parser.add_argument('--depth_property_predictor', default=2, type=int)
    parser.add_argument('--type_property_predictor', choices=['decision_tree', 'deep_net'], default='deep_net', type=str)
    parser.add_argument('--d_pp_out', default=1, type=int, help='Number of output dimensions for property predictor')
    parser.add_argument('--prediction_types', nargs="+", default=None, type=list, 
                        help='List of prediction types for property predictor.\n\
                            Must be list of "classification" and/or "regression"\
                            in the same order as the properties in the properties file')
    parser.add_argument('--hardware', choices=['cpu', 'gpu'], required=True, type=str)
    ### Hyperparameters
    parser.add_argument('--batch_size', default=2000, type=int)
    parser.add_argument('--batch_chunks', default=1, type=int)
    parser.add_argument('--beta', default=0.05, type=float)
    parser.add_argument('--beta_init', default=1e-8, type=float)
    parser.add_argument('--anneal_start', default=0, type=int)
    parser.add_argument('--adam_lr', default=3e-4, type=float)
    parser.add_argument('--lr_scale', default=1, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int)
    parser.add_argument('--eps_scale', default=1, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    ### Data Parameters
    parser.add_argument('--data_source', choices=['zinc', 'pubchem','peptide','peptides_renaud','custom','cdd','peptides_2024'],
                        required=True, type=str)
    parser.add_argument('--train_mols_path', default=None, type=str)
    parser.add_argument('--test_mols_path', default=None, type=str)
    parser.add_argument('--train_props_path', default=None, type=str)
    parser.add_argument('--test_props_path', default=None, type=str)
    parser.add_argument('--vocab_path', default=None, type=str)
    parser.add_argument('--char_weights_path', default=None, type=str)
    
    ### Load Parameters
    parser.add_argument('--checkpoint', default=None, type=str)
    
    ### Save Parameters
    parser.add_argument('--save_name', default=None, type=str)
    parser.add_argument('--save_freq', default=1, type=int)
    
    ### Distributed Data Parallel addition
    parser.add_argument('--init_method', default=None, type=str)
    parser.add_argument('--dist_backend', default='nccl', type=str)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--DDP', choices=['ON','OFF'], default='OFF', type=str)
    
    """AAE Arguments"""
    parser.add_argument('--discriminator_layers', nargs='+', type=int, default=[640, 256], 
                        help='Numbers of features for linear layers in discriminator')

    # whether to use structure model
    parser.add_argument("--use_structure_loss", choices=["yes","no"], default="no", type=str,
                        help="Whether to use structure model for additional loss")
    parser.add_argument("--structure_model_path", type=str, default=None, 
                        help="Path to directory containing structure model weights")

    # loss method Parameters. Isometric learning or Triplet Loss for metric learning experiments.
    parser.add_argument('--loss_method', choices=["isometry","triplet"], default=None, type=str, 
                        help="additional loss method to include. Default is None. Choices are isometry and triplet")

    return parser

def sample_parser():
    print("sample parser function called /n")
    parser = argparse.ArgumentParser()
    ### Load Files
    parser.add_argument('--model', choices=['transvae', 'rnnattn', 'rnn'],
                        required=True, type=str)
    parser.add_argument('--model_ckpt', required=True, type=str)
    parser.add_argument('--mols', default=None, type=str)
    ### Sampling Parameters
    parser.add_argument('--sample_mode', choices=['rand', 'high_entropy', 'k_high_entropy'],
                        required=True, type=str)
    parser.add_argument('--k', default=15, type=int)
    parser.add_argument('--entropy_cutoff', default=5, type=float)
    parser.add_argument('--n_samples', default=30000, type=int)
    parser.add_argument('--n_samples_per_batch', default=100, type=int)
    ### Save Parameters
    parser.add_argument('--save_path', default=None, type=str)

    return parser

def attn_parser():
    print("attn parser function called /n")
    parser = argparse.ArgumentParser()
    ### Load Files
    parser.add_argument('--model', choices=['transvae', 'rnnattn'],
                        required=True, type=str)
    parser.add_argument('--model_ckpt', required=True, type=str)
    parser.add_argument('--mols', required=True, type=str)
    ### Sampling Parameters
    parser.add_argument('--n_samples', default=50, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--batch_chunks', default=1, type=int)
    parser.add_argument('--shuffle', default=False, action='store_true')
    ### Save Parameters
    parser.add_argument('--save_path', default=None, type=str)

    return parser

def vocab_parser():
    parser = argparse.ArgumentParser()
    ### Vocab Parameters
    parser.add_argument('--inputs', required=True, type=str)
    parser.add_argument('--freq_penalty', default=0.5, type=float)
    parser.add_argument('--pad_penalty', default=0.1, type=float)
    parser.add_argument('--vocab_name', default='custom_char_dict', type=str)
    parser.add_argument('--weights_name', default='custom_char_weights', type=str)
    parser.add_argument('--save_dir', default='data', type=str)
    parser.add_argument('--max_len', default=126, type=int)

    return parser
