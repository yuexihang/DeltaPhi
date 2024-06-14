import wandb
import os

def wandb_init(args, resolution, method_name = "Baseline"):
    data_name  = args.CaseName.split('/')[0]
    model_name = "NORM"
    runname    = f"{method_name}-Width{args.width}-Modes{args.modes}"
    # if args.test_only:
        # wandb offline mode , data could be up load later
        # os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(project="PDE", name=runname, entity="XXXXX")
    wandb.config.update(args)
    run.tags = ( (f"{data_name}",) + (f"{model_name}",) + (f"Resolution{resolution}",) + (f"TrainNum{args.num_train}",) )

def wandb_sync(loss_dict):
    wandb.log(loss_dict)
    
    
