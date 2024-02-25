from rdkit import RDLogger

from src import SymbolPredictor, ZinkDataModule

from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, SaveConfigCallback


def cli_main():
    LightningCLI(model_class=SymbolPredictor(),
        datamodule_class=ZinkDataModule,
        save_config_callback=SaveConfigCallback)


if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')
    cli_main()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba-130m")
    parser.add_argument("--tokenizer", type=str, default="chem_tokenizer.json")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="mamba-reactions-test")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--data_path", type=str, default="two_rxns.txt")
    parser.add_argument("--num_epochs", type=int, default=5)
    args = parser.parse_args()

    run(args)
