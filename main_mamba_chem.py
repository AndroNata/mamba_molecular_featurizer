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

