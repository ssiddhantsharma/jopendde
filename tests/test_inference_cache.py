from types import SimpleNamespace

from jopendde.inference import _set_asset_cache_dir


def test_set_asset_cache_dir(tmp_path):
    configs = SimpleNamespace(data=SimpleNamespace())

    _set_asset_cache_dir(configs, tmp_path)

    assert configs.load_checkpoint_dir == str(tmp_path / "checkpoint")
    assert configs.data.ccd_components_file == str(
        tmp_path / "common" / "components.cif"
    )
    assert configs.data.ccd_components_rdkit_mol_file == str(
        tmp_path / "common" / "components.cif.rdkit_mol.pkl"
    )
