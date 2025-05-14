import random
from torch.utils.data import Dataset
from .utils import *
import torch
import random
from torch_geometric.data import Batch
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import Descriptors
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

tokenizer = AutoTokenizer.from_pretrained("./ChemBERTa")


class Molecule:
    def __init__(self, smiles: str, dim=1024, d_type='train'):
        mol = Chem.MolFromSmiles(smiles)
        self.smiles = smiles
        morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)
        self.graph = mol_to_graph_data_obj_simple(mol)
        fp_str = morgan_gen.GetFingerprint(mol).ToBitString()
        self.fp = self.convert_fp_to_tensor(fp_str)

    def convert_fp_to_tensor(self, fp_str):
        fp_list = [int(bit) for bit in fp_str]
        return torch.tensor(fp_list, dtype=torch.float)


class Reaction:
    def __init__(self, reactants: list, products: list, yield_data=None, d_type='train'):
        self.reactants = [Molecule(smiles, d_type=d_type) for smiles in reactants]
        self.products = [Molecule(smiles, d_type=d_type) for smiles in products]
        self.yield_data = yield_data


class ReactionDataset(Dataset):
    def __init__(self, data, d_type='train', train_ratio=1.0):
        self.data = self.build_reaction_dataset(data, d_type)
        self.d_type = d_type

    def build_reaction_dataset(self, data, d_type):
        reaction_dataset = []
        with tqdm(zip(data['rxn'], data['yld']), total=len(data['rxn'])) as bar:
            bar.set_description('data parsing...')
            for reaction_smiles, yield_data in bar:
                reactants, products = parse_reaction(reaction_smiles)
                reaction = Reaction(reactants, products, yield_data, d_type)
                reaction_dataset.append(reaction)
        return reaction_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        reaction = self.data[idx]

        reactants = reaction.reactants
        products = reaction.products
        yield_data = torch.tensor(reaction.yield_data, dtype=torch.float)
        return reactants, products, yield_data

    def calculate_yield_stats(self, indices=None):
        if indices is None:
            indices = range(len(self))

        yield_data_list = [self.data[i].yield_data for i in indices]

        yield_data_tensor = torch.tensor(yield_data_list)

        yield_mean = torch.mean(yield_data_tensor)
        yield_std = torch.std(yield_data_tensor)

        return yield_mean, yield_std


def collate_fn(batch):
    combined_smiles_sequences = []
    combined_graph_sequences = []
    fingerprint_sequences = []
    yield_data = []

    for reactants, products, yld in batch:
        reactant_smiles = ".".join([mol.smiles for mol in reactants])
        product_smiles = ".".join([mol.smiles for mol in products])
        combined_smiles = f"{reactant_smiles}>>{product_smiles}"
        combined_smiles_sequences.append(combined_smiles)
        reactant_graphs = [mol.graph for mol in reactants if mol.graph is not None]
        product_graphs = [mol.graph for mol in products if mol.graph is not None]
        combined_reactant_graph = Batch.from_data_list(reactant_graphs) if reactant_graphs else None
        combined_product_graph = Batch.from_data_list(product_graphs) if product_graphs else None
        combined_graph_sequences.append((combined_reactant_graph, combined_product_graph))
        # fingerprint
        torch.set_printoptions(profile="full")
        fingerprint_seq1 = torch.stack([mol.fp for mol in reactants], dim=0)
        fingerprint_seq2 = torch.stack([mol.fp for mol in products], dim=0)
        fingerprint_seq1 = torch.sum(fingerprint_seq1, dim=0)
        fingerprint_seq2 = torch.sum(fingerprint_seq2, dim=0)
        fingerprint_seq = torch.stack([fingerprint_seq1, fingerprint_seq2], dim=0)
        fingerprint_sequences.append(fingerprint_seq)
        # yield
        yield_data.append(yld)

    tokenized_smiles = tokenizer(combined_smiles_sequences, padding=True, return_tensors='pt')

    yield_data = torch.stack(yield_data)  # (128,)
    fingerprint_sequences = torch.stack(fingerprint_sequences)  # (128, 1024)
    return tokenized_smiles, combined_graph_sequences, fingerprint_sequences, yield_data  # token_smiles, graph, fingerprint, yield
