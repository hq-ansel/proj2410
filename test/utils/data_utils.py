import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from accelerate import Accelerator

class CustomDataset(Dataset):
    def __init__(self, rawdata, teacher_model):
        self.rawdata = rawdata  # List[tensor[seq_len]]
        self.teacher_model = teacher_model
        self.target_idx = None
        self.accelerator = Accelerator()  # Initialize the accelerator

    def set_target_idx(self, idx):
        """Set the index of the target layer to extract from the teacher model output."""
        self.target_idx = idx
        # Adjust the teacher model to output the desired target layer
        self.teacher_model.set_target_idx(idx)

    def prepare_all_data(self):
        """Prepare all data using the teacher model and return results ready for DataLoader."""
        all_inputs = []
        all_outputs = []
        
        # Automatically place the model on the appropriate devices
        self.teacher_model = self.accelerator.prepare(self.teacher_model)

        # Process data in batches to improve efficiency
        dataloader = DataLoader(self.rawdata, batch_size=32)
        for batch in self.accelerator.prepare(dataloader):
            with torch.no_grad():
                output = self.teacher_model(batch)
                output = output.to('cpu')  # Move output to CPU
            all_inputs.append(batch.cpu())  # Ensure inputs are also on CPU
            all_outputs.append(output)

        self.teacher_model.to('cpu')  # Move teacher model back to CPU after processing

        # Concatenate all inputs and outputs
        all_inputs = torch.cat(all_inputs, dim=0)
        all_outputs = torch.cat(all_outputs, dim=0)
        
        # Create a TensorDataset that can be used with a DataLoader
        return TensorDataset(all_inputs, all_outputs)

    def __len__(self):
        return len(self.rawdata)

    def __getitem__(self, idx):
        # Retrieve the raw input
        return self.rawdata[idx]
