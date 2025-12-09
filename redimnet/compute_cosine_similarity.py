import os
import torch
import torch.nn.functional as F

class CosineSimilarity:
    def __init__(self, folder_a, folder_b):
        self.folder_a = folder_a
        self.folder_b = folder_b

    def process(self):
        # 1. Get list of files in both folders
        files_a = set(os.listdir(self.folder_a))
        files_b = set(os.listdir(self.folder_b))

        # 2. Find files that exist in BOTH folders (intersection)
        common_files = files_a.intersection(files_b)

        score_total = 0
        num_processed_files = 0

        print(f"Found {len(common_files)} matching files. Computing similarity...")

        for filename in common_files:
            path_a = os.path.join(self.folder_a, filename)
            path_b = os.path.join(self.folder_b, filename)

            try:
                # 3. Load the tensors (force to CPU to avoid device mismatch errors)
                tensor_a = torch.load(path_a, map_location='cpu')
                tensor_b = torch.load(path_b, map_location='cpu')

                # 4. Compute Cosine Similarity
                # We flatten them to 1D vectors to ensure shapes match for the comparison
                similarity = F.cosine_similarity(tensor_a.flatten(), tensor_b.flatten(), dim=0)

                # Store the result (convert tensor to standard float)
                score = similarity.item()
                score_total += score
                num_processed_files += 1

                print(f"{filename}: {score:.4f}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return score_total / num_processed_files

if __name__ == "__main__":
    folder_a = "/ocean/projects/cis220031p/shared/11785-project/mlsp-csm-project/redimnet/output/inference"
    folder_b = "/ocean/projects/cis220031p/shared/11785-project/mlsp-csm-project/redimnet/output/original"
    comparator = CosineSimilarity(folder_a, folder_b)
    avg_cos_sim = comparator.process()
    print(f"Average cosine similarity {avg_cos_sim}")

    file_1 = f"{folder_a}/POD1000000005_S0000005.pt"
    file_2 = f"{folder_b}/POD1000000005_S0000005.pt"
    tensor_a = torch.load(file_1, map_location='cpu')
    tensor_b = torch.load(file_2, map_location='cpu')
    print(f"Test cos sim {F.cosine_similarity(tensor_a.flatten(), tensor_b.flatten(), dim=0)}")

