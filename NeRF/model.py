import torch
import torch.nn as nn

class Voxels(nn.Module):
    
    def __init__(self, nb_voxels=100, scale=1, device='cpu'):
        super(Voxels, self).__init__()
        
        self.voxels = torch.nn.Parameter(torch.rand((nb_voxels, nb_voxels, nb_voxels, 4), 
                                                    device=device, requires_grad=True))
        
        self.nb_voxels = nb_voxels
        self.device = device
        self.scale = scale
        
    def forward(self, xyz, d):
        
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        
        cond = (x.abs() < (self.scale / 2)) & (y.abs() < (self.scale / 2)) & (z.abs() < (self.scale / 2))
        
        indx = (x[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)
        indy = (y[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)
        indz = (z[cond] / (self.scale / self.nb_voxels) + self.nb_voxels / 2).type(torch.long)
        
        colors_and_densities = torch.zeros((xyz.shape[0], 4), device=xyz.device)
        colors_and_densities[cond, :3] = self.voxels[indx, indy, indz, :3]
        colors_and_densities[cond, -1] = self.voxels[indx, indy, indz, -1]
         
        return torch.sigmoid(colors_and_densities[:, :3]), torch.relu(colors_and_densities[:, -1:])
        
    
    def intersect(self, x, d):
        return self.forward(x, d)
    
    
class Nerf(nn.Module):
    
    def __init__(self, Lpos=10, Ldir=4, hidden_dim=256):
        super(Nerf, self).__init__()
        
        self.block1 = nn.Sequential(nn.Linear(Lpos * 6 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        
        self.block2 = nn.Sequential(nn.Linear(hidden_dim + Lpos * 6 + 3, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1),)
        
        self.rgb_head = nn.Sequential(nn.Linear(hidden_dim + Ldir * 6 + 3, hidden_dim // 2), nn.ReLU(),
                                      nn.Linear(hidden_dim // 2, 3), nn.Sigmoid())
        
        self.Lpos = Lpos
        self.Ldir = Ldir
        
    def positional_encoding(self, x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)
            
                                    
        
    def forward(self, xyz, d):
        
        x_emb = self.positional_encoding(xyz, self.Lpos) # [batch_size, Lpos * 6 + 3]
        d_emb = self.positional_encoding(d, self.Ldir) # [batch_size, Ldir * 6 + 3]
        
        h = self.block1(x_emb) # [batch_size, hidden_dim]
        h = self.block2(torch.cat((h, x_emb), dim=1)) # [batch_size, hidden_dim + 1]
        sigma = h[:, -1]
        h = h[:, :-1] # [batch_size, hidden_dim]
        c = self.rgb_head(torch.cat((h, d_emb), dim=1))
        
        return c, torch.relu(sigma)
        
    
    def intersect(self, x, d):
        return self.forward(x, d)
    

class NeRFWithPoses(nn.Module):
    def __init__(self, Lpos=10, Ldir=4, hidden_dim=256, num_images=0):
        super(NeRFWithPoses, self).__init__()
        # Use the regular NeRF model
        self.nerf = Nerf(Lpos=Lpos, Ldir=Ldir, hidden_dim=hidden_dim)
        
        # Add learnable pose parameters (SE3) for each image
        if num_images > 0:
            # Initialize with zeros
            self.pose_refinement = nn.Embedding(num_images, 6)
            nn.init.zeros_(self.pose_refinement.weight)
        else:
            self.pose_refinement = None
    
    def forward(self, xyz, d):
        return self.nerf(xyz, d)
    
    def intersect(self, x, d):
        return self.nerf.intersect(x, d)
    
    def get_refined_pose(self, pose, img_idx):
        """Refine a camera pose using the learned parameters"""
        if self.pose_refinement is None:
            return pose
        
        # Get the pose refinement parameters for this image
        device = self.pose_refinement.weight.device
        idx_tensor = torch.tensor([img_idx], device=device)
        refinement = self.pose_refinement(idx_tensor)
        
        # Convert axis-angle + translation (6D) to SE(3) transformation
        rotation = refinement[0, :3]  # axis-angle
        translation = refinement[0, 3:6]
        
        # Convert axis-angle to rotation matrix (simplified)
        theta = torch.norm(rotation)
        if theta < 1e-6:
            R = torch.eye(3, device=device)
        else:
            axis = rotation / theta
            K = torch.zeros((3, 3), device=device)
            K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -axis[2], axis[1], axis[2], -axis[0], -axis[1], axis[0]
            R = torch.eye(3, device=device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
        
        # Create the refinement transform
        refinement_mat = torch.eye(4, device=device)
        refinement_mat[:3, :3] = R
        refinement_mat[:3, 3] = translation
        
        # Apply refinement to the original pose
        pose_tensor = torch.tensor(pose, dtype=torch.float32, device=device)
        refined_pose = refinement_mat @ pose_tensor
        
        # Use detach() before converting to numpy to remove gradients
        return refined_pose.detach().cpu().numpy()