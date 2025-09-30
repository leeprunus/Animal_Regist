import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import trimesh
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.sparse as torch_sparse

class HeatKernelSignatureGPU:
    """
    GPU-accelerated Heat Kernel Signature implementation for 3D mesh analysis and part labeling.
    
    The HKS is based on the heat diffusion process on a Riemannian manifold.
    It provides a multi-scale signature that captures both local and global shape properties.
    
    This version uses PyTorch for GPU acceleration.
    """
    
    def __init__(self, mesh, num_eigenvalues=100, time_samples=50, device=None):
        """
        Initialize HKS calculator.
        
        Args:
            mesh: Trimesh object
            num_eigenvalues: Number of eigenvalues/eigenvectors to compute
            time_samples: Number of time scales for HKS
            device: PyTorch device ('cuda', 'cpu', or None for auto-detection)
        """
        self.mesh = mesh
        self.num_eigenvalues = num_eigenvalues
        self.time_samples = time_samples
        
        # Set up device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.n_vertices = len(self.vertices)
        
        # Convert to torch tensors on GPU
        self.vertices_gpu = torch.tensor(self.vertices, dtype=torch.float32, device=self.device)
        self.faces_gpu = torch.tensor(self.faces, dtype=torch.long, device=self.device)
        
        # Computed properties
        self.L = None  # Laplacian matrix
        self.A = None  # Area matrix
        self.eigenvalues = None
        self.eigenvectors = None
        self.hks_values = None
        self.time_scales = None
        
    
    def compute_cotangent_laplacian_gpu(self):
        """
        Compute the cotangent Laplacian matrix and vertex area matrix using GPU acceleration.
        """
        
        vertices = self.vertices_gpu
        faces = self.faces_gpu
        n_vertices = self.n_vertices
        n_faces = len(faces)
        
        # Vertex areas
        vertex_areas = torch.zeros(n_vertices, dtype=torch.float32, device=self.device)
        
        # Prepare for sparse matrix construction
        indices_list = []
        values_list = []
        
        # Process faces in batches for GPU efficiency
        batch_size = min(1000, n_faces)
        
        for batch_start in range(0, n_faces, batch_size):
            batch_end = min(batch_start + batch_size, n_faces)
            face_batch = faces[batch_start:batch_end]
            
            # Get vertex indices for this batch
            i_batch = face_batch[:, 0]
            j_batch = face_batch[:, 1]
            k_batch = face_batch[:, 2]
            
            # Get vertex positions
            vi_batch = vertices[i_batch]
            vj_batch = vertices[j_batch]
            vk_batch = vertices[k_batch]
            
            # Compute edge vectors
            e1_batch = vj_batch - vk_batch  # edge opposite to vertex i
            e2_batch = vk_batch - vi_batch  # edge opposite to vertex j
            e3_batch = vi_batch - vj_batch  # edge opposite to vertex k
            
            # Compute face areas
            face_areas_batch = 0.5 * torch.norm(torch.cross(e3_batch, -e2_batch, dim=1), dim=1)
            
            # Add 1/3 of face area to each vertex
            vertex_areas.scatter_add_(0, i_batch, face_areas_batch / 3.0)
            vertex_areas.scatter_add_(0, j_batch, face_areas_batch / 3.0)
            vertex_areas.scatter_add_(0, k_batch, face_areas_batch / 3.0)
            
            # Compute cotangent values
            def cotangent_batch(v1_batch, v2_batch):
                """Compute cotangent of angles between batched vectors"""
                dot_products = torch.sum(v1_batch * v2_batch, dim=1)
                norms1 = torch.norm(v1_batch, dim=1)
                norms2 = torch.norm(v2_batch, dim=1)
                cos_angles = dot_products / (norms1 * norms2 + 1e-10)
                cos_angles = torch.clamp(cos_angles, -1.0 + 1e-10, 1.0 - 1e-10)
                sin_angles = torch.sqrt(1.0 - cos_angles**2)
                return cos_angles / sin_angles
            
            # Cotangent weights
            cot_i_batch = cotangent_batch(-e2_batch, e3_batch)
            cot_j_batch = cotangent_batch(-e3_batch, e1_batch)
            cot_k_batch = cotangent_batch(-e1_batch, e2_batch)
            
            # Add cotangent weights to sparse matrix lists
            # Edge (j,k) opposite to vertex i
            indices_list.extend([
                torch.stack([j_batch, k_batch]),
                torch.stack([k_batch, j_batch])
            ])
            values_list.extend([
                0.5 * cot_i_batch,
                0.5 * cot_i_batch
            ])
            
            # Edge (k,i) opposite to vertex j
            indices_list.extend([
                torch.stack([k_batch, i_batch]),
                torch.stack([i_batch, k_batch])
            ])
            values_list.extend([
                0.5 * cot_j_batch,
                0.5 * cot_j_batch
            ])
            
            # Edge (i,j) opposite to vertex k
            indices_list.extend([
                torch.stack([i_batch, j_batch]),
                torch.stack([j_batch, i_batch])
            ])
            values_list.extend([
                0.5 * cot_k_batch,
                0.5 * cot_k_batch
            ])
        
        # Concatenate all indices and values
        all_indices = torch.cat(indices_list, dim=1)
        all_values = torch.cat(values_list)
        
        # Create sparse matrix for off-diagonal entries
        L_off_diagonal = torch.sparse_coo_tensor(
            all_indices, all_values, 
            size=(n_vertices, n_vertices),
            device=self.device
        ).coalesce()
        
        # Convert to dense for diagonal computation
        L_off_diagonal_dense = L_off_diagonal.to_dense()
        
        # Set diagonal entries (negative sum of off-diagonal entries)
        diagonal_values = -torch.sum(L_off_diagonal_dense, dim=1)
        L_diagonal = torch.diag(diagonal_values)
        
        # Final Laplacian matrix
        self.L = L_diagonal + L_off_diagonal_dense
        
        # Area matrix (diagonal)
        self.A = torch.diag(vertex_areas)
        
    
    def compute_eigendecomposition_gpu(self):
        """
        Compute eigenvalues and eigenvectors using GPU acceleration.
        """
        
        if self.L is None or self.A is None:
            self.compute_cotangent_laplacian_gpu()
        
        try:
            # For GPU eigendecomposition, we'll use the standard eigenvalue problem
            # A^(-1/2) * L * A^(-1/2) to convert to standard form
            
            # Compute A^(-1/2)
            area_diag = torch.diag(self.A)
            area_sqrt_inv = torch.diag(1.0 / torch.sqrt(area_diag + 1e-10))
            
            # Symmetric matrix for standard eigenvalue problem
            L_symmetric = area_sqrt_inv @ self.L @ area_sqrt_inv
            
            # Use PyTorch's eigenvalue decomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(L_symmetric)
            
            # Take only the smallest eigenvalues
            idx = torch.argsort(torch.abs(eigenvalues))[:self.num_eigenvalues]
            self.eigenvalues = eigenvalues[idx]
            eigenvectors_symmetric = eigenvectors[:, idx]
            
            # Transform back to original space: phi = A^(-1/2) * psi
            self.eigenvectors = area_sqrt_inv @ eigenvectors_symmetric
            
            # Normalize eigenvectors with respect to area measure
            area_diag = torch.diag(self.A)
            for i in range(self.num_eigenvalues):
                norm_squared = torch.sum(self.eigenvectors[:, i]**2 * area_diag)
                self.eigenvectors[:, i] /= torch.sqrt(norm_squared)
            
            
        except Exception as e:
            print(f"GPU eigendecomposition failed: {e}")
            raise
    
    def compute_hks_gpu(self, t_min=None, t_max=None):
        """
        Compute Heat Kernel Signature using GPU vectorization.
        """
        
        if self.eigenvalues is None or self.eigenvectors is None:
            self.compute_eigendecomposition_gpu()
        
        # Remove zero eigenvalue (first eigenvalue should be ~0) and negative eigenvalues
        non_zero_mask = self.eigenvalues > 1e-8
        eigenvalues = self.eigenvalues[non_zero_mask]
        eigenvectors = self.eigenvectors[:, non_zero_mask]
        
        
        if len(eigenvalues) < 2:
            non_zero_mask = torch.abs(self.eigenvalues) > 1e-10
            eigenvalues = self.eigenvalues[non_zero_mask]
            eigenvectors = self.eigenvectors[:, non_zero_mask]
            # Take absolute values to ensure positivity
            eigenvalues = torch.abs(eigenvalues)
        
        # Auto-compute time scales if not provided
        if t_min is None:
            t_min = 4.0 * np.log(10) / eigenvalues[-1].item()
        if t_max is None:
            if len(eigenvalues) > 1:
                t_max = 4.0 * np.log(10) / eigenvalues[1].item()
            else:
                t_max = t_min * 100  # fallback
        
        # Ensure positive time scales
        t_min = max(t_min, 1e-6)
        t_max = max(t_max, t_min * 10)
        
        # Generate logarithmically spaced time scales
        time_scales_np = np.logspace(np.log10(t_min), np.log10(t_max), self.time_samples)
        self.time_scales = torch.tensor(time_scales_np, dtype=torch.float32, device=self.device)
        
        # Vectorized HKS computation
        
        # Compute exponential terms: exp(-lambda_k * t) for all k, t
        # Shape: (num_eigenvalues, num_time_samples)
        exp_terms = torch.exp(-eigenvalues.unsqueeze(1) @ self.time_scales.unsqueeze(0))
        
        # Compute phi_k^2 for all vertices and eigenvalues
        # Shape: (num_vertices, num_eigenvalues)
        phi_squared = eigenvectors ** 2
        
        # Vectorized HKS computation: HKS(v,t) = sum_k exp(-lambda_k * t) * phi_k(v)^2
        # Shape: (num_vertices, num_time_samples)
        self.hks_values = phi_squared @ exp_terms
        
        # Add small epsilon to prevent zeros in log computations
        self.hks_values = torch.clamp(self.hks_values, min=1e-10)
        
    
    def extract_multiscale_features_gpu(self):
        """
        Extract multi-scale features from HKS using GPU acceleration.
        """
        if self.hks_values is None:
            self.compute_hks_gpu()
        
        # Use raw HKS values as features
        features = self.hks_values.clone()
        
        # Check for NaN or infinite values
        if torch.any(torch.isnan(features)) or torch.any(torch.isinf(features)):
            print("Warning: NaN or Inf values detected in HKS. Applying correction...")
            features = torch.nan_to_num(features, nan=1e-10, posinf=1e10, neginf=1e-10)
        
        # Compute derivatives using GPU
        hks_derivatives = torch.gradient(features, dim=1)[0]
        
        # Check derivatives
        if torch.any(torch.isnan(hks_derivatives)) or torch.any(torch.isinf(hks_derivatives)):
            hks_derivatives = torch.nan_to_num(hks_derivatives, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Local extrema detection on GPU
        local_maxima = torch.zeros_like(features)
        local_minima = torch.zeros_like(features)
        
        # Vectorized extrema detection
        if features.shape[1] > 2:  # Only if we have enough time samples
            left_comparison = features[:, 1:-1] > features[:, :-2]
            right_comparison = features[:, 1:-1] > features[:, 2:]
            local_maxima[:, 1:-1] = (left_comparison & right_comparison).float()
            
            left_comparison_min = features[:, 1:-1] < features[:, :-2]
            right_comparison_min = features[:, 1:-1] < features[:, 2:]
            local_minima[:, 1:-1] = (left_comparison_min & right_comparison_min).float()
        
        # Combine features
        features = torch.cat([
            features,           # Raw HKS values
            local_maxima,       # Local maxima
            local_minima,       # Local minima
            hks_derivatives     # HKS derivatives
        ], dim=1)
        
        # Final check for problematic values
        if torch.any(torch.isnan(features)) or torch.any(torch.isinf(features)):
            print("Warning: NaN or Inf values detected in final features. Applying final correction...")
            features = torch.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        print(f"Extracted features shape: {features.shape}")
        print(f"Features value range: [{torch.min(features):.6f}, {torch.max(features):.6f}]")
        return features
    
    def cluster_vertices_gpu(self, n_clusters=8, features=None):
        """
        Cluster vertices based on HKS features.
        """
        print(f"Clustering vertices into {n_clusters} parts...")
        
        if features is None:
            features = self.extract_multiscale_features_gpu()
        
        # Move features to CPU for sklearn
        features_cpu = features.cpu().numpy()
        
        # Standardize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_cpu)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_normalized)
        
        # Print clustering statistics
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"Clustering results:")
        for label, count in zip(unique_labels, counts):
            print(f"  Part {label}: {count} vertices ({100*count/self.n_vertices:.1f}%)")
        
        return labels
    
    def save_labeled_mesh(self, labels, output_path, colormap='tab10'):
        """
        Save mesh with vertex colors based on labels.
        """
        print(f"Saving labeled mesh to {output_path}")
        
        # Create colored mesh
        mesh_colored = self.mesh.copy()
        
        # Generate colors for labels
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
        n_labels = len(np.unique(labels))
        colors = []
        
        for label in labels:
            color = cmap(label / max(n_labels - 1, 1))[:3]  # RGB only
            colors.append([int(c * 255) for c in color])
        
        mesh_colored.visual.vertex_colors = colors
        
        # Save mesh
        mesh_colored.export(output_path)
        print(f"Labeled mesh saved successfully")
    


class TipDetector(HeatKernelSignatureGPU):
    """
    Specialized Heat Kernel Signature implementation for detecting tips and extremities
    in 3D models without anatomical classification.
    
    Extends the base HKS implementation with methods for general tip detection.
    """
    
    def __init__(self, mesh, num_eigenvalues=100, time_samples=50, device=None):
        super().__init__(mesh, num_eigenvalues, time_samples, device)
        
        # Additional properties for tip detection
        self.vertex_curvatures = None
        self.hks_extremality = None
        self.detected_tips = None
        
    def compute_vertex_curvatures(self):
        """
        Compute mean curvature at each vertex using discrete operators.
        """
        
        vertices = self.vertices
        faces = self.faces
        n_vertices = self.n_vertices
        
        # Initialize curvature array
        mean_curvatures = np.zeros(n_vertices)
        vertex_areas = np.zeros(n_vertices)
        
        # Compute face normals and areas
        face_normals = []
        face_areas = []
        
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge1, edge2 = v1 - v0, v2 - v0
            normal = np.cross(edge1, edge2)
            area = 0.5 * np.linalg.norm(normal)
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            
            face_normals.append(normal)
            face_areas.append(area)
        
        face_normals = np.array(face_normals)
        face_areas = np.array(face_areas)
        
        # Build vertex-face adjacency
        vertex_faces = [[] for _ in range(n_vertices)]
        for face_idx, face in enumerate(faces):
            for vertex_idx in face:
                vertex_faces[vertex_idx].append(face_idx)
        
        # Compute mean curvature using angle defect method
        for v_idx in range(n_vertices):
            if not vertex_faces[v_idx]:
                continue
                
            # Get neighboring faces
            neighbor_faces = vertex_faces[v_idx]
            
            # Compute vertex normal as area-weighted average of face normals
            vertex_normal = np.zeros(3)
            total_area = 0
            
            for face_idx in neighbor_faces:
                vertex_normal += face_normals[face_idx] * face_areas[face_idx]
                total_area += face_areas[face_idx] / 3.0  # Vertex gets 1/3 of face area
            
            if total_area > 1e-10:
                vertex_normal /= np.linalg.norm(vertex_normal + 1e-10)
                vertex_areas[v_idx] = total_area
                
                # Compute mean curvature using discrete Laplace-Beltrami
                # This is a basic version - for full implementation, use cotangent weights
                laplace_beltrami = np.zeros(3)
                weight_sum = 0
                
                for face_idx in neighbor_faces:
                    face = faces[face_idx]
                    for other_v in face:
                        if other_v != v_idx:
                            edge_vector = vertices[other_v] - vertices[v_idx]
                            weight = 1.0  # Basic - could use cotangent weights
                            laplace_beltrami += weight * edge_vector
                            weight_sum += weight
                
                if weight_sum > 1e-10:
                    laplace_beltrami /= weight_sum
                    mean_curvatures[v_idx] = np.dot(laplace_beltrami, vertex_normal)
        
        self.vertex_curvatures = mean_curvatures
        
        return mean_curvatures
    
    def compute_hks_extremality_features(self):
        """
        Compute extremality features from HKS that are useful for tip detection.
        """
        if self.hks_values is None:
            self.compute_hks_gpu()
        
        
        hks_cpu = self.hks_values.cpu().numpy()
        n_vertices, n_times = hks_cpu.shape
        
        # Feature 1: HKS peak prominence across time scales
        hks_max_values = np.max(hks_cpu, axis=1)
        hks_min_values = np.min(hks_cpu, axis=1)
        hks_range = hks_max_values - hks_min_values
        
        # Feature 2: HKS persistence (how long features persist across scales)
        hks_persistence = np.zeros(n_vertices)
        for v in range(n_vertices):
            hks_curve = hks_cpu[v, :]
            # Count how many time scales this vertex has above-average HKS
            mean_hks = np.mean(hks_curve)
            persistence = np.sum(hks_curve > mean_hks)
            hks_persistence[v] = persistence / n_times
        
        # Feature 3: HKS gradient magnitude (rate of change across scales)
        hks_gradients = np.gradient(hks_cpu, axis=1)
        hks_gradient_magnitude = np.mean(np.abs(hks_gradients), axis=1)
        
        # Feature 4: Early-time HKS values (local geometric features)
        early_time_hks = np.mean(hks_cpu[:, :n_times//4], axis=1)
        
        # Feature 5: Late-time HKS values (global geometric features)
        late_time_hks = np.mean(hks_cpu[:, 3*n_times//4:], axis=1)
        
        # Combine all features
        extremality_features = np.column_stack([
            hks_range,
            hks_persistence,
            hks_gradient_magnitude,
            early_time_hks,
            late_time_hks,
            hks_max_values
        ])
        
        self.hks_extremality = extremality_features
        
        return extremality_features
    
    def detect_tips(self, curvature_threshold=0.7, hks_threshold=0.85, min_distance_ratio=0.05):
        """
        Detect all tips and extremities based on HKS and curvature without classification.
        
        Args:
            curvature_threshold: Percentile threshold for high curvature vertices
            hks_threshold: Percentile threshold for distinctive HKS signatures
            min_distance_ratio: Minimum distance between tips as ratio of bounding box diagonal
        """
        
        if self.vertex_curvatures is None:
            self.compute_vertex_curvatures()
        
        if self.hks_extremality is None:
            self.compute_hks_extremality_features()
        
        # Normalize features
        curvatures = self.vertex_curvatures
        extremality = self.hks_extremality
        
        # Identify high curvature vertices (potential tips/extremities)
        curvature_percentile = np.percentile(np.abs(curvatures), curvature_threshold * 100)
        high_curvature_mask = np.abs(curvatures) > curvature_percentile
        
        # Identify vertices with distinctive HKS signatures
        hks_prominence = extremality[:, 0]  # HKS range
        hks_percentile = np.percentile(hks_prominence, hks_threshold * 100)
        distinctive_hks_mask = hks_prominence > hks_percentile
        
        # Additional criteria: vertices with high early-time HKS (sharp local features)
        early_hks = extremality[:, 3]
        early_hks_percentile = np.percentile(early_hks, 85)
        sharp_local_mask = early_hks > early_hks_percentile
        
        # Combine criteria: vertices that meet multiple criteria
        candidate_mask = (high_curvature_mask & distinctive_hks_mask) | \
                        (high_curvature_mask & sharp_local_mask) | \
                        (distinctive_hks_mask & sharp_local_mask)
        
        candidate_indices = np.where(candidate_mask)[0]
        
        # Additional filtering: remove vertices that are too close to each other
        if len(candidate_indices) > 0:
            candidate_positions = self.vertices[candidate_indices]
            
            # Calculate minimum distance based on mesh size
            bbox_size = np.max(self.vertices, axis=0) - np.min(self.vertices, axis=0)
            bbox_diagonal = np.linalg.norm(bbox_size)
            min_distance = bbox_diagonal * min_distance_ratio
            
            # Use clustering to group nearby candidates and keep the best one from each cluster
            if len(candidate_indices) > 1:
                distances = cdist(candidate_positions, candidate_positions)
                
                # Compute combined scores for ranking
                curvature_scores = np.abs(curvatures[candidate_indices])
                hks_scores = hks_prominence[candidate_indices]
                early_hks_scores = early_hks[candidate_indices]
                
                # Normalize scores to [0, 1] range
                curvature_scores = (curvature_scores - np.min(curvature_scores)) / (np.max(curvature_scores) - np.min(curvature_scores) + 1e-10)
                hks_scores = (hks_scores - np.min(hks_scores)) / (np.max(hks_scores) - np.min(hks_scores) + 1e-10)
                early_hks_scores = (early_hks_scores - np.min(early_hks_scores)) / (np.max(early_hks_scores) - np.min(early_hks_scores) + 1e-10)
                
                combined_scores = curvature_scores + hks_scores + early_hks_scores
                
                # Non-maximum suppression: keep only local maxima
                filtered_candidates = []
                used_mask = np.zeros(len(candidate_indices), dtype=bool)
                
                # Sort by combined score (highest first)
                sorted_indices = np.argsort(combined_scores)[::-1]
                
                for i in sorted_indices:
                    if used_mask[i]:
                        continue
                    
                    # Add this candidate
                    filtered_candidates.append(candidate_indices[i])
                    
                    # Mark nearby candidates as used
                    nearby_mask = distances[i] < min_distance
                    used_mask[nearby_mask] = True
                
                candidate_indices = np.array(filtered_candidates)
        
        self.detected_tips = candidate_indices
        
        
        return candidate_indices
    
    def save_tips_mesh(self, output_path):
        """
        Save mesh with detected tips highlighted in red.
        """
        print(f"Saving tips mesh to {output_path}")
        
        mesh_colored = self.mesh.copy()
        colors = np.ones((self.n_vertices, 3)) * 128  # Gray default
        
        # Color detected tips in bright red
        if self.detected_tips is not None and len(self.detected_tips) > 0:
            for vertex_idx in self.detected_tips:
                colors[vertex_idx] = [255, 0, 0]  # Red
        
        mesh_colored.visual.vertex_colors = colors.astype(np.uint8)
        mesh_colored.export(output_path)
        print(f"Tips mesh saved successfully with {len(self.detected_tips)} tips highlighted")
    
    def save_tips_npy(self, output_path):
        """
        Save detected tip positions as numpy array in the same format as predict3d.py output.
        Format: numpy array with shape (num_tips, 3) where each row is [x, y, z] coordinates.
        """
        if self.detected_tips is None or len(self.detected_tips) == 0:
            print("No tips detected to save")
            return
        
        # Get the 3D positions of detected tips
        tip_positions = self.vertices[self.detected_tips]  # Shape: (num_tips, 3)
        
        # Save as numpy array
        np.save(output_path, tip_positions)
        
        return tip_positions
    

